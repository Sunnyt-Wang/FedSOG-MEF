"""Checkpoint manager.

Wraps :meth:`accelerate.Accelerator.save_state` / ``load_state`` with:

- A JSON manifest of every saved checkpoint (step, epoch, val_metric, dir)
- Rotation (keep_last_n + optional keep_best)
- ``--resume_from=auto`` resolution to the latest checkpoint

The manifest itself is registered with the accelerator via
``register_for_checkpointing`` so it round-trips with the rest of the state.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CkptEntry:
    step: int
    epoch: int
    dir: str
    val_metric: float | None = None
    is_best: bool = False


class CkptManifest:
    """Mutable list of :class:`CkptEntry`; supports Accelerate's checkpoint hook."""

    def __init__(self) -> None:
        self.entries: list[CkptEntry] = []

    def state_dict(self) -> dict[str, Any]:
        return {"entries": [asdict(e) for e in self.entries]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.entries = [CkptEntry(**e) for e in state.get("entries", [])]

    def add(self, entry: CkptEntry) -> None:
        self.entries.append(entry)

    def latest(self) -> CkptEntry | None:
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.step)


class CheckpointManager:
    def __init__(
        self,
        root_dir: str | Path,
        keep_last_n: int = 3,
        keep_best: bool = True,
        higher_is_better: bool = True,
    ):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.keep_best = keep_best
        self.higher_is_better = higher_is_better
        self.manifest = CkptManifest()

    # --- save / load -------------------------------------------------

    def save(self, accelerator: Any, step: int, epoch: int, val_metric: float | None = None) -> Path:
        ckpt_dir = self.root / f"step_{step:09d}"
        # Add the entry BEFORE save_state so the accelerator's manifest-checkpoint hook
        # captures the new entry. Saving in the reverse order means the manifest persisted
        # inside this checkpoint would not reference itself, breaking resume lookups.
        entry = CkptEntry(step=step, epoch=epoch, dir=str(ckpt_dir), val_metric=val_metric)
        self.manifest.add(entry)
        accelerator.save_state(str(ckpt_dir))
        is_main = getattr(accelerator, "is_main_process", True)
        self._rotate(delete_files=is_main)
        if is_main:
            self._write_manifest_file()
        if hasattr(accelerator, "wait_for_everyone"):
            accelerator.wait_for_everyone()
        return ckpt_dir

    def load(self, accelerator: Any, target: str = "auto") -> CkptEntry | None:
        ckpt_dir = self._resolve_target(target)
        if ckpt_dir is None:
            return None
        # strict=False: our model.safetensors only contains trainable gnn.* / projector.*
        # weights; the frozen LLM keys are absent. accelerate forwards this kwarg to
        # safetensors.torch.load_model.
        accelerator.load_state(str(ckpt_dir), strict=False)
        # After load, manifest has been restored from disk via the registered hook.
        resolved = ckpt_dir.resolve()
        return next((e for e in self.manifest.entries if Path(e.dir).resolve() == resolved), None)

    # --- helpers -----------------------------------------------------

    def _resolve_target(self, target: str) -> Path | None:
        if target == "auto":
            latest = self.manifest.latest()
            if latest is None:
                # Manifest is empty (fresh run) but a previous run may have left state on disk.
                # Look for the highest step_* directory if a manifest file exists.
                manifest_path = self.root / "manifest.json"
                if manifest_path.exists():
                    data = json.loads(manifest_path.read_text())
                    if data.get("entries"):
                        return Path(max(data["entries"], key=lambda e: e["step"])["dir"])
                return None
            return Path(latest.dir).resolve()
        path = Path(target)
        return path.resolve() if path.exists() else None

    def _write_manifest_file(self) -> None:
        (self.root / "manifest.json").write_text(json.dumps(self.manifest.state_dict(), indent=2))

    def _rotate(self, delete_files: bool = True) -> None:
        """Keep at most ``keep_last_n`` recent + 1 best (if enabled)."""
        entries = self.manifest.entries
        if not entries:
            return

        keepers: set[str] = set()
        # Recent
        recent = sorted(entries, key=lambda e: e.step)[-self.keep_last_n :]
        keepers.update(e.dir for e in recent)
        # Best (only if any have metrics)
        if self.keep_best:
            scored = [e for e in entries if e.val_metric is not None]
            if scored:
                best = max(scored, key=lambda e: e.val_metric) if self.higher_is_better else min(
                    scored, key=lambda e: e.val_metric
                )
                for e in entries:
                    e.is_best = e is best
                keepers.add(best.dir)

        # Delete everything not in keepers
        survivors: list[CkptEntry] = []
        for e in entries:
            if e.dir in keepers:
                survivors.append(e)
            else:
                p = Path(e.dir)
                if delete_files and p.exists():
                    shutil.rmtree(p)
        self.manifest.entries = survivors
