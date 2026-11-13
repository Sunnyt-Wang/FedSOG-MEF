#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def _parse_client(spec: str) -> tuple[Path, float]:
    if ":" not in spec:
        return Path(spec), 1.0
    path, weight = spec.rsplit(":", 1)
    return Path(path), float(weight)


def _normalize_weights(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("FedAvg weights must sum to a positive value")
    return [w / total for w in weights]


def _copy_optional_state(first_ckpt: Path, out_dir: Path) -> None:
    # Evaluation only needs model weights, but Accelerate's load_state can also
    # look for registered custom checkpoint payloads. Copying the first client's
    # custom payload keeps the averaged checkpoint compatible with the existing
    # evaluator without inventing a new loading path.
    for name in ("custom_checkpoint_0.pkl",):
        src = first_ckpt / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def fedavg(client_specs: list[str], out_dir: Path) -> dict:
    clients = [_parse_client(spec) for spec in client_specs]
    ckpts = [path for path, _weight in clients]
    weights = _normalize_weights([weight for _path, weight in clients])
    missing = [str(path / "model.safetensors") for path in ckpts if not (path / "model.safetensors").exists()]
    if missing:
        raise FileNotFoundError("missing model.safetensors:\n" + "\n".join(missing))

    states = [load_file(str(path / "model.safetensors"), device="cpu") for path in ckpts]
    keys = set(states[0])
    for i, state in enumerate(states[1:], start=1):
        if set(state) != keys:
            only_first = sorted(keys - set(state))[:10]
            only_other = sorted(set(state) - keys)[:10]
            raise ValueError(
                f"checkpoint {ckpts[i]} has different keys; "
                f"missing={only_first}, extra={only_other}"
            )

    averaged: dict[str, torch.Tensor] = {}
    for key in sorted(keys):
        tensors = [state[key] for state in states]
        ref = tensors[0]
        if not torch.is_floating_point(ref):
            averaged[key] = ref
            continue
        if any(t.shape != ref.shape for t in tensors):
            raise ValueError(f"shape mismatch for key {key}")
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for tensor, weight in zip(tensors, weights, strict=True):
            acc.add_(tensor.float(), alpha=weight)
        averaged[key] = acc.to(dtype=ref.dtype)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(averaged, str(out_dir / "model.safetensors"))
    _copy_optional_state(ckpts[0], out_dir)
    manifest = {
        "mode": "fedavg_model_safetensors",
        "clients": [{"checkpoint": str(path), "weight": weight} for path, weight in zip(ckpts, weights, strict=True)],
        "num_tensors": len(averaged),
    }
    (out_dir / "fedavg_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Average compatible LLM4Bin client checkpoints.")
    parser.add_argument("--client", action="append", required=True, help="checkpoint_dir or checkpoint_dir:weight")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = fedavg(args.client, args.out_dir)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
