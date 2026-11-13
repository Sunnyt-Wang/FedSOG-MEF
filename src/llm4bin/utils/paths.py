from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def require_exists(path: str | os.PathLike, label: str = "") -> Path:
    p = Path(path)
    if not p.exists():
        prefix = f"[{label}] " if label else ""
        raise FileNotFoundError(f"{prefix}path does not exist: {p}")
    return p


def require_all_exist(paths: Iterable[tuple[str, str | os.PathLike]]) -> None:
    missing: list[str] = []
    for label, p in paths:
        if p is None:
            continue
        if not Path(p).exists():
            missing.append(f"  {label}: {p}")
    if missing:
        raise FileNotFoundError("missing required paths:\n" + "\n".join(missing))
