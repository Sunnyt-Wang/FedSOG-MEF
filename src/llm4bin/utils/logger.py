from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_CONFIGURED: set[str] = set()


def _is_main_process() -> bool:
    # Works before accelerate is imported; LOCAL_RANK / RANK set by launcher.
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try:
        return int(rank) == 0
    except ValueError:
        return True


def get_logger(name: str, log_dir: str | os.PathLike | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console: main process only (avoid duplicated output across ranks).
    if _is_main_process():
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    else:
        logger.setLevel(logging.WARNING)

    # File handler: every rank writes its own file (helps diagnose DDP issues).
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        fh = logging.FileHandler(log_dir / f"{name}.rank{rank}.log", mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _CONFIGURED.add(name)
    return logger
