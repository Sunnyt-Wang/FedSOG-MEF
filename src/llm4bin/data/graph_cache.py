from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from ..utils.logger import get_logger

_logger = get_logger(__name__)

# Process-local cache. Same pkl reopened many times (smoke + full) reuses memory.
_CACHE: dict[str, Any] = {}


def load_graph_dict(path: str | Path) -> Any:
    """Load the ``defaultdict[obj_file][fva] -> {graph, opc}`` pickle.

    The file is ~2.8GB for C; we cache it per process and never reload.
    """
    key = str(Path(path).resolve())
    if key in _CACHE:
        return _CACHE[key]
    _logger.info(f"loading graph cache: {key}")
    with open(key, "rb") as f:
        fd = pickle.load(f)
    _CACHE[key] = fd
    _logger.info(f"graph cache loaded: {len(fd)} obj_files")
    return fd


def clear_cache() -> None:
    _CACHE.clear()
