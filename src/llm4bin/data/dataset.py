from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class VulnDataset(Dataset):
    """Dataset for the vulnerability classification task.

    Returns dicts so the collate function does not depend on column order.
    """

    REQUIRED_COLS = ("obj_file", "fva", "func_name", "label", "CWE")

    def __init__(self, csv_path: str | Path, subset_size: int = -1, seed: int = 42):
        df = pd.read_csv(csv_path)
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path}: missing columns {missing}")
        if subset_size > 0 and subset_size < len(df):
            df = df.sample(n=subset_size, random_state=seed).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "obj_file": row["obj_file"],
            "fva": row["fva"],
            "func_name": row["func_name"],
            "label": int(row["label"]),
            "CWE": row["CWE"],
        }


class SourceDataset(Dataset):
    """Dataset for the binary-to-source recovery task.

    The CSV provides ``idb_path / fva / func_name``; the source code is looked up
    in a separate ``src_pool`` pickle keyed by ``func_name``.
    """

    REQUIRED_COLS = ("idb_path", "fva", "func_name")

    def __init__(
        self,
        csv_path: str | Path,
        src_pool: dict[str, str] | None = None,
        subset_size: int = -1,
        seed: int = 42,
    ):
        df = pd.read_csv(csv_path)
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path}: missing columns {missing}")
        if subset_size > 0 and subset_size < len(df):
            df = df.sample(n=subset_size, random_state=seed).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        self.df = df
        self.src_pool = src_pool or {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "idb_path": row["idb_path"],
            "fva": row["fva"],
            "func_name": row["func_name"],
        }
