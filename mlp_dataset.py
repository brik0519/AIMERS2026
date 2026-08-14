"""PyTorch Dataset for arrays created by mlp_preprocess.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class TabularNpyDataset(Dataset):
    """Memory-mapped numeric/categorical arrays with optional targets."""

    def __init__(self, split_dir: str | Path) -> None:
        self.split_dir = Path(split_dir)
        self.numeric = np.load(
            self.split_dir / "numeric.npy", mmap_mode="r"
        )
        self.categorical = np.load(
            self.split_dir / "categorical.npy", mmap_mode="r"
        )
        self.row_ids = np.load(
            self.split_dir / "row_id.npy", mmap_mode="r"
        )

        target_path = self.split_dir / "target.npy"
        self.targets = (
            np.load(target_path, mmap_mode="r")
            if target_path.exists()
            else None
        )

        lengths = {
            len(self.numeric),
            len(self.categorical),
            len(self.row_ids),
        }
        if self.targets is not None:
            lengths.add(len(self.targets))
        if len(lengths) != 1:
            raise ValueError(f"Array length mismatch in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.numeric)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "numeric": torch.tensor(
                self.numeric[index], dtype=torch.float32
            ),
            "categorical": torch.tensor(
                self.categorical[index], dtype=torch.long
            ),
            "row_id": str(self.row_ids[index]),
        }
        if self.targets is not None:
            item["target"] = torch.tensor(
                self.targets[index], dtype=torch.float32
            )
        return item
