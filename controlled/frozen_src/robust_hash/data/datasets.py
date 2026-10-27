from pathlib import Path
from typing import Dict, Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from robust_hash.data.splits import load_split_manifest, filter_by_split
from robust_hash.data.transforms import build_image_tensor
from robust_hash.utils.paths import get_raw_data_root


def resolve_image_path(row: pd.Series, dataset: str) -> str:
    """
    优先使用 manifest 里的 abs_path；
    如果没有，再用 raw_root / dataset / rel_path 组装。
    """
    abs_path = row.get("abs_path", None)
    if isinstance(abs_path, str) and len(abs_path) > 0:
        archived_path = Path(abs_path)
        if archived_path.exists():
            return str(archived_path)

    rel_path = row["rel_path"]
    path = get_raw_data_root() / dataset / rel_path
    if path.exists():
        return str(path)

    raise FileNotFoundError(
        "Image not found. The split manifest may contain an archived server path. "
        f"Tried archived abs_path={abs_path!r} and current data root path={path}. "
        "Set DA_HCR_DATA_ROOT to the directory containing raw/<dataset>/... ."
    )


_resolve_abs_path = resolve_image_path


class PairTrainDataset(Dataset):
    """
    训练集：
    返回 clean 图 和 paired degraded 图。
    """

    def __init__(
        self,
        dataset: str,
        split_seed: int,
        image_size: int = 224,
        paired_severity: str = "medium",
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.split_seed = split_seed
        self.image_size = image_size
        self.paired_severity = paired_severity

        df = load_split_manifest(dataset=dataset, split_seed=split_seed)
        df = filter_by_split(df, "train")
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        image_path = _resolve_abs_path(row, self.dataset)

        clean_image = build_image_tensor(
            image_path=image_path,
            image_size=self.image_size,
            severity="clean",
        )

        paired_image = build_image_tensor(
            image_path=image_path,
            image_size=self.image_size,
            severity=self.paired_severity,
        )

        label = int(row["label"])

        return {
            "clean_image": clean_image,
            "paired_image": paired_image,
            "label": torch.tensor(label, dtype=torch.long),
            "class_name": row["class_name"],
            "rel_path": row["rel_path"],
            "abs_path": image_path,
            "dataset": self.dataset,
            "split": "train",
        }


class RetrievalEvalDataset(Dataset):
    """
    query / gallery / val 评测集。
    severity 由外部指定。
    """

    def __init__(
        self,
        dataset: str,
        split_seed: int,
        split_name: str,
        image_size: int = 224,
        severity: str = "clean",
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.split_seed = split_seed
        self.split_name = split_name
        self.image_size = image_size
        self.severity = severity

        df = load_split_manifest(dataset=dataset, split_seed=split_seed)
        df = filter_by_split(df, split_name)
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        image_path = _resolve_abs_path(row, self.dataset)

        image = build_image_tensor(
            image_path=image_path,
            image_size=self.image_size,
            severity=self.severity,
        )

        label = int(row["label"])

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "class_name": row["class_name"],
            "rel_path": row["rel_path"],
            "abs_path": image_path,
            "dataset": self.dataset,
            "split": self.split_name,
            "severity": self.severity,
        }
