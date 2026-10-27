from pathlib import Path
from typing import Dict

import pandas as pd

from robust_hash.utils.paths import get_split_root


SUPPORTED_SPLITS = {"train", "val", "test_query", "test_retrieval"}


def get_split_manifest_path(dataset: str, split_seed: int) -> Path:
    return get_split_root() / dataset / f"seed_{split_seed}" / "split_manifest.csv"


def load_split_manifest(dataset: str, split_seed: int) -> pd.DataFrame:
    manifest_path = get_split_manifest_path(dataset, split_seed)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path}"
        )

    df = pd.read_csv(manifest_path)

    required_columns = {"rel_path", "label", "class_name", "split"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in manifest: {missing}. "
            f"Manifest path: {manifest_path}"
        )

    return df


def filter_by_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if split_name not in SUPPORTED_SPLITS:
        raise ValueError(
            f"Unsupported split_name={split_name}. "
            f"Supported: {sorted(SUPPORTED_SPLITS)}"
        )
    out = df[df["split"] == split_name].copy()
    out = out.reset_index(drop=True)
    return out


def build_split_dict(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    split_dict: Dict[str, pd.DataFrame] = {}
    for split_name in sorted(SUPPORTED_SPLITS):
        split_dict[split_name] = filter_by_split(df, split_name)
    return split_dict