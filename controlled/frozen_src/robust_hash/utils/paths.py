import os
from pathlib import Path


def get_project_root() -> Path:
    """
    当前文件是:
    src/robust_hash/utils/paths.py

    parents[0] = utils
    parents[1] = robust_hash
    parents[2] = src
    parents[3] = project_root
    """
    return Path(__file__).resolve().parents[3]


def get_data_root() -> Path:
    override = os.environ.get("DA_HCR_DATA_ROOT", "").strip()
    return Path(override).expanduser().resolve() if override else get_project_root() / "data"


def get_raw_data_root() -> Path:
    return get_data_root() / "raw"


def get_split_root() -> Path:
    return get_data_root() / "splits"


def get_output_root() -> Path:
    override = os.environ.get("DA_HCR_OUTPUT_ROOT", "").strip()
    return Path(override).expanduser().resolve() if override else get_project_root() / "outputs"
