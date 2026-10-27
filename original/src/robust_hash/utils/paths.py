from pathlib import Path
import os


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
    return Path(os.environ.get("DAHCR_DATA_ROOT", str(get_project_root() / "data")))


def get_raw_data_root() -> Path:
    return get_data_root() / "raw"


def get_split_root() -> Path:
    # Packaged original manifests retain the path identities used for noise seeds.
    return Path(os.environ.get("DAHCR_SPLIT_ROOT", str(get_project_root() / "data" / "splits")))


def get_output_root() -> Path:
    return get_project_root() / "outputs"


def resolve_source_image(original_path: str | Path) -> Path:
    """Map the file-open location while retaining original_path for noise seeding.

    DAHCR_DATA_ROOT contains raw/<dataset>/<relative image>. No manifest rows or
    original identities are rewritten. Existing original files take precedence.
    """
    original = Path(original_path)
    if original.is_file():
        return original
    identity = original.as_posix()
    marker = "/data/raw/"
    if "DAHCR_DATA_ROOT" in os.environ and marker in identity:
        suffix = identity.split(marker, 1)[1]
        relative = Path(suffix)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe image-path suffix")
        mapped = get_raw_data_root() / relative
        if mapped.is_file():
            return mapped
    raise FileNotFoundError(f"Original image unavailable: {original}; set DAHCR_DATA_ROOT to the directory containing raw/")
