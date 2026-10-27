"""
Protocol: formal_v12_monotonic

Purpose:
Severity-ordered image degradation protocol for robust hashing retrieval.

Design:
clean  : original image, no synthetic degradation
weak   : mild downsampling + JPEG + Gaussian noise
medium : stronger downsampling + JPEG + Gaussian noise
strong : strongest downsampling + JPEG + Gaussian noise

Important:
No color jitter.
No cutout.
Noise is deterministic for each image_path + severity, so validation/final-test are reproducible.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from robust_hash.utils.paths import resolve_source_image


InterpolationName = Literal["nearest", "bilinear", "bicubic", "lanczos"]

INTERPOLATION_MAP = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


DEG_PROTOCOLS = {
    "formal_v12_monotonic": {
        "clean": {
            "downsample_scale": 1.00,
            "jpeg_quality": 100,
            "noise_sigma": 0.0,
            "color_jitter": 0.0,
            "cutout_ratio": 0.0,
        },
        "weak": {
            "downsample_scale": 0.65,
            "jpeg_quality": 60,
            "noise_sigma": 6.0,
            "color_jitter": 0.0,
            "cutout_ratio": 0.0,
        },
        "medium": {
            "downsample_scale": 0.45,
            "jpeg_quality": 38,
            "noise_sigma": 10.0,
            "color_jitter": 0.0,
            "cutout_ratio": 0.0,
        },
        "strong": {
            "downsample_scale": 0.30,
            "jpeg_quality": 22,
            "noise_sigma": 15.0,
            "color_jitter": 0.0,
            "cutout_ratio": 0.0,
        },
    }
}

ACTIVE_PROTOCOL = "formal_v12_monotonic"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _seed_from_key(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _rng_for(image_path: str | Path, severity: str) -> np.random.Generator:
    key = f"{ACTIVE_PROTOCOL}|{severity}|{Path(image_path).as_posix()}"
    return np.random.default_rng(_seed_from_key(key))


def _resize_downup(
    image: Image.Image,
    scale: float,
    interpolation: InterpolationName = "bilinear",
) -> Image.Image:
    if not (0 < scale <= 1.0):
        raise ValueError(f"scale must be in (0, 1], got {scale}")

    if scale == 1.0:
        return image.copy()

    w, h = image.size
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = INTERPOLATION_MAP[interpolation]

    small = image.resize((new_w, new_h), resample=interp)
    return small.resize((w, h), resample=interp)


def _jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    quality = int(quality)

    # quality >= 100 means no extra JPEG recompression.
    if quality >= 100:
        return image.copy()

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _add_gaussian_noise(
    image: Image.Image,
    sigma: float,
    rng: np.random.Generator,
) -> Image.Image:
    sigma = float(sigma)

    if sigma <= 0:
        return image.copy()

    arr = np.asarray(image).astype(np.float32)
    noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _apply_degradation(
    pil_img: Image.Image,
    params: dict,
    rng: np.random.Generator,
) -> Image.Image:
    image = pil_img.convert("RGB")

    image = _resize_downup(
        image,
        scale=float(params["downsample_scale"]),
        interpolation="bilinear",
    )

    image = _jpeg_compress(
        image,
        quality=int(params["jpeg_quality"]),
    )

    image = _add_gaussian_noise(
        image,
        sigma=float(params["noise_sigma"]),
        rng=rng,
    )

    return image


def _build_tensor_pipeline(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class BasicImageTransform:
    """
    Compatibility transform for older helper datasets.
    It only resizes and normalizes without degradation.
    """

    def __init__(self, image_size: int = 224) -> None:
        self.image_size = image_size
        self.pipeline = _build_tensor_pipeline(image_size)

    def __call__(self, pil_img: Image.Image) -> torch.Tensor:
        return self.pipeline(pil_img.convert("RGB"))


def build_image_tensor(
    image_path: str | Path,
    image_size: int = 224,
    severity: str = "clean",
) -> torch.Tensor:
    """
    Main entry used by datasets.py.
    Pipeline:
    read image -> deterministic degradation -> resize -> tensor -> normalize
    """

    assert severity in ("clean", "weak", "medium", "strong"), (
        f"severity must be one of clean/weak/medium/strong, got: {severity}"
    )

    image_path = Path(image_path)
    pil_img = Image.open(resolve_source_image(image_path)).convert("RGB")

    params = DEG_PROTOCOLS[ACTIVE_PROTOCOL][severity]
    rng = _rng_for(image_path, severity)
    degraded = _apply_degradation(pil_img, params, rng)

    pipeline = _build_tensor_pipeline(image_size)
    return pipeline(degraded)


def apply_degradation_for_preview(
    pil_img: Image.Image,
    severity: str,
) -> Image.Image:
    """
    Preview helper.
    Returns degraded PIL image without resize / tensor / normalize.
    """

    assert severity in ("clean", "weak", "medium", "strong"), (
        f"severity must be one of clean/weak/medium/strong, got: {severity}"
    )

    params = DEG_PROTOCOLS[ACTIVE_PROTOCOL][severity]
    rng = np.random.default_rng(_seed_from_key(f"preview|{ACTIVE_PROTOCOL}|{severity}"))
    return _apply_degradation(pil_img.convert("RGB"), params, rng)
