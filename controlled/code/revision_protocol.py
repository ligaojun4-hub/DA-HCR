"""Frozen, deterministic JEI revision inputs and retrieval metrics.

This module does not import torch or touch the legacy implementation. Legacy
clean/weak/medium/strong specs must be routed to the original build_image_tensor.
New corruptions are applied after the same RGB -> PIL bilinear square resize as
the old clean branch, and before ToTensor / ImageNet normalization.

Perspective is a declared new protocol inspired by Zhukov et al., not a claim to
reproduce their incompletely specified zoom parameter l. An input trapezoid is
mapped onto a full output rectangle; this avoids fill regions but includes crop
and nonuniform magnification. It does not simulate three-dimensional viewpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageFilter


PROTOCOL_ID = "jei_revision_20261006_perspective_v1"
PERSPECTIVE_DIRECTIONS = ("top", "bottom", "left", "right")
PERSPECTIVE_RHOS = (0.05, 0.10, 0.20)
SEVERITY_NAMES = ("weak", "medium", "strong")

SPECS: list[dict[str, Any]] = [
    {"name": name, "kind": "legacy", "family": "legacy_composite",
     "level": name, "standard_severity": name}
    for name in ("clean", "weak", "medium", "strong")
]
for _rho, _level in zip(PERSPECTIVE_RHOS, SEVERITY_NAMES):
    for _direction in PERSPECTIVE_DIRECTIONS:
        SPECS.append({
            "name": f"perspective_{_direction}_rho{_rho:.2f}",
            "kind": "perspective", "family": "perspective", "level": _level,
            "direction": _direction, "rho": _rho,
        })
for _sigma, _level in zip((0.5, 1.0, 2.0), SEVERITY_NAMES):
    SPECS.append({
        "name": f"blur_sigma{_sigma:.1f}", "kind": "blur", "family": "gaussian_blur",
        "level": _level, "sigma": _sigma,
        "implementation": "Pillow ImageFilter.GaussianBlur, radius=sigma",
    })
for _magnitude, _level in zip((15, 30, 45), SEVERITY_NAMES):
    for _sign, _suffix, _family in ((-1, "m", "brightness_dark"), (1, "p", "brightness_bright")):
        SPECS.append({
            "name": f"brightness_{_suffix}{_magnitude}", "kind": "brightness",
            "family": _family, "level": _level, "delta": _sign * _magnitude,
        })
SPECS_BY_NAME = {spec["name"]: spec for spec in SPECS}


def _validate_geometry(width: int, height: int, rho: float, direction: str) -> None:
    if width < 2 or height < 2:
        raise ValueError("Perspective geometry requires width and height >= 2")
    if not np.isfinite(rho) or not 0 <= rho < 0.5:
        raise ValueError("rho must be finite and in [0, 0.5)")
    if direction not in PERSPECTIVE_DIRECTIONS:
        raise ValueError(f"Unknown perspective direction: {direction!r}")


def perspective_source_points(width: int, height: int, rho: float, direction: str) -> np.ndarray:
    """Input corners, consistently TL/TR/BR/BL; pixel indices use W-1/H-1."""
    _validate_geometry(width, height, float(rho), direction)
    r = float(rho)
    normalized = {
        "top": ((r, 0), (1-r, 0), (1, 1), (0, 1)),
        "bottom": ((0, 0), (1, 0), (1-r, 1), (r, 1)),
        "left": ((0, r), (1, 0), (1, 1), (0, 1-r)),
        "right": ((0, 0), (1, r), (1, 1-r), (0, 1)),
    }[direction]
    return np.asarray(normalized, dtype=np.float64) * np.array((width-1, height-1), dtype=np.float64)


def perspective_target_points(width: int, height: int) -> np.ndarray:
    if width < 2 or height < 2:
        raise ValueError("Perspective geometry requires width and height >= 2")
    return np.array(((0, 0), (width-1, 0), (width-1, height-1), (0, height-1)), dtype=np.float64)


def perspective_homography(width: int, height: int, rho: float, direction: str) -> np.ndarray:
    """Return source -> target H with H[2,2]=1, in pixel-index coordinates."""
    source = perspective_source_points(width, height, rho, direction)
    if float(rho) == 0.0:
        return np.eye(3, dtype=np.float64)
    target = perspective_target_points(width, height)
    rows, values = [], []
    for (x, y), (u, v) in zip(source, target):
        rows.append((x, y, 1, 0, 0, 0, -u*x, -u*y))
        values.append(u)
        rows.append((0, 0, 0, x, y, 1, -v*x, -v*y))
        values.append(v)
    h = np.linalg.solve(np.asarray(rows, dtype=np.float64), np.asarray(values, dtype=np.float64))
    result = np.append(h, 1.0).reshape(3, 3)
    if not np.isfinite(result).all() or abs(np.linalg.det(result)) < 1e-12:
        raise ValueError("Invalid or singular homography")
    return result


def project_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or matrix.shape != (3, 3):
        raise ValueError("Expected points [N,2] and matrix [3,3]")
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if (np.abs(homogeneous[:, 2]) < 1e-12).any():
        raise ValueError("Projection intersects the line at infinity")
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def pillow_inverse_coefficients(matrix: np.ndarray) -> tuple[float, ...]:
    """Pillow output->input coefficients, including its pixel-center convention.

    H is defined on indices 0..W-1. Pillow's resampler evaluates the transform at
    output centers x+0.5,y+0.5 and subtracts 0.5 for interpolation. Conjugating the
    inverse by these translations preserves the declared integer-index geometry.
    """
    inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
    to_center = np.array(((1, 0, .5), (0, 1, .5), (0, 0, 1)), dtype=np.float64)
    from_center = np.array(((1, 0, -.5), (0, 1, -.5), (0, 0, 1)), dtype=np.float64)
    coefficients = to_center @ inverse @ from_center
    if abs(coefficients[2, 2]) < 1e-12:
        raise ValueError("Cannot normalize Pillow inverse coefficients")
    coefficients /= coefficients[2, 2]
    return tuple(float(v) for v in coefficients.reshape(-1)[:8])


def sampling_boundary_report(matrix: np.ndarray, width: int, height: int, tolerance: float = 1e-7) -> dict[str, Any]:
    ys, xs = np.mgrid[0:height, 0:width]
    sampling_points = project_points(np.column_stack((xs.ravel(), ys.ravel())), np.linalg.inv(matrix))
    outside = (
        (sampling_points[:, 0] < -tolerance) | (sampling_points[:, 0] > width-1+tolerance)
        | (sampling_points[:, 1] < -tolerance) | (sampling_points[:, 1] > height-1+tolerance)
    )
    return {"num_output_pixels": width * height, "num_outside": int(outside.sum()),
            "minimum_xy": sampling_points.min(axis=0).tolist(),
            "maximum_xy": sampling_points.max(axis=0).tolist(), "tolerance_pixels": tolerance}


def warp_perspective(image: Image.Image, rho: float, direction: str) -> Image.Image:
    """Warp a working-resolution RGB image without a second resize."""
    image = image.convert("RGB")
    width, height = image.size
    matrix = perspective_homography(width, height, rho, direction)
    if float(rho) == 0.0:
        return image.copy()
    return image.transform(image.size, Image.Transform.PERSPECTIVE,
                           pillow_inverse_coefficients(matrix), resample=Image.Resampling.BILINEAR)


def resize_working_rgb(image: Image.Image, image_size: int = 224) -> Image.Image:
    """Equivalent to torchvision Resize((size,size)) on a PIL RGB image."""
    if image_size < 2:
        raise ValueError("image_size must be >= 2")
    return image.convert("RGB").resize((image_size, image_size), resample=Image.Resampling.BILINEAR)


def apply_new_pil(image: Image.Image | str | Path, spec: dict[str, Any], image_size: int = 224) -> Image.Image:
    """Return a new-corruption RGB PIL image, before tensor normalization.

    Legacy specs are deliberately not reimplemented: use build_image_tensor(path,
    image_size, severity=spec['standard_severity']) from the original source.
    """
    kind = spec.get("kind")
    if kind == "legacy" or spec.get("standard_severity") is not None:
        raise ValueError("Route legacy specs to the original build_image_tensor; do not reimplement them here")
    if not isinstance(image, Image.Image):
        with Image.open(image) as decoded:
            image = decoded.convert("RGB")
    working = resize_working_rgb(image, image_size)
    if kind == "perspective":
        return warp_perspective(working, float(spec["rho"]), str(spec["direction"]))
    if kind == "blur":
        sigma = float(spec["sigma"])
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("Blur sigma must be finite and nonnegative")
        return working.filter(ImageFilter.GaussianBlur(radius=sigma))
    if kind == "brightness":
        delta = spec["delta"]
        if not np.isfinite(delta) or int(delta) != delta:
            raise ValueError("Brightness delta must be a finite integer on the 8-bit pixel scale")
        pixels = np.asarray(working).astype(np.int32)
        return Image.fromarray(np.clip(pixels + int(delta), 0, 255).astype(np.uint8))
    raise ValueError(f"Unknown new corruption kind: {kind!r}")


def _validated_codes(codes: Any, name: str) -> np.ndarray:
    result = np.asarray(codes)
    if result.ndim != 2 or result.shape[1] == 0:
        raise ValueError(f"{name} must have shape [N,K] with K>0")
    if not np.isfinite(result).all() or not ((result == -1) | (result == 1)).all():
        raise ValueError(f"{name} must contain only finite -1/+1 values; binarize zero explicitly")
    return result.astype(np.float32, copy=False)


def hamming_distance_matrix(query_codes: Any, gallery_codes: Any) -> np.ndarray:
    query = _validated_codes(query_codes, "query_codes")
    gallery = _validated_codes(gallery_codes, "gallery_codes")
    if query.shape[1] != gallery.shape[1]:
        raise ValueError("Query and gallery bit lengths differ")
    return 0.5 * (query.shape[1] - query @ gallery.T)


def _ids(ids: Sequence[Any], expected: int, name: str, unique: bool = False) -> list[str]:
    result = [str(value) for value in ids]
    if len(result) != expected:
        raise ValueError(f"{name} length does not match code rows")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique so gallery tie/self rules are unambiguous")
    return result


def evaluate_codes(query_codes: Any, query_labels: Any, gallery_codes: Any, gallery_labels: Any,
                   query_ids: Sequence[Any], gallery_ids: Sequence[Any], exclude_self: bool = False,
                   chunk_size: int = 256) -> dict[str, Any]:
    """Chunked full-gallery AP; stable tie key is lexicographic gallery_id.

    AP divides the sum of precisions at all relevant ranks by the total relevant
    gallery count after optional self removal. Queries without a relevant item
    have NaN AP and are explicitly counted. mAP is None if no query is usable.
    Memory is O(chunk_size*gallery_size), plus the input codes and output vectors.
    """
    query = _validated_codes(query_codes, "query_codes")
    gallery = _validated_codes(gallery_codes, "gallery_codes")
    if query.shape[1] != gallery.shape[1]:
        raise ValueError("Query and gallery bit lengths differ")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    qlabels = np.asarray(query_labels).reshape(-1)
    glabels = np.asarray(gallery_labels).reshape(-1)
    if len(qlabels) != len(query) or len(glabels) != len(gallery):
        raise ValueError("Single-label vector length must match code rows")
    qids = _ids(query_ids, len(query), "query_ids")
    gids = _ids(gallery_ids, len(gallery), "gallery_ids", unique=True)
    secondary_order = np.argsort(np.asarray(gids, dtype=str), kind="stable")
    gallery_lookup = {image_id: i for i, image_id in enumerate(gids)}
    ap = np.full(len(query), np.nan, dtype=np.float64)
    relevant_counts = np.zeros(len(query), dtype=np.int64)
    available_gallery_counts = np.zeros(len(query), dtype=np.int64)
    tie_fraction = np.full(len(query), np.nan, dtype=np.float64)
    for start in range(0, len(query), chunk_size):
        distances = 0.5 * (query.shape[1] - query[start:start+chunk_size] @ gallery.T)
        for offset, distance_row in enumerate(distances):
            index = start + offset
            eligible = secondary_order
            if exclude_self and qids[index] in gallery_lookup:
                eligible = eligible[eligible != gallery_lookup[qids[index]]]
            available_gallery_counts[index] = len(eligible)
            if not len(eligible):
                continue
            _, counts = np.unique(distance_row[eligible], return_counts=True)
            tie_fraction[index] = counts[counts > 1].sum() / float(len(eligible))
            ranked = eligible[np.argsort(distance_row[eligible], kind="stable")]
            relevant = glabels[ranked] == qlabels[index]
            count = int(relevant.sum())
            relevant_counts[index] = count
            if count == 0:
                continue
            ranks = np.arange(1, len(ranked)+1, dtype=np.float64)
            cumulative = np.cumsum(relevant, dtype=np.float64)
            ap[index] = float((cumulative[relevant] / ranks[relevant]).sum() / count)
    valid = np.isfinite(ap)
    without_gallery = available_gallery_counts == 0
    without_relevant = (relevant_counts == 0) & ~without_gallery
    return {
        "mAP": float(ap[valid].mean()) if valid.any() else None,
        "per_query_ap": ap, "query_ids": qids, "valid_query_mask": valid,
        "relevant_counts": relevant_counts, "available_gallery_counts": available_gallery_counts,
        "num_query_used": int(valid.sum()), "num_query_total": len(query),
        "num_query_without_relevant": int(without_relevant.sum()),
        "num_query_without_gallery": int(without_gallery.sum()),
        "invalid_query_ids": [qids[i] for i in np.flatnonzero(~valid)],
        "per_query_tied_gallery_fraction": tie_fraction,
        "mean_tied_gallery_fraction": float(np.nanmean(tie_fraction)) if np.isfinite(tie_fraction).any() else None,
        "tie_rule": "ascending Hamming distance, then lexicographic unique gallery_id",
        "exclude_self": bool(exclude_self), "gallery_size": len(gallery), "hash_bits": query.shape[1],
    }


def code_diagnostics(codes: Any) -> dict[str, Any]:
    values = _validated_codes(codes, "codes")
    if not len(values):
        return {"num_images": 0, "hash_bits": values.shape[1], "num_unique_codes": 0,
                "unique_code_ratio": None, "bit_plus_fraction": [], "mean_bit_entropy_bits": None,
                "num_constant_bits": None, "constant_bit_indices": []}
    plus_fraction = (values > 0).mean(axis=0)
    entropy = np.zeros(values.shape[1], dtype=np.float64)
    interior = (plus_fraction > 0) & (plus_fraction < 1)
    p = plus_fraction[interior]
    entropy[interior] = -p * np.log2(p) - (1-p) * np.log2(1-p)
    packed = np.packbits(values > 0, axis=1)
    unique = len(np.unique(packed, axis=0))
    constant = np.flatnonzero(~interior)
    return {"num_images": len(values), "hash_bits": values.shape[1], "num_unique_codes": unique,
            "unique_code_ratio": unique / float(len(values)), "bit_plus_fraction": plus_fraction.tolist(),
            "mean_bit_entropy_bits": float(entropy.mean()), "num_constant_bits": len(constant),
            "constant_bit_indices": constant.tolist()}


def bit_flip_diagnostics(clean_codes: Any, degraded_codes: Any,
                         clean_ids: Sequence[Any] | None = None,
                         degraded_ids: Sequence[Any] | None = None) -> dict[str, Any]:
    clean = _validated_codes(clean_codes, "clean_codes")
    degraded = _validated_codes(degraded_codes, "degraded_codes")
    if clean.shape != degraded.shape:
        raise ValueError("Clean and degraded codes must have identical [N,K] shape")
    if (clean_ids is None) != (degraded_ids is None):
        raise ValueError("Provide both ID vectors or neither")
    if clean_ids is not None and _ids(clean_ids, len(clean), "clean_ids") != _ids(degraded_ids, len(degraded), "degraded_ids"):
        raise ValueError("Clean/degraded query IDs are not aligned; do not compare unrelated rows")
    hamming = np.count_nonzero(clean != degraded, axis=1)
    flips = hamming.astype(np.float64) / clean.shape[1]
    mean_flip = float(flips.mean()) if len(clean) else None
    return {"bit_flip_rate": mean_flip,
            "per_query_bit_flip_rate": flips, "per_query_hamming": hamming,
            "mean_hamming_distance": float(hamming.mean()) if len(clean) else None,
            "sign_consistency": 1-mean_flip if mean_flip is not None else None,
            "num_queries": len(clean), "hash_bits": clean.shape[1],
            "metric_relationship": "sign_consistency=1-BFR; mean_hamming=K*BFR; these are redundant diagnostics"}


def bit_flip_rate(clean_codes: Any, degraded_codes: Any) -> float | None:
    return bit_flip_diagnostics(clean_codes, degraded_codes)["bit_flip_rate"]
