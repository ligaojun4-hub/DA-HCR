#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import io
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEGRADE_PARAMS = {
    "clean": None,
    "weak": {"scale": 0.75, "jpeg_quality": 70, "blur_sigma": 0.5},
    "medium": {"scale": 0.50, "jpeg_quality": 50, "blur_sigma": 1.0},
    "strong": {"scale": 0.25, "jpeg_quality": 30, "blur_sigma": 1.5},
}


@dataclass
class SplitConfig:
    seed: int
    train_frac: float
    val_frac: float
    query_frac: float
    exclude_clutter: bool


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def maybe_caltech_root(root: Path) -> Path:
    candidates = [
        root / "256_ObjectCategories",
        root / "caltech256" / "256_ObjectCategories",
        root / "Caltech256" / "256_ObjectCategories",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return root


def list_images_by_class(root: Path, exclude_clutter: bool) -> Tuple[List[str], Dict[str, List[str]]]:
    root = maybe_caltech_root(root)
    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    class_dirs = sorted(class_dirs, key=lambda p: p.name.lower())

    classes: List[str] = []
    class_to_images: Dict[str, List[str]] = {}

    for class_dir in class_dirs:
        name_lower = class_dir.name.lower()
        if exclude_clutter and ("clutter" in name_lower or name_lower.startswith("257.")):
            continue

        images = sorted(
            str(p)
            for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if images:
            classes.append(class_dir.name)
            class_to_images[class_dir.name] = images

    if not classes:
        raise RuntimeError(f"no image class folders found under {root}")

    return classes, class_to_images


def build_or_load_split(args: argparse.Namespace) -> dict:
    split_path = Path(args.split_json)
    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    data_root = maybe_caltech_root(Path(args.data_root))
    classes, class_to_images = list_images_by_class(data_root, args.exclude_clutter)
    rng = random.Random(args.seed)

    items = {"train": [], "val": [], "test_query": [], "gallery": []}
    class_meta = []

    for label, class_name in enumerate(classes):
        paths = class_to_images[class_name][:]
        rng.shuffle(paths)
        n = len(paths)

        n_train = max(1, int(round(n * args.train_frac)))
        n_val = max(1, int(round(n * args.val_frac)))
        n_query = max(1, int(round(n * args.query_frac)))

        if n_train + n_val + n_query >= n:
            n_train = max(1, int(0.6 * n))
            n_val = max(1, int(0.1 * n))
            n_query = max(1, int(0.1 * n))
        n_gallery = n - n_train - n_val - n_query
        if n_gallery <= 0:
            n_gallery = 1
            n_train = max(1, n_train - 1)

        train = paths[:n_train]
        val = paths[n_train : n_train + n_val]
        query = paths[n_train + n_val : n_train + n_val + n_query]
        gallery = paths[n_train + n_val + n_query :]

        for split_name, split_items in [
            ("train", train),
            ("val", val),
            ("test_query", query),
            ("gallery", gallery),
        ]:
            for path in split_items:
                rel = str(Path(path).resolve().relative_to(data_root.resolve()))
                items[split_name].append([rel, label])

        class_meta.append({"name": class_name, "label": label, "num_images": n})

    split = {
        "data_root": str(data_root.resolve()),
        "config": asdict(
            SplitConfig(
                seed=args.seed,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                query_frac=args.query_frac,
                exclude_clutter=args.exclude_clutter,
            )
        ),
        "classes": class_meta,
        "items": items,
    }

    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    return split


def resolve_item_paths(split: dict, part: str) -> List[Tuple[str, int]]:
    data_root = Path(split["data_root"])
    resolved = []
    for path_str, label in split["items"][part]:
        path = Path(path_str)
        if not path.is_absolute():
            path = data_root / path
        resolved.append((str(path), int(label)))
    return resolved


def degrade_image(image: Image.Image, level: str) -> Image.Image:
    params = DEGRADE_PARAMS[level]
    if params is None:
        return image.copy()

    image = image.convert("RGB")
    width, height = image.size
    scale = params["scale"]
    small_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))

    out = image.resize(small_size, Image.BICUBIC).resize((width, height), Image.BICUBIC)

    buffer = io.BytesIO()
    out.save(buffer, format="JPEG", quality=int(params["jpeg_quality"]))
    buffer.seek(0)
    out = Image.open(buffer).convert("RGB")

    sigma = float(params["blur_sigma"])
    if sigma > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=sigma))
    return out


def make_transform(input_size: int, is_train: bool):
    ops = [transforms.Resize((input_size, input_size))]
    if is_train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(ops)


class ImageSplitDataset(Dataset):
    def __init__(self, items: Sequence[Tuple[str, int]], transform, degrade_level: str = "clean"):
        self.items = list(items)
        self.transform = transform
        self.degrade_level = degrade_level

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        image = Image.open(path).convert("RGB")
        image = degrade_image(image, self.degrade_level)
        return self.transform(image), label, path


class PairDegAugDataset(Dataset):
    def __init__(self, items: Sequence[Tuple[str, int]], transform, degrade_level: str = "medium"):
        self.items = list(items)
        self.transform = transform
        self.degrade_level = degrade_level

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        image = Image.open(path).convert("RGB")
        clean = self.transform(image)
        degraded = self.transform(degrade_image(image, self.degrade_level))
        return clean, degraded, label, path


class HashNetResNet18(nn.Module):
    def __init__(self, bits: int, pretrained: bool):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = resnet18(weights=weights)
        except Exception:
            from torchvision.models import resnet18

            base = resnet18(pretrained=pretrained)

        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.hash_head = nn.Linear(in_features, bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.hash_head(feat)


def hashnet_pairwise_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
    quant_weight: float,
) -> torch.Tensor:
    u = torch.tanh(beta * z)
    theta = 0.5 * torch.matmul(u, u.t())
    same = labels.view(-1, 1).eq(labels.view(1, -1)).float()
    mask = ~torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)

    pair_loss = F.softplus(theta) - same * theta
    pos = (same * mask.float()).sum().clamp_min(1.0)
    neg = ((1.0 - same) * mask.float()).sum().clamp_min(1.0)
    weights = torch.where(same > 0.5, 0.5 / pos, 0.5 / neg)
    pair_loss = (pair_loss * weights * mask.float()).sum()

    quant_loss = (u.abs() - 1.0).pow(2).mean()
    return pair_loss + quant_weight * quant_loss


def train_one_epoch(model, loader, optimizer, scaler, device, args) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_clean = 0.0
    total_degraded = 0.0
    steps = 0

    for x_clean, x_degraded, labels, _ in loader:
        x_clean = x_clean.to(device, non_blocking=True)
        x_degraded = x_degraded.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            z_clean = model(x_clean)
            z_degraded = model(x_degraded)
            loss_clean = hashnet_pairwise_loss(z_clean, labels, args.beta, args.quant_weight)
            loss_degraded = hashnet_pairwise_loss(z_degraded, labels, args.beta, args.quant_weight)
            loss = 0.5 * (loss_clean + loss_degraded)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().cpu())
        total_clean += float(loss_clean.detach().cpu())
        total_degraded += float(loss_degraded.detach().cpu())
        steps += 1

    denom = max(steps, 1)
    return {
        "train_loss": total_loss / denom,
        "train_clean_sup_loss": total_clean / denom,
        "train_degraded_sup_loss": total_degraded / denom,
    }


@torch.no_grad()
def collect_codes(model, loader, device):
    model.eval()
    codes = []
    labels = []
    paths = []
    for x, y, p in loader:
        x = x.to(device, non_blocking=True)
        z = model(x)
        b = torch.sign(z)
        b[b == 0] = 1
        codes.append(b.cpu().to(torch.int8))
        labels.append(torch.as_tensor(y, dtype=torch.long))
        paths.extend(list(p))
    return torch.cat(codes, dim=0), torch.cat(labels, dim=0), paths


def average_precision(relevant: np.ndarray) -> float:
    total_relevant = int(relevant.sum())
    if total_relevant == 0:
        return 0.0
    cumsum = np.cumsum(relevant)
    precision = cumsum / (np.arange(len(relevant)) + 1)
    return float((precision * relevant).sum() / total_relevant)


def mean_average_precision(
    q_codes: torch.Tensor,
    q_labels: torch.Tensor,
    q_paths: Sequence[str],
    db_codes: torch.Tensor,
    db_labels: torch.Tensor,
    db_paths: Sequence[str],
) -> float:
    db_codes_f = db_codes.float()
    ap_values = []
    bits = q_codes.shape[1]

    db_labels_np = db_labels.numpy()
    db_paths_np = np.asarray(db_paths)

    for i in range(q_codes.shape[0]):
        q = q_codes[i].float()
        dist = ((bits - torch.mv(db_codes_f, q)) / 2.0).numpy()
        order = np.argsort(dist, kind="stable")
        relevant = (db_labels_np[order] == int(q_labels[i])).astype(np.float32)
        same_path = db_paths_np[order] == q_paths[i]
        relevant[same_path] = 0.0
        ap_values.append(average_precision(relevant))

    return float(np.mean(ap_values)) if ap_values else 0.0


@torch.no_grad()
def eval_all_levels(model, loaders: Dict[str, DataLoader], device) -> Dict[str, object]:
    db_codes, db_labels, db_paths = collect_codes(model, loaders["gallery"], device)
    metrics: Dict[str, object] = {}

    for level in ["clean", "weak", "medium", "strong"]:
        q_codes, q_labels, q_paths = collect_codes(model, loaders[f"query_{level}"], device)
        metrics[level] = {
            "mAP": mean_average_precision(
                q_codes, q_labels, q_paths, db_codes, db_labels, db_paths
            )
        }

    metrics["robust_avg"] = float(
        (metrics["weak"]["mAP"] + metrics["medium"]["mAP"] + metrics["strong"]["mAP"]) / 3.0
    )
    return metrics


def make_loaders(split: dict, args) -> Dict[str, DataLoader]:
    train_tf = make_transform(args.input_size, is_train=True)
    eval_tf = make_transform(args.input_size, is_train=False)

    train_items = resolve_item_paths(split, "train")
    val_items = resolve_item_paths(split, "val")
    query_items = resolve_item_paths(split, "test_query")
    gallery_items = resolve_item_paths(split, "gallery")

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
    }

    return {
        "train": DataLoader(
            PairDegAugDataset(train_items, train_tf, degrade_level=args.train_degrade),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        ),
        "val_gallery": DataLoader(
            ImageSplitDataset(val_items, eval_tf, degrade_level="clean"),
            batch_size=args.eval_batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
        "val_query_clean": DataLoader(
            ImageSplitDataset(val_items, eval_tf, degrade_level="clean"),
            batch_size=args.eval_batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
        "val_query_weak": DataLoader(
            ImageSplitDataset(val_items, eval_tf, degrade_level="weak"),
            batch_size=args.eval_batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
        "val_query_medium": DataLoader(
            ImageSplitDataset(val_items, eval_tf, degrade_level="medium"),
            batch_size=args.eval_batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
        "val_query_strong": DataLoader(
            ImageSplitDataset(val_items, eval_tf, degrade_level="strong"),
            batch_size=args.eval_batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
        "gallery": DataLoader(
            ImageSplitDataset(gallery_items, eval_tf, degrade_level="clean"),
            batch_size=args.eval_batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
        "query_clean": DataLoader(
            ImageSplitDataset(query_items, eval_tf, degrade_level="clean"),
            batch_size=args.eval_batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
        "query_weak": DataLoader(
            ImageSplitDataset(query_items, eval_tf, degrade_level="weak"),
            batch_size=args.eval_batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
        "query_medium": DataLoader(
            ImageSplitDataset(query_items, eval_tf, degrade_level="medium"),
            batch_size=args.eval_batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
        "query_strong": DataLoader(
            ImageSplitDataset(query_items, eval_tf, degrade_level="strong"),
            batch_size=args.eval_batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
    }


def val_loader_view(loaders: Dict[str, DataLoader]) -> Dict[str, DataLoader]:
    return {
        "gallery": loaders["val_gallery"],
        "query_clean": loaders["val_query_clean"],
        "query_weak": loaders["val_query_weak"],
        "query_medium": loaders["val_query_medium"],
        "query_strong": loaders["val_query_strong"],
    }


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Base + DegAug control: Caltech256 + HashNet/ResNet18."
    )
    parser.add_argument("--data-root", required=True, help="Caltech256 root or 256_ObjectCategories.")
    parser.add_argument("--out-dir", default="outputs/degaug_caltech256_hashnet_seed42")
    parser.add_argument("--split-json", default="outputs/splits/caltech256_seed42_split.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bits", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--lr-backbone", type=float, default=5e-5)
    parser.add_argument("--lr-head", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1.0, help="HashNet tanh continuation beta.")
    parser.add_argument("--quant-weight", type=float, default=1e-3)
    parser.add_argument("--train-degrade", choices=["weak", "medium", "strong"], default="medium")
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--query-frac", type=float, default=0.10)
    parser.add_argument("--exclude-clutter", action="store_true", default=True)
    parser.add_argument("--include-clutter", action="store_false", dest="exclude_clutter")
    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--amp", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.amp = bool(args.amp)
    args.pretrained = bool(args.pretrained)
    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    ckpt_path = out_dir / "checkpoints" / "caltech256_hashnet_degaug_best.pth"
    history_path = out_dir / "tables" / "caltech256_hashnet_degaug_history.json"
    val_path = out_dir / "tables" / "caltech256_hashnet_degaug_val_curve.json"
    test_path = out_dir / "tables" / "caltech256_hashnet_degaug_test_curve.json"
    meta_path = out_dir / "tables" / "caltech256_hashnet_degaug_test_meta.json"

    split = build_or_load_split(args)
    loaders = make_loaders(split, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HashNetResNet18(bits=args.bits, pretrained=args.pretrained).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": model.hash_head.parameters(), "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    print(f"[device] {device}")
    print("[experiment] Caltech256 HashNet Base+DegAug")
    print(f"[data_root] {split['data_root']}")
    print(f"[classes] {len(split['classes'])}")
    print(
        "[split] "
        f"train={len(split['items']['train'])}, "
        f"val={len(split['items']['val'])}, "
        f"query={len(split['items']['test_query'])}, "
        f"gallery={len(split['items']['gallery'])}"
    )
    print("[loss] 0.5 * (L_sup(clean) + L_sup(degraded)); no DA-HCR consistency term")

    history = []
    val_curve = []
    best = None

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, loaders["train"], optimizer, scaler, device, args)
        scheduler.step()

        row = {"epoch": epoch, **train_stats}
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            val_metrics = eval_all_levels(model, val_loader_view(loaders), device)
            row.update(
                {
                    "val_clean_mAP": val_metrics["clean"]["mAP"],
                    "val_weak_mAP": val_metrics["weak"]["mAP"],
                    "val_medium_mAP": val_metrics["medium"]["mAP"],
                    "val_strong_mAP": val_metrics["strong"]["mAP"],
                    "val_robust_avg": val_metrics["robust_avg"],
                }
            )
            val_curve.append({"epoch": epoch, **val_metrics})

            candidate = {
                "epoch": epoch,
                "val_robust_avg": float(val_metrics["robust_avg"]),
                "val_clean_mAP": float(val_metrics["clean"]["mAP"]),
            }
            better = best is None or candidate["val_robust_avg"] > best["val_robust_avg"]
            if best is not None and candidate["val_robust_avg"] == best["val_robust_avg"]:
                better = candidate["val_clean_mAP"] >= best["val_clean_mAP"]

            if better:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)
                best = {**candidate, "checkpoint_path": str(ckpt_path)}

        history.append(row)
        print(row)

    if best is None:
        raise RuntimeError("validation was never evaluated")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = eval_all_levels(model, loaders, device)

    metadata = {
        "dataset": "Caltech256",
        "baseline": "HashNet",
        "method": "Base+DegAug",
        "bits": args.bits,
        "seed": args.seed,
        "train_degrade": args.train_degrade,
        "degrade_params": DEGRADE_PARAMS,
        "loss": "0.5 * (L_sup(clean) + L_sup(degraded))",
        "no_dahcr_consistency": True,
        "best_epoch": best["epoch"],
        "val_robust_avg": best["val_robust_avg"],
        "val_clean_mAP": best["val_clean_mAP"],
        "test_metrics": test_metrics,
        "split_json": str(Path(args.split_json)),
        "checkpoint_path": str(ckpt_path),
        "args": vars(args),
    }

    save_json(history_path, history)
    save_json(val_path, val_curve)
    save_json(test_path, test_metrics)
    save_json(meta_path, metadata)

    print("=" * 100)
    print("[DONE] Caltech256 HashNet Base+DegAug")
    print(f"[BEST] epoch={best['epoch']} val_robust_avg={best['val_robust_avg']:.6f}")
    print(f"[TEST] clean={test_metrics['clean']['mAP']:.6f} "
          f"weak={test_metrics['weak']['mAP']:.6f} "
          f"medium={test_metrics['medium']['mAP']:.6f} "
          f"strong={test_metrics['strong']['mAP']:.6f} "
          f"robust_avg={test_metrics['robust_avg']:.6f}")
    print(f"[META] {meta_path}")


if __name__ == "__main__":
    main()
