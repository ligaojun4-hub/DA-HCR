import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robust_hash.data.datasets import RetrievalEvalDataset
from robust_hash.evaluation.retrieval import (
    extract_codes_labels_paths,
    compute_map_from_codes,
)
from robust_hash.models.hash_model import RobustHashModel


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_loader(dataset, split_seed, split_name, severity,
                 image_size, batch_size, num_workers, device_type, max_samples=0):
    ds = RetrievalEvalDataset(
        dataset=dataset, split_seed=split_seed,
        split_name=split_name, image_size=image_size, severity=severity,
    )
    if max_samples > 0:
        ds = Subset(ds, list(range(min(max_samples, len(ds)))))
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device_type == "cuda"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",       type=str, required=True)
    parser.add_argument("--dataset",          type=str, required=True)
    parser.add_argument("--split-seed",       type=int, default=42)
    parser.add_argument("--image-size",       type=int, default=224)
    parser.add_argument("--eval-batch-size",  type=int, default=16)
    parser.add_argument("--eval-num-workers", type=int, default=12)
    parser.add_argument("--device",           type=str, default="auto")
    parser.add_argument("--save-json",        action="store_true")
    parser.add_argument("--max-query",        type=int, default=0)
    parser.add_argument("--max-gallery",      type=int, default=0)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = int(checkpoint["num_classes"])
    hash_bits   = int(checkpoint.get("args", {}).get("hash_bits", 64))

    model = RobustHashModel(
        num_classes=num_classes, hash_bits=hash_bits,
        backbone_name="resnet18", pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    gallery_loader = build_loader(
        dataset=args.dataset, split_seed=args.split_seed,
        split_name="test_retrieval", severity="clean",
        image_size=args.image_size, batch_size=args.eval_batch_size,
        num_workers=args.eval_num_workers, device_type=device.type,
        max_samples=args.max_gallery,
    )

    print("extracting gallery codes ...")
    gallery_codes, gallery_labels, gallery_paths = extract_codes_labels_paths(
        model=model, dataloader=gallery_loader, device=device,
    )
    print(f"gallery_size: {gallery_codes.shape[0]}")

    severity_list = ["clean", "weak", "medium", "strong"]
    results = {}

    for sev in severity_list:
        query_loader = build_loader(
            dataset=args.dataset, split_seed=args.split_seed,
            split_name="test_query", severity=sev,
            image_size=args.image_size, batch_size=args.eval_batch_size,
            num_workers=args.eval_num_workers, device_type=device.type,
            max_samples=args.max_query,
        )
        print(f"extracting query codes for {sev} ...")
        query_codes, query_labels, query_paths = extract_codes_labels_paths(
            model=model, dataloader=query_loader, device=device,
        )
        metrics = compute_map_from_codes(
            query_codes=query_codes, query_labels=query_labels,
            gallery_codes=gallery_codes, gallery_labels=gallery_labels,
            query_paths=query_paths, gallery_paths=gallery_paths,
            exclude_self=True,  # 与训练评测保持一致
        )
        results[f"mAP_{sev}"] = float(metrics["mAP"])
        print(f"mAP_{sev}: {metrics['mAP']:.6f}")

    results["robust_avg"] = (
        results["mAP_weak"] + results["mAP_medium"] + results["mAP_strong"]
    ) / 3.0
    results["robust_avg_4"] = (
        results["mAP_clean"] + results["mAP_weak"]
        + results["mAP_medium"] + results["mAP_strong"]
    ) / 4.0

    print(f"\nrobust_avg (3档, 不含clean): {results['robust_avg']:.6f}")
    print(f"robust_avg_4 (4档):          {results['robust_avg_4']:.6f}")

    order_ok = (
        results["mAP_clean"] > results["mAP_weak"] >
        results["mAP_medium"] > results["mAP_strong"]
    )
    results["order_ok"] = int(order_ok)
    if order_ok:
        print("\n✅ GATE PASSED: clean > weak > medium > strong")
    else:
        print("\n❌ GATE FAILED: 顺序不满足")

    summary = {
        "checkpoint": str(ckpt_path),
        "dataset": args.dataset, "split_seed": args.split_seed,
        "image_size": args.image_size, "device": str(device),
        "num_classes": num_classes, "hash_bits": hash_bits,
        "results": results,
    }

    if args.save_json:
        out_path = ckpt_path.parent / "test_metrics.json"
        save_json(out_path, summary)
        print(f"saved: {out_path}")

    print("=== EVAL CHECKPOINT FINISHED ===")


if __name__ == "__main__":
    main()