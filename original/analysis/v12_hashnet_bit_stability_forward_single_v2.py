from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from robust_hash.data.datasets import RetrievalEvalDataset
from robust_hash.models.hash_model import RobustHashModel


def load_model(run_dir: Path, device: torch.device) -> RobustHashModel:
    ckpt_path = run_dir / "best_val.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    num_classes = int(checkpoint["num_classes"])
    hash_bits = int(checkpoint.get("args", {}).get("hash_bits", 64))

    model = RobustHashModel(
        num_classes=num_classes,
        hash_bits=hash_bits,
        backbone_name="resnet18",
        pretrained=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    print(f"loaded checkpoint: {ckpt_path}")
    print(f"num_classes={num_classes}, hash_bits={hash_bits}")
    return model


def unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    if isinstance(batch, dict):
        x = batch["image"]
        y = batch["label"]
        rel = batch.get("rel_path", batch.get("path", [""] * len(y)))

        if torch.is_tensor(y):
            y = y.detach().cpu().long()
        else:
            y = torch.tensor(y, dtype=torch.long)

        if isinstance(rel, str):
            rel = [rel]

        return x, y, list(rel)

    if isinstance(batch, (tuple, list)):
        x = batch[0]
        y = batch[1]
        rel = batch[2] if len(batch) > 2 else [""] * len(y)

        if torch.is_tensor(y):
            y = y.detach().cpu().long()
        else:
            y = torch.tensor(y, dtype=torch.long)

        return x, y, list(rel)

    raise RuntimeError(f"Unsupported batch type: {type(batch)}")


@torch.no_grad()
def extract_codes(
    model: RobustHashModel,
    dataset: str,
    split_seed: int,
    split_name: str,
    severity: str,
    image_size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> Dict[str, Any]:
    ds = RetrievalEvalDataset(
        dataset=dataset,
        split_seed=split_seed,
        split_name=split_name,
        image_size=image_size,
        severity=severity,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )

    zs = []
    bs = []
    ys = []
    rels: List[str] = []

    for batch in loader:
        x, y, rel = unpack_batch(batch)
        x = x.to(device, non_blocking=True)

        out = model.forward_single(x)

        if not isinstance(out, dict):
            raise RuntimeError(f"forward_single output is not dict: {type(out)}")
        if "z" not in out or "b" not in out:
            raise RuntimeError(f"forward_single output missing z/b. keys={list(out.keys())}")

        z = out["z"].detach().float().cpu()
        b = out["b"].detach().float().cpu()

        if z.ndim != 2:
            raise RuntimeError(f"z should be 2D, got shape={tuple(z.shape)}")
        if b.shape != z.shape:
            raise RuntimeError(f"b shape {tuple(b.shape)} != z shape {tuple(z.shape)}")

        zs.append(z)
        bs.append(b)
        ys.append(y.cpu().long())
        rels.extend(rel)

    z_all = torch.cat(zs, dim=0)
    b_all = torch.cat(bs, dim=0)
    y_all = torch.cat(ys, dim=0)

    return {
        "z": z_all,
        "b": b_all,
        "y": y_all,
        "rel_path": rels,
        "severity": severity,
        "num_samples": int(z_all.shape[0]),
        "hash_bits": int(z_all.shape[1]),
    }


def align_by_rel(clean: Dict[str, Any], degraded: Dict[str, Any]):
    clean_map = {p: i for i, p in enumerate(clean["rel_path"])}
    deg_map = {p: i for i, p in enumerate(degraded["rel_path"])}
    common = [p for p in clean["rel_path"] if p in deg_map]

    if not common:
        raise RuntimeError("No common rel_path between clean and degraded outputs.")

    ci = torch.tensor([clean_map[p] for p in common], dtype=torch.long)
    di = torch.tensor([deg_map[p] for p in common], dtype=torch.long)

    return clean["z"][ci], clean["b"][ci], degraded["z"][di], degraded["b"][di]


def compute_metrics(clean_z, clean_b, deg_z, deg_b) -> Dict[str, float]:
    bit_diff = (clean_b != deg_b).float()
    hamming = bit_diff.sum(dim=1)

    bit_flip_rate = bit_diff.mean().item()
    sign_consistency = 1.0 - bit_flip_rate

    cos_z = F.cosine_similarity(clean_z, deg_z, dim=1)
    l2_z = torch.norm(clean_z - deg_z, p=2, dim=1)

    abs_clean = torch.abs(clean_z)
    abs_deg = torch.abs(deg_z)

    return {
        "num_samples": int(clean_z.shape[0]),
        "hash_bits": int(clean_z.shape[1]),
        "bit_flip_rate": bit_flip_rate,
        "sign_consistency": sign_consistency,
        "mean_hamming": hamming.mean().item(),
        "median_hamming": hamming.median().item(),
        "std_hamming": hamming.std(unbiased=True).item() if hamming.numel() > 1 else 0.0,
        "mean_cosine_z": cos_z.mean().item(),
        "mean_l2_z": l2_z.mean().item(),
        "mean_abs_clean_z": abs_clean.mean().item(),
        "mean_abs_degraded_z": abs_deg.mean().item(),
        "near_zero_rate_clean_abs_lt_0.2": (abs_clean < 0.2).float().mean().item(),
        "near_zero_rate_degraded_abs_lt_0.2": (abs_deg < 0.2).float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="caltech256")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-name", default="test_query")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--da-run", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    severities = ["clean", "weak", "medium", "strong"]
    runs = {
        "base": Path(args.base_run),
        "dasupcon": Path(args.da_run),
    }

    all_rows = []
    strength = {}

    for method, run_dir in runs.items():
        print(f"\n========== METHOD {method} ==========")
        model = load_model(run_dir, device)

        outputs = {}

        for sev in severities:
            print(f"extracting method={method}, severity={sev}")
            outputs[sev] = extract_codes(
                model=model,
                dataset=args.dataset,
                split_seed=args.split_seed,
                split_name=args.split_name,
                severity=sev,
                image_size=args.image_size,
                batch_size=args.batch_size,
                workers=args.workers,
                device=device,
            )
            print(
                f"  num_samples={outputs[sev]['num_samples']}, "
                f"hash_bits={outputs[sev]['hash_bits']}"
            )

        for sev in ["weak", "medium", "strong"]:
            clean_z, clean_b, deg_z, deg_b = align_by_rel(outputs["clean"], outputs[sev])
            metrics = compute_metrics(clean_z, clean_b, deg_z, deg_b)

            row = {
                "protocol": "formal_v12_monotonic",
                "dataset": args.dataset,
                "baseline": "hashnet",
                "method": method,
                "split_seed": args.split_seed,
                "split_name": args.split_name,
                "severity_pair": f"clean_to_{sev}",
                **metrics,
                "source_run_dir": str(run_dir),
            }

            all_rows.append(row)

        for sev in severities:
            z = outputs[sev]["z"]
            b = outputs[sev]["b"]

            strength[f"{method}_{sev}"] = {
                "method": method,
                "severity": sev,
                "num_samples": int(z.shape[0]),
                "hash_bits": int(z.shape[1]),
                "mean_abs_z": torch.abs(z).mean().item(),
                "median_abs_z": torch.abs(z).median().item(),
                "near_zero_rate_abs_lt_0.2": (torch.abs(z) < 0.2).float().mean().item(),
                "bit_balance_mean_abs_bit_mean": torch.abs(b.mean(dim=0)).mean().item(),
            }

    csv_path = out_dir / "caltech256_hashnet_bit_stability_mechanism_v12.csv"
    json_path = out_dir / "caltech256_hashnet_code_strength_mechanism_v12.json"

    fields = [
        "protocol",
        "dataset",
        "baseline",
        "method",
        "split_seed",
        "split_name",
        "severity_pair",
        "num_samples",
        "hash_bits",
        "bit_flip_rate",
        "sign_consistency",
        "mean_hamming",
        "median_hamming",
        "std_hamming",
        "mean_cosine_z",
        "mean_l2_z",
        "mean_abs_clean_z",
        "mean_abs_degraded_z",
        "near_zero_rate_clean_abs_lt_0.2",
        "near_zero_rate_degraded_abs_lt_0.2",
        "source_run_dir",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    json_path.write_text(json.dumps(strength, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n========== BIT STABILITY MECHANISM ROWS ==========")
    for r in all_rows:
        print(
            f"{r['method']:8s} {r['severity_pair']:16s} | "
            f"flip={r['bit_flip_rate']:.6f} | "
            f"sign={r['sign_consistency']:.6f} | "
            f"hamming={r['mean_hamming']:.4f} | "
            f"cos={r['mean_cosine_z']:.6f} | "
            f"near0_deg={r['near_zero_rate_degraded_abs_lt_0.2']:.6f}"
        )

    print("\n========== DA IMPROVEMENT OVER BASE IN BIT STABILITY ==========")
    for sev in ["weak", "medium", "strong"]:
        pair = f"clean_to_{sev}"

        base = next(r for r in all_rows if r["method"] == "base" and r["severity_pair"] == pair)
        da = next(r for r in all_rows if r["method"] == "dasupcon" and r["severity_pair"] == pair)

        print(
            f"{pair}: "
            f"base_flip={base['bit_flip_rate']:.6f}, "
            f"da_flip={da['bit_flip_rate']:.6f}, "
            f"delta_flip={da['bit_flip_rate'] - base['bit_flip_rate']:.6f}; "
            f"base_sign={base['sign_consistency']:.6f}, "
            f"da_sign={da['sign_consistency']:.6f}, "
            f"delta_sign={da['sign_consistency'] - base['sign_consistency']:.6f}; "
            f"base_ham={base['mean_hamming']:.4f}, "
            f"da_ham={da['mean_hamming']:.4f}, "
            f"delta_ham={da['mean_hamming'] - base['mean_hamming']:.4f}"
        )

    print("\nsaved:", csv_path)
    print("saved:", json_path)


if __name__ == "__main__":
    main()
