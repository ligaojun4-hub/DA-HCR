import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robust_hash.data.datasets import PairTrainDataset, RetrievalEvalDataset
from robust_hash.data.splits import load_split_manifest
from robust_hash.evaluation.retrieval import evaluate_query_gallery_map
from robust_hash.losses.hash_losses_v3_dasupcon import compute_total_loss_v3_dasupcon
from robust_hash.models.hash_model import RobustHashModel
from robust_hash.utils.seed import set_global_seed



def get_current_hashnet_beta(epoch: int, base_beta: float) -> float:
    """
    HashNet continuation schedule.
    The beta value increases by stages so tanh(beta*z) gradually approaches sign(z).
    """
    if epoch <= 4:
        return base_beta
    if epoch <= 8:
        return base_beta * 2.0
    if epoch <= 12:
        return base_beta * 4.0
    return base_beta * 8.0


def build_run_dir(project_root, dataset, baseline, method, split_seed, hash_bits):
    forced = os.environ.get("DAHCR_FORCE_RUN_DIR", "").strip()
    if forced:
        return Path(forced)
    return project_root / "outputs" / "formal_v23_nhl_nested_hash_tune" / "train_runs" / dataset / baseline / method / f"seed_{split_seed}" / f"bits_{hash_bits}"


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def infer_num_classes(dataset: str, split_seed: int) -> int:
    df = load_split_manifest(dataset=dataset, split_seed=split_seed)
    return int(df["label"].nunique())


def make_random_subset(dataset, max_samples: int, seed: int):
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:max_samples].tolist()
    return Subset(dataset, indices)


def save_checkpoint(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def parse_eval_epochs(eval_epochs_str: str) -> Set[int]:
    out = set()
    for x in eval_epochs_str.split(','):
        x = x.strip()
        if x:
            out.add(int(x))
    return out


def evaluate_full_val_metrics(model, dataset, split_seed, image_size, eval_batch_size, eval_num_workers, device):
    severities = ["clean", "weak", "medium", "strong"]
    gallery_ds = RetrievalEvalDataset(dataset=dataset, split_seed=split_seed, split_name="val", image_size=image_size, severity="clean")
    gallery_loader = DataLoader(gallery_ds, batch_size=eval_batch_size, shuffle=False, num_workers=eval_num_workers, pin_memory=(device.type == "cuda"))
    out = {}
    for sev in severities:
        print(f"  [full val] severity={sev}")
        query_ds = RetrievalEvalDataset(dataset=dataset, split_seed=split_seed, split_name="val", image_size=image_size, severity=sev)
        query_loader = DataLoader(query_ds, batch_size=eval_batch_size, shuffle=False, num_workers=eval_num_workers, pin_memory=(device.type == "cuda"))
        metrics = evaluate_query_gallery_map(model=model, query_loader=query_loader, gallery_loader=gallery_loader, device=device, exclude_self=True)
        out[f"mAP_{sev}"] = float(metrics["mAP"])
    out["robust_avg"] = (out["mAP_weak"] + out["mAP_medium"] + out["mAP_strong"]) / 3.0
    out["robust_avg_4"] = (out["mAP_clean"] + out["mAP_weak"] + out["mAP_medium"] + out["mAP_strong"]) / 4.0
    out["order_ok"] = float(out["mAP_clean"] > out["mAP_weak"] > out["mAP_medium"] > out["mAP_strong"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--baseline", type=str, default="greedyhash", choices=["greedyhash", "dpn", "hashnet", "dhn", "dpsh", "dsh", "shc", 'nhl'])
    parser.add_argument("--method", type=str, default="base", choices=["base", "dasupcon"])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--hash-bits", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--eval-num-workers", type=int, default=24)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--same-image-weight", type=float, default=2.0)
    parser.add_argument("--same-class-weight", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--eval-epochs", type=str, default="10,14,18,22,26,30")
    parser.add_argument("--paired-severity", type=str, default="weak,medium")
    args = parser.parse_args()

    set_global_seed(args.split_seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    eval_epochs = {e for e in parse_eval_epochs(args.eval_epochs) if 1 <= e <= args.max_epochs}
    eval_epochs.add(args.max_epochs)
    num_classes = infer_num_classes(args.dataset, args.split_seed)

    run_dir = build_run_dir(PROJECT_ROOT, args.dataset, args.baseline, args.method, args.split_seed, args.hash_bits)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=== V23 NHL-STYLE RUN INFO ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print(f"num_classes: {num_classes}")
    print(f"device: {device}")
    print(f"run_dir: {run_dir}")
    print(f"eval_epochs: {sorted(eval_epochs)}")

    # Fair comparison rule:
    # Base and DA-SupCon must use the same paired severities.
    # Do not force base to clean-only.
    paired_severities = [s.strip() for s in args.paired_severity.split(',') if s.strip()]

    train_datasets = []
    for psev in paired_severities:
        ds = PairTrainDataset(dataset=args.dataset, split_seed=args.split_seed, image_size=args.image_size, paired_severity=psev)
        train_datasets.append(ds)
        print(f"  train paired_severity={psev} -> size={len(ds)}")

    combined_train_ds = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
    combined_train_ds = make_random_subset(combined_train_ds, max_samples=args.max_train_samples, seed=args.split_seed)
    print(f"combined_train size: {len(combined_train_ds)}")

    train_loader = DataLoader(combined_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    model = RobustHashModel(num_classes=num_classes, hash_bits=args.hash_bits, backbone_name="resnet18", pretrained=args.pretrained).to(device)

    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if any(k in name.lower() for k in ["backbone", "encoder", "conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "resnet"]):
            backbone_params.append(p)
        else:
            head_params.append(p)
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)

    history = []
    best_val_score = -1.0
    best_val_epoch = -1
    save_json(run_dir / "config.json", vars(args))

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        sums = {"total": 0.0, "base": 0.0, "ce": 0.0, "pair": 0.0, "quant": 0.0, "polar": 0.0, "ortho": 0.0, "dasupcon": 0.0, "eff": 0.0, "ratio": 0.0}
        steps = 0
        for step_idx, batch in enumerate(train_loader, start=1):
            clean_x = batch["clean_image"].to(device, non_blocking=True)
            paired_x = batch["paired_image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model.forward_pair(clean_x, paired_x)
            loss = compute_total_loss_v3_dasupcon(
                outputs=outputs,
                labels=labels,
                baseline=args.baseline,
                method=args.method,
                alpha=args.alpha,
                temperature=args.temperature,
                same_image_weight=args.same_image_weight,
                same_class_weight=args.same_class_weight,
                gamma=args.gamma,
                beta=get_current_hashnet_beta(epoch, args.beta) if args.baseline == 'hashnet' else args.beta,
                pairwise_weight=args.pairwise_weight,
            )
            loss["total_loss"].backward()
            optimizer.step()
            sums["total"] += float(loss["total_loss"].detach())
            sums["base"] += float(loss["base_loss"].detach())
            sums["ce"] += float(loss["ce_loss"].detach())
            sums["pair"] += float(loss["pair_loss"].detach())
            sums["quant"] += float(loss["quant_loss"].detach())
            sums["polar"] += float(loss["polar_loss"].detach())
            sums["ortho"] += float(loss["ortho_loss"].detach())
            sums["dasupcon"] += float(loss["dasupcon_loss"].detach())
            sums["eff"] += float(loss["effective_dasupcon"].detach())
            sums["ratio"] += float(loss["effective_ratio"].detach())
            steps += 1
            if step_idx % args.log_interval == 0 or step_idx == 1:
                print(f"[Epoch {epoch} | Step {step_idx}/{len(train_loader)}] total={loss['total_loss'].item():.4f} base={loss['base_loss'].item():.4f} ce={loss['ce_loss'].item():.4f} pair={loss['pair_loss'].item():.4f} quant={loss['quant_loss'].item():.4f} dasupcon={loss['dasupcon_loss'].item():.4f} eff={loss['effective_dasupcon'].item():.4f} ratio={loss['effective_ratio'].item():.4f}")

        epoch_log = {
            "epoch": epoch,
            "train_total_loss": sums["total"] / max(steps, 1),
            "train_base_loss": sums["base"] / max(steps, 1),
            "train_ce_loss": sums["ce"] / max(steps, 1),
            "train_pair_loss": sums["pair"] / max(steps, 1),
            "train_quant_loss": sums["quant"] / max(steps, 1),
            "train_polar_loss": sums["polar"] / max(steps, 1),
            "train_ortho_loss": sums["ortho"] / max(steps, 1),
            "train_dasupcon_loss": sums["dasupcon"] / max(steps, 1),
            "train_effective_dasupcon": sums["eff"] / max(steps, 1),
            "train_effective_ratio": sums["ratio"] / max(steps, 1),
        }

        save_checkpoint(run_dir / "last.pt", {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "args": vars(args), "num_classes": num_classes, "metrics": epoch_log})

        if epoch in eval_epochs:
            print(f"\n[Epoch {epoch}] start FULL val ...")
            val_metrics = evaluate_full_val_metrics(model, args.dataset, args.split_seed, args.image_size, args.eval_batch_size, args.eval_num_workers, device)
            epoch_log.update(val_metrics)
            payload = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "args": vars(args), "num_classes": num_classes, "metrics": epoch_log}
            save_checkpoint(run_dir / f"eval_epoch_{epoch:03d}.pt", payload)
            if epoch_log["robust_avg"] > best_val_score:
                best_val_score = epoch_log["robust_avg"]
                best_val_epoch = epoch
                save_checkpoint(run_dir / "best_val.pt", payload)
        else:
            epoch_log["val_skipped"] = 1.0

        history.append(epoch_log)
        save_json(run_dir / "history.json", history)
        print(f"\n=== EPOCH {epoch} SUMMARY ===")
        for k, v in epoch_log.items():
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    save_json(run_dir / "summary.json", {"best_val_epoch": best_val_epoch, "best_val_robust_avg": best_val_score, "history_len": len(history)})


if __name__ == "__main__":
    main()
