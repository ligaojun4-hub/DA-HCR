#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import torch

from robust_hash.models.hash_model import RobustHashModel
from robust_hash.evaluation.retrieval import extract_codes_labels_paths, hamming_distance_matrix


SEVS = ["clean", "weak", "medium", "strong"]


def load_eval_module():
    p = Path("scripts/07_eval_checkpoint.py")
    spec = importlib.util.spec_from_file_location("eval_checkpoint_mod", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pick_state_dict(obj):
    if not isinstance(obj, dict):
        raise RuntimeError("checkpoint is not a dict")

    for k in ["model_state_dict", "state_dict", "model", "net"]:
        if k in obj and isinstance(obj[k], dict):
            return obj[k]

    # fallback: checkpoint itself may be state dict
    tensor_count = sum(1 for v in obj.values() if torch.is_tensor(v))
    if tensor_count > 5:
        return obj

    raise RuntimeError("cannot find model state dict in checkpoint")


def strip_module_prefix(state):
    if not state:
        return state
    keys = list(state.keys())
    if keys and all(k.startswith("module.") for k in keys[: min(10, len(keys))]):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def build_model(num_classes, hash_bits, checkpoint, device):
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = strip_module_prefix(pick_state_dict(ckpt))

    last_err = None
    for kwargs in [
        {"num_classes": num_classes, "hash_bits": hash_bits},
        {"n_classes": num_classes, "hash_bits": hash_bits},
        {"num_classes": num_classes, "bits": hash_bits},
        {"hash_bits": hash_bits, "num_classes": num_classes},
    ]:
        try:
            model = RobustHashModel(**kwargs)
            msg = model.load_state_dict(state, strict=False)
            model.to(device)
            model.eval()
            print("MODEL_LOAD_OK")
            print("missing_keys:", len(msg.missing_keys))
            print("unexpected_keys:", len(msg.unexpected_keys))
            return model
        except Exception as e:
            last_err = e

    raise RuntimeError(f"failed to build RobustHashModel: {last_err}")


def to_tensor(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    return torch.as_tensor(x)


def distance_matrix(q_codes, g_codes):
    try:
        d = hamming_distance_matrix(q_codes, g_codes)
        return to_tensor(d).float()
    except Exception:
        q = to_tensor(q_codes).float()
        g = to_tensor(g_codes).float()
        q = torch.sign(q)
        g = torch.sign(g)
        q[q == 0] = 1
        g[g == 0] = 1
        return (q.shape[1] - q @ g.t()) / 2.0


def compute_pk_recall(q_codes, q_labels, g_codes, g_labels, k_list):
    q_labels = to_tensor(q_labels).long().view(-1)
    g_labels = to_tensor(g_labels).long().view(-1)

    dist = distance_matrix(q_codes, g_codes)
    max_k = min(max(k_list), dist.shape[1])
    order = torch.topk(dist, k=max_k, largest=False, dim=1).indices

    sorted_labels = g_labels[order]
    match = sorted_labels.eq(q_labels.view(-1, 1))

    total_rel = g_labels.view(1, -1).eq(q_labels.view(-1, 1)).sum(dim=1).clamp_min(1)

    row = {}
    for k in k_list:
        kk = min(k, max_k)
        hit = match[:, :kk].sum(dim=1).float()
        row[f"P@{k}"] = float((hit / kk).mean().item())
        row[f"Recall@{k}"] = float((hit / total_rel.float()).mean().item())

    pr_rows = []
    for k in [1, 5, 10, 20, 50, 100, 200, 500, 1000]:
        if k > dist.shape[1]:
            continue
        kk = min(k, max_k)
        hit = match[:, :kk].sum(dim=1).float()
        pr_rows.append({
            "K": k,
            "Precision": float((hit / kk).mean().item()),
            "Recall": float((hit / total_rel.float()).mean().item()),
        })

    return row, pr_rows


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--hash-bits", type=int, default=64)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    print("===== EXTRA P@K / RECALL EVAL =====")
    print("checkpoint:", args.checkpoint)
    print("dataset:", args.dataset)
    print("num_classes:", args.num_classes)
    print("hash_bits:", args.hash_bits)
    print("device:", device)
    print("out_dir:", args.out_dir)

    eval_mod = load_eval_module()
    model = build_model(args.num_classes, args.hash_bits, args.checkpoint, device)

    print("building gallery loader: test_retrieval clean")
    gallery_loader = eval_mod.build_loader(
        args.dataset, args.split_seed, "test_retrieval", "clean",
        args.image_size, args.batch_size, args.num_workers, device.type, 0
    )
    g_codes, g_labels, g_paths = extract_codes_labels_paths(model, gallery_loader, device)
    print("gallery size:", len(g_labels))

    k_list = [1, 5, 10, 50, 100, 500, 1000]
    rows = []
    pr_all = []

    for sev in SEVS:
        print("query severity:", sev)
        query_loader = eval_mod.build_loader(
            args.dataset, args.split_seed, "test_query", sev,
            args.image_size, args.batch_size, args.num_workers, device.type, 0
        )
        q_codes, q_labels, q_paths = extract_codes_labels_paths(model, query_loader, device)

        m, pr = compute_pk_recall(q_codes, q_labels, g_codes, g_labels, k_list)

        row = {
            "tag": args.tag,
            "dataset": args.dataset,
            "severity": sev,
            "checkpoint": args.checkpoint,
            "num_query": len(q_labels),
            "num_gallery": len(g_labels),
            "hash_bits": args.hash_bits,
        }
        row.update(m)
        rows.append(row)

        for x in pr:
            y = {
                "tag": args.tag,
                "dataset": args.dataset,
                "severity": sev,
                "checkpoint": args.checkpoint,
            }
            y.update(x)
            pr_all.append(y)

        print(json.dumps(row, indent=2))

    avg = {
        "tag": args.tag,
        "dataset": args.dataset,
        "severity": "robust_avg_weak_medium_strong",
        "checkpoint": args.checkpoint,
        "num_query": rows[0]["num_query"],
        "num_gallery": rows[0]["num_gallery"],
        "hash_bits": args.hash_bits,
    }

    weak_rows = [r for r in rows if r["severity"] in ["weak", "medium", "strong"]]
    for k in [x for x in rows[0] if x.startswith("P@") or x.startswith("Recall@")]:
        avg[k] = sum(float(r[k]) for r in weak_rows) / len(weak_rows)
    rows.append(avg)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out / "pk_recall_by_severity.csv", rows)
    save_csv(out / "pr_curve_points.csv", pr_all)
    (out / "extra_pk_recall.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("===== DONE =====")
    print("saved:", out / "pk_recall_by_severity.csv")
    print("saved:", out / "pr_curve_points.csv")


if __name__ == "__main__":
    main()
