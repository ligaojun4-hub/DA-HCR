from pathlib import Path
import argparse
import json
import os
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models

def resolve_image_path(row, project_root: Path) -> Path:
    for key in ["abs_path", "path"]:
        if key in row and isinstance(row[key], str) and row[key]:
            p = Path(row[key])
            if p.exists():
                return p
    for key in ["rel_path", "image_path", "filepath"]:
        if key in row and isinstance(row[key], str) and row[key]:
            p = project_root / row[key]
            if p.exists():
                return p
    raise FileNotFoundError(f"Cannot resolve image path from row keys={list(row.keys())}")

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--bits", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    project_root = Path.cwd()
    out_pt = Path(args.out)
    out_pt.parent.mkdir(parents=True, exist_ok=True)

    manifest = project_root / "data" / "splits" / args.dataset / f"seed_{args.split_seed}" / "split_manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"ERROR: split manifest not found: {manifest}")

    df = pd.read_csv(manifest)
    required = {"split", "label", "class_name"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: split_manifest missing columns: {missing}; columns={list(df.columns)}")

    train_df = df[df["split"] == "train"].copy()
    non_train_count = len(df[df["split"] != "train"])
    if len(train_df) == 0:
        raise SystemExit("ERROR: no train rows found")

    train_df["label"] = train_df["label"].astype(int)
    num_classes = int(train_df["label"].max()) + 1

    label_to_names = train_df.groupby("label")["class_name"].nunique()
    if int(label_to_names.max()) != 1:
        bad = label_to_names[label_to_names != 1]
        raise SystemExit(f"ERROR: label maps to multiple class names: {bad.head()}")

    class_names = (
        train_df.sort_values("label")
        .groupby("label")["class_name"]
        .first()
        .reindex(range(num_classes))
        .tolist()
    )

    counts = train_df.groupby("label").size().reindex(range(num_classes), fill_value=0).astype(int)
    if counts.min() <= 0:
        raise SystemExit(f"ERROR: some classes have zero train samples: {counts[counts <= 0].head()}")

    print("CENTER_SOURCE: TRAIN_ONLY")
    print("dataset:", args.dataset)
    print("manifest:", manifest)
    print("all_rows:", len(df))
    print("train_rows:", len(train_df))
    print("non_train_rows_not_used:", non_train_count)
    print("num_classes:", num_classes)
    print("train_count_min:", int(counts.min()))
    print("train_count_max:", int(counts.max()))
    print("train_count_mean:", float(counts.mean()))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        net = models.resnet18(weights=weights)
    except Exception:
        net = models.resnet18(pretrained=True)

    net.fc = torch.nn.Identity()
    net.eval().to(device)

    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    feat_sum = None
    feat_counts = torch.zeros(num_classes, dtype=torch.float32)
    batch_x, batch_y = [], []

    def flush():
        nonlocal feat_sum, batch_x, batch_y
        if not batch_x:
            return
        x = torch.stack(batch_x).to(device)
        y = torch.tensor(batch_y, dtype=torch.long)
        f = net(x).detach().cpu()
        if feat_sum is None:
            feat_sum = torch.zeros(num_classes, f.shape[1], dtype=torch.float32)
        for i, lab in enumerate(y.tolist()):
            feat_sum[lab] += f[i]
            feat_counts[lab] += 1
        batch_x, batch_y = [], []

    rows = train_df.to_dict("records")
    total = len(rows)

    for i, row in enumerate(rows, 1):
        try:
            img_path = resolve_image_path(row, project_root)
            img = Image.open(img_path).convert("RGB")
            batch_x.append(tfm(img))
            batch_y.append(int(row["label"]))
        except Exception as e:
            print("SKIP_IMAGE:", e)
            continue

        if len(batch_x) >= 64:
            flush()

        if i % 1000 == 0:
            print(f"processed {i} / {total}")

    flush()

    if feat_sum is None:
        raise SystemExit("ERROR: no features extracted")

    if int((feat_counts <= 0).sum().item()) > 0:
        raise SystemExit("ERROR: some labels have zero extracted features")

    proto = feat_sum / feat_counts.clamp_min(1).view(-1, 1)
    proto = F.normalize(proto, dim=1)

    g = torch.Generator(device="cpu")
    seed = 2025 + num_classes + args.bits
    g.manual_seed(seed)
    proj = torch.randn(proto.shape[1], args.bits, generator=g)
    raw = proto @ proj
    centers = torch.where(raw >= 0, torch.ones_like(raw), -torch.ones_like(raw)).float()

    for b in range(args.bits):
        if abs(centers[:, b].sum().item()) == centers.shape[0]:
            centers[::2, b] *= -1

    payload = {
        "centers": centers,
        "class_names": class_names,
        "counts": feat_counts,
        "bits": args.bits,
        "num_classes": num_classes,
        "dataset": args.dataset,
        "split_seed": args.split_seed,
        "center_source": "train_only",
        "source_manifest": str(manifest),
        "method": "SHC-inspired train-only data-dependent semantic centers from ResNet18 visual prototypes",
    }

    torch.save(payload, out_pt)

    meta = {
        "out_pt": str(out_pt),
        "dataset": args.dataset,
        "split_seed": args.split_seed,
        "center_source": "train_only",
        "source_manifest": str(manifest),
        "num_classes": num_classes,
        "bits": args.bits,
        "center_shape": list(centers.shape),
        "all_manifest_rows": int(len(df)),
        "train_rows_used": int(len(train_df)),
        "non_train_rows_not_used": int(non_train_count),
        "min_train_count": float(feat_counts.min().item()),
        "max_train_count": float(feat_counts.max().item()),
        "mean_train_count": float(feat_counts.mean().item()),
        "first_10_class_names": class_names[:10],
        "last_10_class_names": class_names[-10:],
    }

    out_json = out_pt.with_suffix(".json")
    out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("CENTER_DONE_TRAIN_ONLY")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
