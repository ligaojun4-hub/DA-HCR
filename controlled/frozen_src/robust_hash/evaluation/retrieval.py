from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_codes_labels_paths(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    提取哈希码、标签、路径。

    返回:
        codes:  [N, B] in {-1, +1}
        labels: [N]
        paths:  list[str]
    """
    model.eval()

    all_codes = []
    all_labels = []
    all_paths: List[str] = []

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        outputs = model.forward_single(images)
        codes = outputs["b"]  # already STE sign output in {-1, +1}

        all_codes.append(codes.detach().cpu())
        all_labels.append(labels.detach().cpu())

        if "rel_path" in batch:
            batch_paths = list(batch["rel_path"])
        else:
            batch_paths = list(batch["abs_path"])
        all_paths.extend(batch_paths)

    codes = torch.cat(all_codes, dim=0).float()
    labels = torch.cat(all_labels, dim=0).long()
    return codes, labels, all_paths


@torch.no_grad()
def extract_codes_and_labels(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    兼容旧接口。
    """
    codes, labels, _ = extract_codes_labels_paths(
        model=model,
        dataloader=dataloader,
        device=device,
    )
    return codes, labels


def hamming_distance_matrix(
    query_codes: torch.Tensor,
    gallery_codes: torch.Tensor,
) -> torch.Tensor:
    """
    对于 {-1, +1} 哈希码:
        Hamming(x, y) = 0.5 * (B - x·y)

    返回:
        dist: [Nq, Ng]
    """
    num_bits = query_codes.shape[1]
    sim = torch.matmul(query_codes, gallery_codes.t())  # [Nq, Ng]
    dist = 0.5 * (num_bits - sim)
    return dist


def compute_map_from_codes(
    query_codes: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_codes: torch.Tensor,
    gallery_labels: torch.Tensor,
    query_paths: Optional[List[str]] = None,
    gallery_paths: Optional[List[str]] = None,
    exclude_self: bool = False,
) -> Dict[str, float]:
    """
    最基础的 single-label mAP@all。

    当 query/gallery 来自同一个 split（例如 val 对 val）时，
    可以通过 exclude_self=True 且提供 query_paths/gallery_paths 来排除 self-match。
    """
    dist = hamming_distance_matrix(query_codes, gallery_codes)  # [Nq, Ng]
    num_query = query_codes.shape[0]

    ap_list = []

    for i in range(num_query):
        q_label = query_labels[i]
        d_i = dist[i]  # [Ng]

        sort_idx = torch.argsort(d_i, dim=0, descending=False).tolist()

        filtered_idx = []
        for idx in sort_idx:
            if (
                exclude_self
                and query_paths is not None
                and gallery_paths is not None
                and gallery_paths[idx] == query_paths[i]
            ):
                continue
            filtered_idx.append(idx)

        if len(filtered_idx) == 0:
            continue

        filtered_idx_tensor = torch.tensor(filtered_idx, dtype=torch.long)
        ranked_labels = gallery_labels[filtered_idx_tensor]

        relevant = (ranked_labels == q_label).float()
        num_relevant = int(relevant.sum().item())

        if num_relevant == 0:
            continue

        cumsum_rel = torch.cumsum(relevant, dim=0)
        ranks = torch.arange(1, relevant.shape[0] + 1, dtype=torch.float32)

        precision_at_k = cumsum_rel / ranks
        ap = (precision_at_k * relevant).sum() / num_relevant
        ap_list.append(float(ap.item()))

    mean_ap = 0.0 if len(ap_list) == 0 else sum(ap_list) / len(ap_list)

    return {
        "mAP": mean_ap,
        "num_query_used": len(ap_list),
        "num_query_total": num_query,
    }


@torch.no_grad()
def evaluate_query_gallery_map(
    model,
    query_loader: DataLoader,
    gallery_loader: DataLoader,
    device: torch.device,
    exclude_self: bool = False,
) -> Dict[str, float]:
    """
    提取 query / gallery 哈希码并计算 mAP。
    """
    query_codes, query_labels, query_paths = extract_codes_labels_paths(
        model=model,
        dataloader=query_loader,
        device=device,
    )
    gallery_codes, gallery_labels, gallery_paths = extract_codes_labels_paths(
        model=model,
        dataloader=gallery_loader,
        device=device,
    )

    metrics = compute_map_from_codes(
        query_codes=query_codes,
        query_labels=query_labels,
        gallery_codes=gallery_codes,
        gallery_labels=gallery_labels,
        query_paths=query_paths,
        gallery_paths=gallery_paths,
        exclude_self=exclude_self,
    )

    metrics["query_size"] = int(query_codes.shape[0])
    metrics["gallery_size"] = int(gallery_codes.shape[0])
    metrics["hash_bits"] = int(query_codes.shape[1])

    return metrics