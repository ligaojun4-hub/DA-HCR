from typing import Dict
import torch
import torch.nn.functional as F


def _hard_sign(z: torch.Tensor) -> torch.Tensor:
    b = torch.sign(z)
    return torch.where(b == 0, torch.ones_like(b), b)


def quantization_loss(z: torch.Tensor) -> torch.Tensor:
    b = _hard_sign(z).detach()
    return ((z - b) ** 2).mean()


def _pair_targets(labels: torch.Tensor):
    n = labels.shape[0]
    device = labels.device
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)
    target = (labels.view(-1, 1) == labels.view(1, -1)).float()
    return target[mask], mask


def _pairwise_logistic_loss_from_logits(pair_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pair_logits.numel() == 0:
        return pair_logits.new_tensor(0.0)
    return (F.softplus(pair_logits) - target * pair_logits).mean()


def _weighted_hashnet_pairwise_loss(z: torch.Tensor, labels: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    h = torch.tanh(beta * z)
    target, mask = _pair_targets(labels)
    theta = 0.5 * (h @ h.t())
    vals = theta[mask]
    if vals.numel() == 0:
        return z.new_tensor(0.0)
    pos = target.sum()
    neg = target.numel() - pos
    weight = torch.ones_like(target)
    if pos.item() > 0 and neg.item() > 0:
        weight = torch.where(target > 0.5, (pos + neg) / (pos + 1e-6), (pos + neg) / (neg + 1e-6))
    loss = F.softplus(vals) - target * vals
    return (weight * loss).mean()


def _dpsh_pairwise_loss(z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    DPSH core pairwise logistic likelihood.

    theta_ij = 0.5 * u_i^T u_j
    L_pair = log(1 + exp(theta_ij)) - s_ij * theta_ij
    """
    target, mask = _pair_targets(labels)
    theta = 0.5 * (z @ z.t())
    vals = theta[mask]
    return _pairwise_logistic_loss_from_logits(vals, target)




def _dsh_pairwise_loss(z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    DSH-style pairwise supervised hashing loss.
    Same-class pairs are pulled close; different-class pairs are pushed apart by a margin.
    """
    labels = labels.view(-1)
    z = z.float()
    n = z.shape[0]
    if n <= 1:
        return z.new_tensor(0.0)

    sim = labels.unsqueeze(0).eq(labels.unsqueeze(1)).float()
    dist_sq = torch.cdist(z, z, p=2).pow(2)

    margin = float(2 * z.shape[1])
    pos = sim * dist_sq
    neg = (1.0 - sim) * torch.clamp(margin - dist_sq, min=0.0)

    mask = 1.0 - torch.eye(n, device=z.device, dtype=z.dtype)
    denom = mask.sum().clamp_min(1.0)

    return 0.5 * ((pos + neg) * mask).sum() / denom


def _dsh_quantization_loss(z: torch.Tensor) -> torch.Tensor:
    """
    DSH binary regularization term.
    Encourage continuous outputs to approach +/-1 before sign binarization.
    """
    return torch.mean(torch.abs(torch.abs(z) - 1.0))


def _dhn_pairwise_loss(z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    u = torch.tanh(z)
    target, mask = _pair_targets(labels)
    theta = 0.5 * (u @ u.t())
    vals = theta[mask]
    return _pairwise_logistic_loss_from_logits(vals, target)


def _dhn_quantization_loss(z: torch.Tensor) -> torch.Tensor:
    u = torch.tanh(z)
    b = _hard_sign(u).detach()
    return ((u - b) ** 2).mean()


def _make_random_target_codes(num_classes: int, hash_bits: int, device: torch.device, seed: int = 2020) -> torch.Tensor:
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    codes = torch.randint(0, 2, (num_classes, hash_bits), generator=g).float()
    codes = codes * 2.0 - 1.0
    return codes.to(device=device)


def _dpn_polarization_loss(z: torch.Tensor, labels: torch.Tensor, margin: float = 1.0, seed: int = 2020) -> torch.Tensor:
    num_classes = int(labels.max().item()) + 1
    t = _make_random_target_codes(num_classes, z.shape[1], z.device, seed=seed)[labels.long()]
    return F.relu(margin - z * t).mean()


def _hadamard(n: int, device: torch.device) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h.to(device=device)


def _make_orthogonal_targets(num_classes: int, hash_bits: int, device: torch.device, seed: int = 2021) -> torch.Tensor:
    n = 1
    while n < max(num_classes, hash_bits):
        n *= 2
    if hash_bits >= num_classes:
        H = _hadamard(n, device=device)
        return H[:num_classes, :hash_bits].contiguous()
    # C > K 时无法构造 C 个严格互相正交的 K-bit 目标码；先用固定随机二值目标兜底。
    return _make_random_target_codes(num_classes, hash_bits, device, seed=seed)


def _orthohash_loss(z: torch.Tensor, labels: torch.Tensor, scale: float = 8.0) -> torch.Tensor:
    num_classes = int(labels.max().item()) + 1
    O = _make_orthogonal_targets(num_classes, z.shape[1], z.device)
    logits = scale * (F.normalize(z, p=2, dim=1) @ F.normalize(O, p=2, dim=1).t())
    return F.cross_entropy(logits, labels.long())




_SHC_CENTER_CACHE = {}

def _shc_label_index(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim > 1:
        return labels.argmax(dim=1).long()
    return labels.long().view(-1)

def _load_shc_centers(device: torch.device) -> torch.Tensor:
    import os
    path = os.environ.get("DAHCR_SHC_CENTER_PATH", "")
    if not path:
        raise RuntimeError("baseline=shc requires DAHCR_SHC_CENTER_PATH")
    key = (path, str(device))
    if key in _SHC_CENTER_CACHE:
        return _SHC_CENTER_CACHE[key]
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    centers = payload["centers"] if isinstance(payload, dict) and "centers" in payload else payload
    centers = centers.float().to(device)
    if centers.ndim != 2:
        raise RuntimeError(f"SHC centers must be 2D, got {centers.shape}")
    _SHC_CENTER_CACHE[key] = centers
    return centers

def _shc_style_loss(z: torch.Tensor, labels: torch.Tensor, gamma: float = 0.05):
    h = torch.tanh(z)
    centers_all = _load_shc_centers(h.device)
    labels_idx = _shc_label_index(labels).to(h.device)

    if labels_idx.max().item() >= centers_all.shape[0]:
        raise RuntimeError(f"label max {labels_idx.max().item()} >= center classes {centers_all.shape[0]}")
    centers = centers_all[labels_idx]

    if centers.shape[1] != h.shape[1]:
        raise RuntimeError(f"center bits {centers.shape[1]} != hash bits {h.shape[1]}")

    h_norm = torch.nn.functional.normalize(h, dim=1)
    c_norm = torch.nn.functional.normalize(centers, dim=1)

    center_loss = (1.0 - (h_norm * c_norm).sum(dim=1)).mean()
    quant_loss = ((h.abs() - 1.0) ** 2).mean()
    total_loss = center_loss + float(gamma) * quant_loss
    return total_loss, center_loss, quant_loss




def _nhl_quant_loss(z: torch.Tensor) -> torch.Tensor:
    h = torch.tanh(z)
    return ((h.abs() - 1.0) ** 2).mean()

def _nhl_nested_pair_loss(z: torch.Tensor, labels: torch.Tensor, beta: float = 1.0):
    """
    NHL-style nested hash supervision.
    Full 64-bit output and nested prefixes are supervised together.
    This keeps the same ResNet18 backbone and hash head but changes the hash supervision structure.
    """
    bit = z.shape[1]
    prefix_specs = []
    if bit >= 64:
        prefix_specs = [(16, 0.25), (32, 0.50), (bit, 1.00)]
    elif bit >= 32:
        prefix_specs = [(16, 0.50), (bit, 1.00)]
    else:
        prefix_specs = [(bit, 1.00)]

    pair_terms = []
    quant_terms = []
    weights = []

    for k, w in prefix_specs:
        zk = z[:, :k]
        pair_terms.append(float(w) * _weighted_hashnet_pairwise_loss(zk, labels, beta=beta))
        quant_terms.append(float(w) * _nhl_quant_loss(zk))
        weights.append(float(w))

    denom = sum(weights)
    pair_loss = sum(pair_terms) / denom
    quant_loss = sum(quant_terms) / denom
    return pair_loss, quant_loss




def _focalpair_quant_loss(z: torch.Tensor) -> torch.Tensor:
    h = torch.tanh(z)
    return ((h.abs() - 1.0) ** 2).mean()

def _focalpair_hard_pair_loss(z: torch.Tensor, labels: torch.Tensor, beta: float = 1.0, focal_gamma: float = 2.0) -> torch.Tensor:
    """
    FocalPairHash-style hard-pair reweighted supervised hashing objective.
    Positive/negative pairs are supervised by pairwise logistic loss, and difficult pairs receive larger focal weights.
    """
    labels = labels.view(-1, 1)
    sim = (labels == labels.t()).float()

    logits = 0.5 * float(beta) * (z @ z.t())
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, sim, reduction="none")

    prob = torch.sigmoid(logits)
    pt = torch.where(sim > 0.5, prob, 1.0 - prob)
    focal_weight = (1.0 - pt).clamp_min(1e-6).pow(float(focal_gamma))

    eye = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    mask = ~eye

    loss = (focal_weight * bce)[mask].mean()
    return loss




_SAAT_CENTER_CACHE = {}

def _saat_label_index(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim > 1:
        return labels.argmax(dim=1).long()
    return labels.long().view(-1)

def _saat_fallback_centers(num_classes: int, hash_bits: int, device) -> torch.Tensor:
    c = torch.arange(num_classes, device=device, dtype=torch.float32).view(-1, 1)
    b = torch.arange(hash_bits, device=device, dtype=torch.float32).view(1, -1)
    vals = torch.sin((c + 1.0) * 12.9898 + (b + 1.0) * 78.233)
    centers = torch.where(vals >= 0, torch.ones_like(vals), -torch.ones_like(vals))
    return centers

def _saat_load_centers(labels_idx: torch.Tensor, hash_bits: int) -> torch.Tensor:
    import os
    device = labels_idx.device
    path = os.environ.get("SAAT_CENTER_PATH", "")
    key = (path, str(device), int(hash_bits))
    if path:
        if key not in _SAAT_CENTER_CACHE:
            obj = torch.load(path, map_location=device)
            if isinstance(obj, dict) and "centers" in obj:
                centers = obj["centers"].float().to(device)
            else:
                centers = obj.float().to(device)
            _SAAT_CENTER_CACHE[key] = centers
        centers = _SAAT_CENTER_CACHE[key]
        if int(labels_idx.max().item()) < centers.shape[0] and centers.shape[1] == hash_bits:
            return centers
    num_classes = int(labels_idx.max().item()) + 1
    return _saat_fallback_centers(num_classes, hash_bits, device)

def _saat_pairwise_loss(z: torch.Tensor, labels: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    labels_idx = _saat_label_index(labels).view(-1, 1)
    sim = (labels_idx == labels_idx.t()).float()
    logits = 0.5 * float(beta) * (z @ z.t())
    loss_mat = torch.nn.functional.binary_cross_entropy_with_logits(logits, sim, reduction="none")
    eye = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    return loss_mat[~eye].mean()

def _saat_quant_loss(z: torch.Tensor) -> torch.Tensor:
    h = torch.tanh(z)
    return ((h.abs() - 1.0) ** 2).mean()

def _saat_mainstay_loss(z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels_idx = _saat_label_index(labels)
    h = torch.tanh(z)
    centers = _saat_load_centers(labels_idx, h.shape[1])
    target = centers[labels_idx]
    # semantic mainstay guidance: align continuous hash output to train-split semantic mainstay code
    return torch.nn.functional.mse_loss(h, target)

def _saat_style_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 1.0,
    gamma: float = 0.1,
    lambda_mainstay: float = 1.0,
):
    """
    SAAT-style adapted objective:
      supervised pairwise hashing + train-split semantic mainstay code guidance + quantization.
    This keeps the semantic-mainstay guidance idea under the unified degradation protocol.
    It is an adapted baseline, not an official reproduction.
    """
    pair_loss = _saat_pairwise_loss(z, labels, beta=beta)
    quant_loss = _saat_quant_loss(z)
    mainstay_loss = _saat_mainstay_loss(z, labels)
    total_loss = pair_loss + float(lambda_mainstay) * mainstay_loss + float(gamma) * quant_loss
    return total_loss, pair_loss, quant_loss, mainstay_loss


def view_baseline_loss(out: Dict[str, torch.Tensor], labels: torch.Tensor, baseline: str, gamma: float = 0.1, beta: float = 1.0, pairwise_weight: float = 1.0) -> Dict[str, torch.Tensor]:
    z = out['z']
    logits = out['logits']
    zero = z.new_tensor(0.0)
    ce_loss = quant_loss = pair_loss = polar_loss = ortho_loss = zero

    if baseline == "shc" or baseline == 'shc':
        z = out["z"] if isinstance(out, dict) and "z" in out else out
        total_loss, center_loss, quant_loss = _shc_style_loss(z, labels, gamma=gamma)
        zero = z.new_tensor(0.0)
        return {
            "total_loss": total_loss,
            "base_loss": total_loss,
            "ce_loss": zero,
            "pair_loss": center_loss,
            "quant_loss": quant_loss,
            "polar_loss": zero,
            "ortho_loss": zero,
        }

    if baseline == "focalpair" or baseline == 'focalpair':
        z = out["z"] if isinstance(out, dict) and "z" in out else out
        pair_loss = _focalpair_hard_pair_loss(z, labels, beta=beta, focal_gamma=2.0)
        quant_loss = _focalpair_quant_loss(z)
        total_loss = float(pairwise_weight) * pair_loss + float(gamma) * quant_loss
        zero = z.new_tensor(0.0)
        return {
            "total_loss": total_loss,
            "base_loss": total_loss,
            "ce_loss": zero,
            "pair_loss": pair_loss,
            "quant_loss": quant_loss,
            "polar_loss": zero,
            "ortho_loss": zero,
        }

    if baseline == "saat" or baseline == 'saat':
        z = out["z"] if isinstance(out, dict) and "z" in out else out
        total_loss, pair_loss, quant_loss, mainstay_loss = _saat_style_loss(
            z, labels, beta=beta, gamma=gamma, lambda_mainstay=1.0
        )
        zero = z.new_tensor(0.0)
        return {
            "total_loss": total_loss,
            "base_loss": total_loss,
            "ce_loss": zero,
            "pair_loss": pair_loss,
            "quant_loss": quant_loss,
            "polar_loss": mainstay_loss,
            "ortho_loss": zero,
        }

    if baseline == "nhl" or baseline == 'nhl':
        z = out["z"] if isinstance(out, dict) and "z" in out else out
        pair_loss, quant_loss = _nhl_nested_pair_loss(z, labels, beta=beta)
        total_loss = float(pairwise_weight) * pair_loss + float(gamma) * quant_loss
        zero = z.new_tensor(0.0)
        return {
            "total_loss": total_loss,
            "base_loss": total_loss,
            "ce_loss": zero,
            "pair_loss": pair_loss,
            "quant_loss": quant_loss,
            "polar_loss": zero,
            "ortho_loss": zero,
        }

    if baseline == 'greedyhash':
        ce_loss = F.cross_entropy(logits, labels)
        quant_loss = quantization_loss(z)
        total = ce_loss + gamma * quant_loss
        base = ce_loss
    elif baseline == 'dpn':
        polar_loss = _dpn_polarization_loss(z, labels, margin=1.0)
        total = base = polar_loss
    elif baseline == 'hashnet':
        pair_loss = _weighted_hashnet_pairwise_loss(z, labels, beta=beta)
        total = base = pairwise_weight * pair_loss
    elif baseline == 'dpsh':
        pair_loss = _dpsh_pairwise_loss(z, labels)
        quant_loss = quantization_loss(z)
        total = pairwise_weight * pair_loss + gamma * quant_loss
        base = pairwise_weight * pair_loss
    elif baseline == "dsh":
        pair_loss = _dsh_pairwise_loss(z, labels)
        quant_loss = _dsh_quantization_loss(z)
        total = pairwise_weight * pair_loss + gamma * quant_loss
        base = pairwise_weight * pair_loss
    elif baseline == 'dhn':
        pair_loss = _dhn_pairwise_loss(z, labels)
        quant_loss = _dhn_quantization_loss(z)
        total = pairwise_weight * pair_loss + gamma * quant_loss
        base = pairwise_weight * pair_loss
    else:
        raise ValueError(f'Unsupported baseline: {baseline}')

    return {'total_loss': total, 'base_loss': base, 'ce_loss': ce_loss, 'pair_loss': pair_loss, 'quant_loss': quant_loss, 'polar_loss': polar_loss, 'ortho_loss': ortho_loss}


def da_supcon_loss(clean_z: torch.Tensor, paired_z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1, same_image_weight: float = 2.0, same_class_weight: float = 1.0) -> torch.Tensor:
    features = torch.cat([clean_z, paired_z], dim=0)
    features = F.normalize(features, dim=1)
    full_labels = torch.cat([labels, labels], dim=0)
    n = labels.shape[0]
    total_n = full_labels.shape[0]
    sim = features @ features.t() / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(total_n, dtype=torch.bool, device=features.device)
    label_eq = full_labels.view(-1, 1) == full_labels.view(1, -1)
    pos_mask = label_eq & (~self_mask)
    weights = torch.zeros_like(sim)
    weights[pos_mask] = same_class_weight
    idx = torch.arange(n, device=features.device)
    weights[idx, idx + n] = same_image_weight
    weights[idx + n, idx] = same_image_weight
    denom_mask = ~self_mask
    exp_sim = torch.exp(sim) * denom_mask.float()
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
    pos_weight_sum = weights.sum(dim=1)
    valid = pos_weight_sum > 0
    if valid.sum() == 0:
        return clean_z.new_tensor(0.0)
    loss_per_anchor = -(weights * log_prob).sum(dim=1) / (pos_weight_sum + 1e-12)
    return loss_per_anchor[valid].mean()


def compute_total_loss_v3_dasupcon(outputs: Dict[str, Dict[str, torch.Tensor]], labels: torch.Tensor, baseline: str, method: str, alpha: float = 0.1, temperature: float = 0.1, same_image_weight: float = 2.0, same_class_weight: float = 1.0, gamma: float = 0.1, beta: float = 1.0, pairwise_weight: float = 1.0) -> Dict[str, torch.Tensor]:
    clean = view_baseline_loss(outputs['clean'], labels, baseline, gamma=gamma, beta=beta, pairwise_weight=pairwise_weight)
    paired = view_baseline_loss(outputs['paired'], labels, baseline, gamma=gamma, beta=beta, pairwise_weight=pairwise_weight)
    total_loss = 0.5 * (clean['total_loss'] + paired['total_loss'])
    base_loss = 0.5 * (clean['base_loss'] + paired['base_loss'])
    ce_loss = 0.5 * (clean['ce_loss'] + paired['ce_loss'])
    pair_loss = 0.5 * (clean['pair_loss'] + paired['pair_loss'])
    quant_loss = 0.5 * (clean['quant_loss'] + paired['quant_loss'])
    polar_loss = 0.5 * (clean['polar_loss'] + paired['polar_loss'])
    ortho_loss = 0.5 * (clean['ortho_loss'] + paired['ortho_loss'])
    if method == 'base':
        supcon = outputs['clean']['z'].new_tensor(0.0)
        return {'total_loss': total_loss, 'base_loss': base_loss, 'ce_loss': ce_loss, 'pair_loss': pair_loss, 'quant_loss': quant_loss, 'polar_loss': polar_loss, 'ortho_loss': ortho_loss, 'dasupcon_loss': supcon, 'effective_dasupcon': supcon, 'effective_ratio': supcon}
    if method != 'dasupcon':
        raise ValueError(f'Unsupported method: {method}')
    supcon = da_supcon_loss(outputs['clean']['z'], outputs['paired']['z'], labels, temperature, same_image_weight, same_class_weight)
    effective = total_loss.new_tensor(float(alpha)) * supcon
    final_total = total_loss + effective
    ratio = effective.detach() / (base_loss.detach().abs() + 1e-12)
    return {'total_loss': final_total, 'base_loss': base_loss, 'ce_loss': ce_loss, 'pair_loss': pair_loss, 'quant_loss': quant_loss, 'polar_loss': polar_loss, 'ortho_loss': ortho_loss, 'dasupcon_loss': supcon, 'effective_dasupcon': effective, 'effective_ratio': ratio}
