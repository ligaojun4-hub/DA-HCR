"""
Hash losses for robust hashing project.

Methods:
  GreedyHash baseline (baseline=greedyhash):
    - method=base   : L = CE + gamma * Quant
    - method=supcon : L = CE + alpha * SupCon + gamma * Quant
  
  CSQ baseline (baseline=csq):
    - method=base   : L = CSQ_BCE + gamma * Quant
    - method=supcon : L = CSQ_BCE + alpha * SupCon + gamma * Quant
"""
from typing import Dict, Optional
import torch
import torch.nn.functional as F


def compute_base_loss(
    clean_logits: torch.Tensor,
    paired_logits: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Base CE loss: 0.5 * (CE(clean) + CE(paired))"""
    ce_clean = F.cross_entropy(clean_logits, labels)
    ce_paired = F.cross_entropy(paired_logits, labels)
    base_loss = 0.5 * (ce_clean + ce_paired)
    return {
        "ce_clean": ce_clean,
        "ce_paired": ce_paired,
        "base_loss": base_loss,
    }


def compute_supcon_loss(
    clean_z: torch.Tensor,
    paired_z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised Contrastive Loss"""
    features = torch.cat([clean_z, paired_z], dim=0)
    features = F.normalize(features, dim=1)
    full_labels = torch.cat([labels, labels], dim=0)
    batch_size = full_labels.shape[0]

    sim_matrix = torch.matmul(features, features.t()) / temperature
    sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
    sim_matrix = sim_matrix - sim_max.detach()

    label_eq = full_labels.unsqueeze(0) == full_labels.unsqueeze(1)
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    pos_mask = label_eq & (~self_mask)
    denom_mask = ~self_mask

    exp_sim = torch.exp(sim_matrix) * denom_mask.float()
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = pos_mask.float().sum(dim=1)
    valid = pos_count > 0
    if valid.sum() == 0:
        return clean_z.new_tensor(0.0)

    mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1)
    mean_log_prob_pos = mean_log_prob_pos[valid] / pos_count[valid]
    return -mean_log_prob_pos.mean()



def compute_consistency_loss(
    clean_z: torch.Tensor,
    paired_z: torch.Tensor,
) -> torch.Tensor:
    """Cross-degradation hash consistency loss.

    L_cons = mean(||tanh(z_clean) - tanh(z_paired)||^2)

    It aligns the soft hash representations of the clean and degraded
    views of the same image without imposing extra class-level or pairwise
    semantic constraints.
    """
    clean_h = torch.tanh(clean_z)
    paired_h = torch.tanh(paired_z)
    return ((clean_h - paired_h) ** 2).mean()

def compute_quantization_loss(
    clean_z: torch.Tensor,
    paired_z: torch.Tensor,
) -> torch.Tensor:
    """Quantization loss: ||z - sign(z)||^2"""
    b_clean = torch.sign(clean_z).detach()
    b_paired = torch.sign(paired_z).detach()
    quant_clean = ((clean_z - b_clean) ** 2).mean()
    quant_paired = ((paired_z - b_paired) ** 2).mean()
    return 0.5 * (quant_clean + quant_paired)


def compute_csq_loss(
    clean_z: torch.Tensor,
    paired_z: torch.Tensor,
    labels: torch.Tensor,
    hash_centers: torch.Tensor,
) -> torch.Tensor:
    """CSQ central similarity loss (BCE form).
    
    Args:
        clean_z, paired_z: [B, hash_bits] continuous codes
        labels: [B] class indices
        hash_centers: [num_classes, hash_bits] in {-1, +1}
    """
    target_centers = hash_centers[labels]
    target_binary = (target_centers + 1.0) / 2.0
    bce_clean = F.binary_cross_entropy_with_logits(clean_z, target_binary, reduction="mean")
    bce_paired = F.binary_cross_entropy_with_logits(paired_z, target_binary, reduction="mean")
    return 0.5 * (bce_clean + bce_paired)


def compute_total_loss(
    outputs: Dict[str, Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    method: str,
    alpha: float = 0.5,
    gamma: float = 0.1,
    supcon_temperature: float = 0.1,
    lambda_cons: float = 0.1,
    baseline: str = "greedyhash",
    hash_centers: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute total training loss based on baseline + method.
    
    baseline:
      - greedyhash : uses CE classifier loss as base
      - csq        : uses CSQ central similarity (BCE) loss as base
    
    method:
      - base   : baseline original = base_loss + gamma * Quant
      - supcon : baseline + SupCon = base_loss + alpha * SupCon + gamma * Quant
    
    For baseline=csq, hash_centers must be provided as [num_classes, hash_bits] tensor.
    """
    clean_logits = outputs["clean"]["logits"]
    paired_logits = outputs["paired"]["logits"]
    clean_z = outputs["clean"]["z"]
    paired_z = outputs["paired"]["z"]

    # Compute components
    base_dict = compute_base_loss(clean_logits, paired_logits, labels)
    quant_loss = compute_quantization_loss(clean_z, paired_z)
    if method == "supcon":
        supcon_loss = compute_supcon_loss(clean_z, paired_z, labels, temperature=supcon_temperature)
    else:
        supcon_loss = clean_z.new_tensor(0.0)
    cons_loss = compute_consistency_loss(clean_z, paired_z)

    # Choose base term according to baseline
    if baseline == "greedyhash":
        base_term = base_dict["base_loss"]
        csq_loss = clean_z.new_tensor(0.0)
    elif baseline == "csq":
        if hash_centers is None:
            raise ValueError("baseline=csq requires hash_centers tensor.")
        csq_loss = compute_csq_loss(clean_z, paired_z, labels, hash_centers)
        base_term = csq_loss
    else:
        raise ValueError(f"Unsupported baseline: {baseline}")

    # Combine according to method
    if method == "base":
        total_loss = base_term + gamma * quant_loss
    elif method == "supcon":
        total_loss = base_term + alpha * supcon_loss + gamma * quant_loss
    elif method == "cons":
        total_loss = base_term + gamma * quant_loss + lambda_cons * cons_loss
    else:
        raise ValueError(f"Unsupported method: {method}")

    return {
        "total_loss": total_loss,
        "base_loss": base_dict["base_loss"],
        "ce_clean": base_dict["ce_clean"],
        "ce_paired": base_dict["ce_paired"],
        "supcon_loss": supcon_loss,
        "cons_loss": cons_loss,
        "lambda_cons": clean_z.new_tensor(float(lambda_cons)),
        "effective_cons": clean_z.new_tensor(float(lambda_cons)) * cons_loss,
        "effective_ratio": (clean_z.new_tensor(float(lambda_cons)) * cons_loss) / (base_term.detach().abs() + 1e-12),
        "quant_loss": quant_loss,
        "csq_loss": csq_loss,
    }

