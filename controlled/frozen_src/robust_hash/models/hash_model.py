from typing import Dict

import torch
import torch.nn as nn

from robust_hash.models.backbone import build_backbone


class RobustHashModel(nn.Module):
    """
    通用鲁棒哈希模型骨架：
    backbone -> feature -> hash projection -> classifier
    """

    def __init__(
        self,
        num_classes: int,
        hash_bits: int = 64,
        backbone_name: str = "resnet18",
        pretrained: bool = False,
    ) -> None:
        super().__init__()

        self.backbone, feat_dim = build_backbone(
            name=backbone_name,
            pretrained=pretrained,
        )

        self.hash_proj = nn.Linear(feat_dim, hash_bits)
        self.classifier = nn.Linear(hash_bits, num_classes)

        self.num_classes = num_classes
        self.hash_bits = hash_bits
        self.backbone_name = backbone_name

    @staticmethod
    def ste_sign(x: torch.Tensor) -> torch.Tensor:
        """
        Straight-through estimator for sign.
        forward: sign(x)
        backward: identity gradient
        """
        signed = torch.sign(x)
        signed[signed == 0] = 1.0
        return x + (signed - x).detach()

    def encode_continuous(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        z = self.hash_proj(feat)
        b = self.ste_sign(z)
        logits = self.classifier(b)

        return {
            "feat": feat,
            "z": z,
            "b": b,
            "logits": logits,
        }

    def forward_single(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.encode_continuous(x)

    def forward_pair(
        self,
        clean_x: torch.Tensor,
        paired_x: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        clean_out = self.forward_single(clean_x)
        paired_out = self.forward_single(paired_x)

        return {
            "clean": clean_out,
            "paired": paired_out,
        }