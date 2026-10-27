from typing import Tuple

import torch
import torch.nn as nn


class SmallFallbackBackbone(nn.Module):
    """
    如果 torchvision 不可用，先用一个很小的 CNN 骨架做代码检查。
    正式训练时仍然优先使用 ResNet18。
    """

    def __init__(self, out_dim: int = 512) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return x


def build_backbone(name: str = "resnet18", pretrained: bool = False) -> Tuple[nn.Module, int]:
    """
    返回 backbone 和输出维度。
    """
    if name != "resnet18":
        raise ValueError(f"Unsupported backbone: {name}")

    try:
        from torchvision import models as tv_models

        if pretrained:
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        model = tv_models.resnet18(weights=weights)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feat_dim

    except Exception as e:
        raise RuntimeError(
            "The manuscript experiments require torchvision ResNet18. "
            "A fallback CNN is not permitted in this release."
        ) from e
