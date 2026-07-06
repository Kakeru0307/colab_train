"""segmentation-models-pytorch ベースの U-Net。"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn

DEFAULT_ENCODER = "resnet18"


def build_unet(
    in_channels: int = 11,
    out_channels: int = 11,
    encoder: str = DEFAULT_ENCODER,
    encoder_weights: str | None = None,
) -> nn.Module:
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
        activation=None,
    )
