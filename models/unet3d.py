"""
Brain Tumor Segmentation — 3D U-Net Architecture
=================================================

A full 3D U-Net for volumetric medical image segmentation, optimised
for the BraTS brain tumor segmentation task.

Architecture Overview
---------------------
The 3D U-Net follows an **encoder–decoder** structure with **skip
connections**:

**Encoder (contracting path):**
    Each level applies two 3×3×3 convolutions (with InstanceNorm +
    LeakyReLU) followed by 2×2×2 max-pooling.  Feature maps double
    at each level:  32 → 64 → 128 → 256.

**Bottleneck:**
    Two 3×3×3 convolutions at the deepest level (256 → 512 channels).

**Decoder (expanding path):**
    Each level applies a 2×2×2 transposed convolution (halving
    channels), concatenates the corresponding encoder features via
    a skip connection, then applies two 3×3×3 convolutions.

**Output head:**
    A 1×1×1 convolution maps the final feature map to ``num_classes``
    channels (logits — no softmax, because the loss functions apply
    it internally).

**Skip connections:**
    Concatenation of encoder feature maps with the decoder at each
    level.  This preserves fine-grained spatial detail lost during
    downsampling, which is critical for precise boundary delineation
    in tumor segmentation.

**Feature map scaling** (default ``base_filters=32``)::

    Encoder:   32 → 64 → 128 → 256
    Bottleneck:           256 → 512
    Decoder:  256 → 128 →  64 →  32
    Output:    32 → num_classes

**Why InstanceNorm instead of BatchNorm?**
    With the small batch sizes typical of 3D medical imaging (often
    batch_size=1–2), BatchNorm statistics are unreliable.  InstanceNorm
    normalises each sample independently and performs better in this
    regime.

Usage::

    from models.unet3d import UNet3D

    model = UNet3D(in_channels=4, num_classes=4, base_filters=32)
    x = torch.randn(1, 4, 128, 128, 128)
    logits = model(x)   # → (1, 4, 128, 128, 128)
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


# ------------------------------------------------------------------ #
#  Building blocks
# ------------------------------------------------------------------ #
class ConvBlock3D(nn.Module):
    """
    Two consecutive 3D convolution layers, each followed by
    InstanceNorm and LeakyReLU.

    ``Conv3d → InstanceNorm3d → LeakyReLU`` × 2

    Parameters
    ----------
    in_ch : int
        Input channels.
    out_ch : int
        Output channels.
    dropout : float
        Dropout probability applied after the second activation.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.block(x))


class DownBlock(nn.Module):
    """Encoder block: ConvBlock followed by 2×2×2 max-pool."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = ConvBlock3D(in_ch, out_ch, dropout=dropout)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pooled_output, skip_features)."""
        features = self.conv(x)
        pooled = self.pool(features)
        return pooled, features


class UpBlock(nn.Module):
    """
    Decoder block: transposed convolution upsample → concatenate skip
    → ConvBlock.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_ch, out_ch, kernel_size=2, stride=2, bias=False,
        )
        # after concat the input channels double
        self.conv = ConvBlock3D(out_ch * 2, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # handle potential size mismatches from odd spatial dims
        if x.shape != skip.shape:
            x = _pad_to_match(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ------------------------------------------------------------------ #
#  Utility: spatial padding to match shapes
# ------------------------------------------------------------------ #
def _pad_to_match(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pad ``x`` so its spatial dims match ``target``."""
    diffs = [target.shape[i] - x.shape[i] for i in range(2, 5)]
    # F.pad expects (W_left, W_right, H_left, H_right, D_left, D_right)
    pad = []
    for d in reversed(diffs):
        pad.extend([d // 2, d - d // 2])
    return nn.functional.pad(x, pad)


# ------------------------------------------------------------------ #
#  3D U-Net
# ------------------------------------------------------------------ #
class UNet3D(nn.Module):
    """
    Full 3D U-Net for volumetric segmentation.

    Parameters
    ----------
    in_channels : int
        Number of input channels (4 for BraTS: T1, T1ce, T2, FLAIR).
    num_classes : int
        Number of output segmentation classes.
    base_filters : int
        Number of feature maps in the first encoder level.
        Subsequent levels double: ``base → 2b → 4b → 8b``.
    dropout : float
        Dropout probability applied in conv blocks.
    deep_supervision : bool
        If ``True``, return intermediate decoder outputs for deep
        supervision loss (list of tensors at each decoder level).
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_filters: int = 32,
        dropout: float = 0.2,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision
        f = base_filters  # shorthand

        # ---- Encoder ---- #
        self.enc1 = DownBlock(in_channels, f, dropout=0.0)
        self.enc2 = DownBlock(f, f * 2, dropout=0.0)
        self.enc3 = DownBlock(f * 2, f * 4, dropout=dropout)
        self.enc4 = DownBlock(f * 4, f * 8, dropout=dropout)

        # ---- Bottleneck ---- #
        self.bottleneck = ConvBlock3D(f * 8, f * 16, dropout=dropout)

        # ---- Decoder ---- #
        self.dec4 = UpBlock(f * 16, f * 8, dropout=dropout)
        self.dec3 = UpBlock(f * 8, f * 4, dropout=dropout)
        self.dec2 = UpBlock(f * 4, f * 2, dropout=0.0)
        self.dec1 = UpBlock(f * 2, f, dropout=0.0)

        # ---- Output head ---- #
        self.output_conv = nn.Conv3d(f, num_classes, kernel_size=1)

        # ---- Deep supervision heads (optional) ---- #
        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv3d(f * 8, num_classes, kernel_size=1),
                nn.Conv3d(f * 4, num_classes, kernel_size=1),
                nn.Conv3d(f * 2, num_classes, kernel_size=1),
            ])

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming initialisation for all conv layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Input volume, shape ``(B, C, D, H, W)``.

        Returns
        -------
        Tensor or list[Tensor]
            - If ``deep_supervision=False``: logits ``(B, num_classes, D, H, W)``.
            - If ``deep_supervision=True``: list of logits at each
              decoder level (finest first).
        """
        # Encoder
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        d4 = self.dec4(x, skip4)
        d3 = self.dec3(d4, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)

        out = self.output_conv(d1)

        if self.deep_supervision and self.training:
            ds_outs = [
                nn.functional.interpolate(
                    self.ds_heads[0](d4), size=out.shape[2:], mode="trilinear", align_corners=False
                ),
                nn.functional.interpolate(
                    self.ds_heads[1](d3), size=out.shape[2:], mode="trilinear", align_corners=False
                ),
                nn.functional.interpolate(
                    self.ds_heads[2](d2), size=out.shape[2:], mode="trilinear", align_corners=False
                ),
            ]
            return [out] + ds_outs

        return out

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"{self.__class__.__name__}("
            f"params={params:,}, "
            f"deep_supervision={self.deep_supervision})"
        )
