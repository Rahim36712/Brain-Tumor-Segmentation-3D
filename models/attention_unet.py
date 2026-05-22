"""
Brain Tumor Segmentation — Attention U-Net 3D
==============================================

An enhanced 3D U-Net that adds **Attention Gates** on the skip
connections between encoder and decoder.

Why Attention U-Net?
--------------------
Standard U-Net skip connections pass ALL encoder features to the
decoder — including features from healthy tissue that may confuse
the segmentation.  Attention Gates learn to **suppress irrelevant
regions** and **focus on salient structures** (tumors).

Key improvements over vanilla U-Net:
1. **Selective feature emphasis** — attention weights highlight
   tumor-relevant spatial locations in the skip features.
2. **Better small-structure segmentation** — especially beneficial
   for the small Enhancing Tumor (ET) class in BraTS.
3. **Minimal overhead** — attention gates add < 5% extra parameters
   compared to the base U-Net.

Trade-offs:
- Slightly more memory usage per layer
- Marginal increase in training time (~5–10%)
- Requires careful learning-rate tuning for the gating parameters

Reference:
    Oktay et al., *Attention U-Net: Learning Where to Look for the
    Pancreas*, 2018.  arXiv:1804.03999

Architecture::

    Encoder features ──→ Attention Gate ──→ Weighted features ──┐
                            ↑                                    │
    Decoder features ───────┘                              Concatenate
                                                               ↓
                                                         ConvBlock

Usage::

    from models.attention_unet import AttentionUNet3D

    model = AttentionUNet3D(in_channels=4, num_classes=4)
    x = torch.randn(1, 4, 128, 128, 128)
    logits = model(x)   # → (1, 4, 128, 128, 128)
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from models.unet3d import ConvBlock3D, DownBlock, _pad_to_match


# ------------------------------------------------------------------ #
#  Attention Gate
# ------------------------------------------------------------------ #
class AttentionGate(nn.Module):
    """
    Additive attention gate for 3D feature maps.

    Computes spatial attention weights from the gating signal
    (decoder features) and the skip connection (encoder features).

    The gate output is ``skip * σ(ψ(ReLU(W_g·g + W_x·x + b)))``,
    where ``σ`` is sigmoid and ``ψ`` is a 1×1×1 conv.

    Parameters
    ----------
    gate_ch : int
        Channels in the gating signal (from decoder).
    skip_ch : int
        Channels in the skip connection (from encoder).
    inter_ch : int
        Intermediate channels inside the gate (typically skip_ch // 2).
    """

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int) -> None:
        super().__init__()

        self.W_gate = nn.Sequential(
            nn.Conv3d(gate_ch, inter_ch, kernel_size=1, bias=False),
            nn.InstanceNorm3d(inter_ch, affine=True),
        )
        self.W_skip = nn.Sequential(
            nn.Conv3d(skip_ch, inter_ch, kernel_size=1, bias=False),
            nn.InstanceNorm3d(inter_ch, affine=True),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, gate: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        gate : Tensor
            Gating signal from the decoder, shape ``(B, gate_ch, D', H', W')``.
        skip : Tensor
            Skip connection from the encoder, shape ``(B, skip_ch, D, H, W)``.

        Returns
        -------
        Tensor
            Attention-weighted skip features, same shape as ``skip``.
        """
        # Up-sample gate to match skip spatial size
        g = self.W_gate(gate)
        if g.shape[2:] != skip.shape[2:]:
            g = nn.functional.interpolate(
                g, size=skip.shape[2:], mode="trilinear", align_corners=False
            )

        x = self.W_skip(skip)
        attn = self.psi(self.relu(g + x))  # (B, 1, D, H, W)
        return skip * attn


# ------------------------------------------------------------------ #
#  Attention Up-Block
# ------------------------------------------------------------------ #
class AttentionUpBlock(nn.Module):
    """
    Decoder block with attention-gated skip connection.

    1. Transposed convolution to upsample decoder features
    2. Attention gate filters the encoder skip features
    3. Concatenate upsampled + attended skip
    4. ConvBlock
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_ch, out_ch, kernel_size=2, stride=2, bias=False,
        )
        self.attn_gate = AttentionGate(
            gate_ch=out_ch,
            skip_ch=out_ch,
            inter_ch=out_ch // 2,
        )
        self.conv = ConvBlock3D(out_ch * 2, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = _pad_to_match(x, skip)

        # Apply attention gate to the skip features
        skip_attended = self.attn_gate(gate=x, skip=skip)
        x = torch.cat([x, skip_attended], dim=1)
        return self.conv(x)


# ------------------------------------------------------------------ #
#  Attention U-Net 3D
# ------------------------------------------------------------------ #
class AttentionUNet3D(nn.Module):
    """
    3D U-Net with Attention Gates on all skip connections.

    Identical encoder and bottleneck to ``UNet3D``, but each decoder
    level uses ``AttentionUpBlock`` instead of ``UpBlock``.

    Parameters
    ----------
    in_channels : int
        Number of input channels (4 for BraTS).
    num_classes : int
        Number of output segmentation classes.
    base_filters : int
        Feature maps at the first encoder level.
    dropout : float
        Dropout probability for deeper conv blocks.
    deep_supervision : bool
        If ``True``, return multi-scale outputs for deep supervision.
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
        f = base_filters

        # ---- Encoder (shared design with UNet3D) ---- #
        self.enc1 = DownBlock(in_channels, f, dropout=0.0)
        self.enc2 = DownBlock(f, f * 2, dropout=0.0)
        self.enc3 = DownBlock(f * 2, f * 4, dropout=dropout)
        self.enc4 = DownBlock(f * 4, f * 8, dropout=dropout)

        # ---- Bottleneck ---- #
        self.bottleneck = ConvBlock3D(f * 8, f * 16, dropout=dropout)

        # ---- Decoder with attention gates ---- #
        self.dec4 = AttentionUpBlock(f * 16, f * 8, dropout=dropout)
        self.dec3 = AttentionUpBlock(f * 8, f * 4, dropout=dropout)
        self.dec2 = AttentionUpBlock(f * 4, f * 2, dropout=0.0)
        self.dec1 = AttentionUpBlock(f * 2, f, dropout=0.0)

        # ---- Output head ---- #
        self.output_conv = nn.Conv3d(f, num_classes, kernel_size=1)

        # ---- Deep supervision heads ---- #
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
            Input, shape ``(B, C, D, H, W)``.

        Returns
        -------
        Tensor or list[Tensor]
            Segmentation logits.
        """
        # Encoder
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with attention
        d4 = self.dec4(x, skip4)
        d3 = self.dec3(d4, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)

        out = self.output_conv(d1)

        if self.deep_supervision and self.training:
            ds_outs = [
                nn.functional.interpolate(
                    self.ds_heads[0](d4), size=out.shape[2:],
                    mode="trilinear", align_corners=False,
                ),
                nn.functional.interpolate(
                    self.ds_heads[1](d3), size=out.shape[2:],
                    mode="trilinear", align_corners=False,
                ),
                nn.functional.interpolate(
                    self.ds_heads[2](d2), size=out.shape[2:],
                    mode="trilinear", align_corners=False,
                ),
            ]
            return [out] + ds_outs

        return out

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"{self.__class__.__name__}("
            f"params={params:,}, "
            f"deep_supervision={self.deep_supervision})"
        )
