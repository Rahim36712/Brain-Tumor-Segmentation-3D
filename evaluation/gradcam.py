"""
Brain Tumor Segmentation — Grad-CAM Explainability
===================================================

Gradient-weighted Class Activation Mapping (Grad-CAM) adapted for
3D convolutional neural networks.

Grad-CAM produces a heatmap highlighting which spatial regions of
the input volume most influenced the model's prediction for a given
class.  This is valuable in medical imaging for:

1. **Clinical trust** — showing *where* the model is "looking"
   builds confidence among radiologists.
2. **Model debugging** — if the heatmap highlights irrelevant areas
   (e.g. skull instead of tumor), the model may be learning
   spurious correlations.
3. **Failure analysis** — understanding why the model missed a
   tumor region.

Method:
    1. Forward pass — capture activations at a target conv layer.
    2. Backward pass — compute gradients of the target class score
       w.r.t. those activations.
    3. Weight each activation channel by its mean gradient (global
       average pooling of gradients).
    4. ReLU the weighted combination → positive-only heatmap.
    5. Upsample to input spatial dimensions.

Reference:
    Selvaraju et al., *Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization*, ICCV 2017.

Usage::

    from evaluation.gradcam import GradCAM3D

    gcam = GradCAM3D(model, target_layer="bottleneck")
    heatmap = gcam(input_tensor, target_class=3)  # ET class
    # heatmap shape: (D, H, W), values in [0, 1]
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM3D:
    """
    Grad-CAM for 3D segmentation models.

    Parameters
    ----------
    model : nn.Module
        Trained 3D segmentation model.
    target_layer_name : str
        Name of the target convolutional layer to hook.
        Common choices:
        - ``"bottleneck"`` — deepest features (coarsest, most semantic)
        - ``"enc4"`` — fourth encoder block
        - ``"dec1"`` — first decoder block (finest spatial detail)
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_name: str = "bottleneck",
    ) -> None:
        self.model = model
        self.model.eval()

        self.target_layer_name = target_layer_name
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register hooks on the target layer
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Attach forward and backward hooks to the target layer."""
        target = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target = module
                break

        if target is None:
            available = [n for n, _ in self.model.named_modules() if n]
            raise ValueError(
                f"Layer '{self.target_layer_name}' not found. "
                f"Available layers: {available[:20]}"
            )

        def forward_hook(module, input, output):
            # For DownBlock, output is (pooled, features) — we want features
            if isinstance(output, tuple):
                self.activations = output[1].detach()
            else:
                self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self.gradients = grad_output[0].detach()
            else:
                self.gradients = grad_output.detach()

        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)

    @torch.enable_grad()
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 3,
        device: str = "cuda",
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap.

        Parameters
        ----------
        input_tensor : Tensor
            Input volume, shape ``(1, C, D, H, W)``.
        target_class : int
            Class index to explain (e.g. 3 for Enhancing Tumor).
        device : str
            Device for computation.

        Returns
        -------
        ndarray
            Heatmap of shape ``(D, H, W)`` with values in [0, 1].
        """
        self.model.to(device)
        input_tensor = input_tensor.to(device).requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, list):
            output = output[0]  # full-resolution output

        # Score for the target class — mean over spatial dims
        score = output[0, target_class].mean()

        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Hooks did not capture activations/gradients. "
                "Check target_layer_name."
            )

        # Global average pooling of gradients → channel weights
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)  # (1, Ch, 1, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, d, h, w)

        # ReLU — keep only positive contributions
        cam = F.relu(cam)

        # Upsample to input spatial size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def generate_overlay(
        self,
        mri_slice: np.ndarray,
        cam_slice: np.ndarray,
        alpha: float = 0.4,
        colormap: str = "jet",
    ) -> np.ndarray:
        """
        Create a Grad-CAM overlay on an MRI slice.

        Parameters
        ----------
        mri_slice : ndarray
            2D MRI slice (H, W), normalized to [0, 1].
        cam_slice : ndarray
            2D Grad-CAM heatmap (H, W), values in [0, 1].
        alpha : float
            Heatmap opacity.
        colormap : str
            Matplotlib colormap name.

        Returns
        -------
        ndarray
            RGB overlay image (H, W, 3), values in [0, 1].
        """
        import matplotlib.pyplot as plt

        cmap = plt.cm.get_cmap(colormap)
        heatmap_colored = cmap(cam_slice)[:, :, :3]  # drop alpha

        # Normalize MRI to [0, 1]
        mri_norm = mri_slice.astype(np.float32)
        if mri_norm.max() > 0:
            mri_norm = mri_norm / mri_norm.max()

        # Stack grayscale to RGB
        mri_rgb = np.stack([mri_norm] * 3, axis=-1)

        # Blend
        overlay = (1 - alpha) * mri_rgb + alpha * heatmap_colored.astype(np.float32)
        return np.clip(overlay, 0, 1)
