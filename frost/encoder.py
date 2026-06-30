"""DINOv3 ViT-L/16 feature encoder for FROST.

FROST uses a frozen, general-domain **DINOv3 ViT-L/16 (LVD-1689M)** backbone.
We take the last-layer patch tokens (1024-D, patch size 16) at a single input
resolution of 1024x1024, giving a 64x64 patch grid.

`DINOv3FeatureExtractor` is a thin wrapper around the torch.hub DINOv3 model
that exposes a single `__call__(imgs) -> (B, T, D, h, w)` so the FROST model can
treat feature extraction as one frozen op. The wrapper also resizes the feature
map to the canonical (image_size / patch_size) grid (a no-op at 1024 since
1024 / 16 == 64), which keeps the rest of the pipeline resolution-agnostic.
"""
from __future__ import annotations

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv3FeatureExtractor(nn.Module):
    """Single-scale, last-layer DINOv3 patch-token extractor.

    Args:
        encoder: a DINOv3 hub model exposing
            ``get_intermediate_layers(x, n=1, reshape=True) -> [(B, D, h, w)]``.
        image_size: input resolution fed to the encoder (default 1024).
        patch_size: encoder patch size (16 for ViT-L/16).
    """

    def __init__(self, encoder: nn.Module, image_size: int = 1024, patch_size: int = 16):
        super().__init__()
        self.encoder = encoder
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)

    @torch.no_grad()
    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B, T, C, H, W). Returns (B, T, D, h, w) with h = w = image_size/patch."""
        B, T = imgs.shape[:2]
        x = einops.rearrange(imgs, "b t c h w -> (b t) c h w")
        grid = self.image_size // self.patch_size
        enc_dtype = next(self.encoder.parameters()).dtype
        x = x.to(enc_dtype)
        f = self.encoder.get_intermediate_layers(x, n=1, reshape=True)[0].float()
        if f.shape[-1] != grid:
            f = F.interpolate(f, size=(grid, grid), mode="bilinear", align_corners=False)
        return einops.rearrange(f, "(b t) c h w -> b t c h w", b=B)


_HUB_NAME = "dinov3_vitl16"


def build_encoder(weights_path: str, device: str = "cuda") -> nn.Module:
    """Load the frozen DINOv3 ViT-L/16 backbone from the official hub repo.

    Args:
        weights_path: path to ``dinov3_vitl16_pretrain_lvd1689m-<hash>.pth``.
            The official hub loader expects a ``-<8charhash>.pth`` filename
            suffix; the LVD-1689M checkpoint ships as
            ``dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth``.
        device: device to place the encoder on.

    Returns:
        A frozen, eval-mode DINOv3 ViT-L/16 module.
    """
    enc = torch.hub.load(
        "facebookresearch/dinov3", _HUB_NAME,
        weights=weights_path, source="github", skip_validation=True,
    )
    enc = enc.to(device).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc
