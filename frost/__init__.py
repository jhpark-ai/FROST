"""FROST: Training-Free Few-Shot Segmentation with Frozen Features and
Nonparametric Statistics (Junghwan Park)."""

from .model import FROST
from .encoder import DINOv3FeatureExtractor, build_encoder

__all__ = ["FROST", "DINOv3FeatureExtractor", "build_encoder", "build_frost"]


def build_frost(weights_path: str, image_size: int = 1024, device: str = "cuda",
                resize_to_orig_size: bool = True) -> FROST:
    """Build a ready-to-use FROST model from a DINOv3 ViT-L/16 checkpoint path.

    Args:
        weights_path: path to dinov3_vitl16_pretrain_lvd1689m-<hash>.pth.
        image_size: encoder input resolution (default 1024).
        device: torch device string.
        resize_to_orig_size: resize the output mask back to the query's
            original (H, W) if True, else keep it at image_size.
    """
    raw = build_encoder(weights_path, device=device)
    enc = DINOv3FeatureExtractor(raw, image_size=image_size, patch_size=16)
    return FROST(encoder=enc, raw_encoder=raw, image_size=image_size,
                 device=device, resize_to_orig_size=resize_to_orig_size).to(device).eval()
