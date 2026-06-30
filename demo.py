"""Minimal FROST inference demo.

Given one reference image + its binary mask and a query image, segment the same
class in the query and save the predicted mask as a PNG.

Example:
    python demo.py \
        --weights /path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
        --ref-image ref.jpg --ref-mask ref_mask.png \
        --query-image query.jpg --out pred.png

DINOv3 ViT-L/16 LVD-1689M weights are from the official DINOv3 repo
(facebookresearch/dinov3); the loader expects the '-<8charhash>.pth' filename.
"""
import argparse

import numpy as np
from PIL import Image

from frost import build_frost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True,
                    help="path to dinov3_vitl16_pretrain_lvd1689m-<hash>.pth")
    ap.add_argument("--ref-image", required=True, help="support image path")
    ap.add_argument("--ref-mask", required=True,
                    help="binary support mask path (nonzero = foreground)")
    ap.add_argument("--query-image", required=True, help="query image path")
    ap.add_argument("--out", default="pred.png", help="output mask PNG path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image-size", type=int, default=1024)
    args = ap.parse_args()

    # Build the frozen-backbone FROST model (training-free).
    model = build_frost(args.weights, image_size=args.image_size, device=args.device)

    # Load inputs as PIL.
    ref_pil = Image.open(args.ref_image).convert("RGB")
    ref_mask = Image.open(args.ref_mask)              # any mode; nonzero = FG
    query_pil = Image.open(args.query_image).convert("RGB")

    # One-shot segmentation. For k-shot, call model.set_reference(...) k times,
    # then model.set_target(...) and model.segment().
    mask = model.segment_one(ref_pil, ref_mask, query_pil)  # (H, W) bool tensor

    out = (mask.detach().cpu().numpy().astype(np.uint8) * 255)
    Image.fromarray(out, mode="L").save(args.out)
    print(f"saved predicted mask -> {args.out}  (foreground pixels: {int(mask.sum())})")


if __name__ == "__main__":
    main()
