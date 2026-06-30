"""FROST -- training-free few-shot segmentation with frozen features.

Paper: "FROST: Training-Free Few-Shot Segmentation with Frozen Features and
Nonparametric Statistics" (Junghwan Park).

A single-path, training-free in-context segmentation model. Given a few support
(reference) image+mask pairs and a query image, FROST segments the same class in
the query without any fine-tuning or learned head. The only neural component is
a frozen DINOv3 ViT-L/16 backbone; everything else is closed-form statistics.

Forward path (paper stages):

  STAGE A -- frozen features
    A1. DINOv3 ViT-L/16 last-layer tokens at 1024x1024 (64x64 patch grid).
    A2. Support flip augmentation: encode each reference under {identity, horizontal-flip}
        (2 views), pool FG/BG anchors over both. RS imagery is hflip-symmetric.
    A3. L2-normalize -> SVD positional debiasing (project out the top-250
        directions of a noise-image token covariance).
    A4. Within-class shrinkage whitening: full covariance Sigma_W with
        Ledoit-Wolf-style shrinkage lambda=0.95 toward a scaled identity;
        f' = Sigma_W^{-1/2} f. No post-whitening L2 renorm (v3 default).

  STAGE B -- nonparametric posterior (see frost/density.py)
    B1. von Mises-Fisher (cosine) Parzen KDE log-posterior ratio over all
        support FG/BG anchors; bandwidth sigma by leave-one-out support margin.

  STAGE C -- spatial regularization + decision
    C1. bilateral label-propagation (5 product kernels, kappa=0) smooths the
        continuous posterior on the 64x64 grid.
    C2. candidate-mask intersection: forward (cosine to FG prototype > 0) AND
        backward (reciprocal top-k=3 nearest-neighbour vote), with
        support-adaptive dilation r = round(R_max * p_fg), R_max=4.
    C3. Upsample-then-threshold: bilinearly upsample the CONTINUOUS smoothed posterior to image
        resolution, threshold at tau=0 there (sub-patch zero-crossing boundary),
        and re-intersect the upsampled candidate.

  Op order: KDE -> shrinkage whitening ->  ->  threshold -> candidate ∩.
"""
from __future__ import annotations

import math
from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .data import (build_transform, denormalize, downsample_mask, load_image,
                   load_mask, upsample_mask)
from .density import density_ratio_predict

# --- baked v3-default hyper-parameters --------------------------------------
_SVD_COMPONENTS = 250            # SVD positional-debias subspace rank
_SHRINK_LAMBDA = 0.95            # full-covariance whitening shrinkage lambda
_BACKWARD_TOP_K = 3             # reciprocal-NN backward-vote top-k
_CAND_ADAPT_RMAX = 4           # support-adaptive candidate dilation R_max
_SIGMA_GRID = (0.05, 0.10, 0.20, 0.50, 1.00)
_THRESHOLD = 0.0           # Bayes-optimal tau
_BILATERAL_TAU_F = 0.20
_BILATERAL_TAU_I = 0.05
_BILATERAL_D_MAX = 16.0
_BILATERAL_TAU_M = 1.0
_BILATERAL_KAPPA = 0.0               # class-proto kernel off (kappa=0)
_BILATERAL_N_ITERS = 10
_BILATERAL_ALPHA = 0.70


class FROST(nn.Module):
    """Training-free few-shot segmentation with a frozen DINOv3 ViT-L/16 encoder.

    Args:
        encoder: feature extractor exposing ``__call__(imgs (B,T,C,H,W)) ->
            (B,T,D,h,w)``; use ``frost.encoder.DINOv3FeatureExtractor``.
        raw_encoder: the underlying hub backbone (for building the positional
            debias basis via a noise-image forward). If None, the positional
            basis is built lazily from ``encoder`` on first ``set_reference``.
        image_size: input resolution (default 1024).
        device: torch device string.
    """

    def __init__(
        self,
        encoder: nn.Module,
        raw_encoder: Optional[nn.Module] = None,
        image_size: int = 1024,
        device: str = "cuda",
        resize_to_orig_size: bool = True,
    ):
        super().__init__()
        self._extract_features = encoder
        self._raw_encoder = raw_encoder if raw_encoder is not None else getattr(
            encoder, "encoder", None)
        self.device = device
        self.image_size = image_size
        # If True, the returned mask is resized back to the query's original
        # (H, W); if False it stays at the model's input resolution (image_size).
        self.resize_to_orig_size = resize_to_orig_size
        self._transform = build_transform(image_size)
        self.positional_basis = self._build_positional_basis(device)
        self.reset_state()

    # ──────── state ────────
    def reset_state(self) -> None:
        self._ref_images = None
        self._ref_masks = None
        self._tgt_image = None
        self._orig_tgt_size = None
        self._orig_ref_size = None

    def set_reference(self, image, mask) -> None:
        """Add a reference (support) image + binary mask. Call repeatedly for k-shot."""
        img_tensor, self._orig_ref_size = load_image(image, self._transform, self.device)
        mask_tensor = load_mask(mask, self.image_size, self.device)
        if self._ref_images is None:
            self._ref_images = img_tensor
            self._ref_masks = mask_tensor
        else:
            self._ref_images = torch.cat([self._ref_images, img_tensor], dim=0)
            self._ref_masks = torch.cat([self._ref_masks, mask_tensor], dim=0)

    def set_target(self, image) -> None:
        """Set the query (target) image."""
        img_tensor, self._orig_tgt_size = load_image(image, self._transform, self.device)
        self._tgt_image = img_tensor

    @torch.no_grad()
    def segment(self) -> torch.Tensor:
        """Segment using the set reference(s) and target. Returns (H, W) bool mask."""
        if self._ref_images is None or self._ref_masks is None or self._tgt_image is None:
            raise RuntimeError("segment() requires reference image(s)+mask(s) and a target.")
        pred = self.predict_mask(self._ref_images, self._ref_masks, self._tgt_image)
        self.reset_state()
        return pred

    @torch.no_grad()
    def segment_one(self, ref_pil, ref_mask, query_pil) -> torch.Tensor:
        """Convenience 1-shot call. Returns (H, W) bool mask at the query's original size."""
        self.reset_state()
        self.set_reference(ref_pil, ref_mask)
        self.set_target(query_pil)
        return self.segment()

    # ──────── support flip augmentation ────────
    def _flip_augment(self, ref_images: torch.Tensor, ref_masks: torch.Tensor):
        """Identity + horizontal flip -> 2 views (STAGE A2)."""
        rl = [ref_images, torch.flip(ref_images, dims=(-1,))]
        ml = [ref_masks, torch.flip(ref_masks, dims=(-1,))]
        return torch.cat(rl, dim=0), torch.cat(ml, dim=0)

    # ──────── SVD positional debias (STAGE A3) ────────
    @torch.no_grad()
    def _build_positional_basis(self, device: str) -> torch.Tensor:
        """Top-r SVD directions of a noise-image token map = positional subspace."""
        from torchvision.transforms.functional import normalize
        enc = self._raw_encoder
        _dt = next(enc.parameters()).dtype
        noise_img = normalize(
            torch.zeros(1, 3, self.image_size, self.image_size),
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        ).to(device, dtype=_dt)
        noise_fmaps = enc.to(device).get_intermediate_layers(
            noise_img, n=1, reshape=True)[0].float()
        noise_fmaps = F.normalize(noise_fmaps, p=2, dim=1)
        E = einops.rearrange(noise_fmaps, "b c h w -> c (b h w)")
        E = E - E.mean(dim=1, keepdim=True)
        U, _, _ = torch.linalg.svd(E, full_matrices=False)
        return U[:, :_SVD_COMPONENTS].contiguous()

    def _debias_features(self, fmaps_norm: torch.Tensor) -> torch.Tensor:
        """Project features onto the orthogonal complement of the positional subspace."""
        B, T, C, H, W = fmaps_norm.shape
        X = fmaps_norm.reshape(B * T, C, H * W)
        basis = self.positional_basis.to(X.device).to(X.dtype)
        P_perp = torch.eye(C, device=X.device, dtype=X.dtype) - basis @ basis.T
        X_deb = torch.matmul(P_perp.unsqueeze(0), X).reshape(B, T, C, H, W)
        return F.normalize(X_deb, p=2, dim=2)

    # ──────── candidate localization (STAGE C2 forward+backward) ────────
    def _locate_candidates(self, sim_maps, ref_masks, feat_tgt_deb,
                           ref_prototype, h, w) -> torch.Tensor:
        # forward: positive cosine to the aggregated FG prototype
        sim_fwd = torch.einsum("bchw,cd->bhw", feat_tgt_deb, ref_prototype).squeeze(0)
        forward_mask = sim_fwd > 0
        if forward_mask.sum() == 0:
            forward_mask = sim_fwd > float(torch.quantile(sim_fwd, 0.9))
        # backward: reciprocal top-k NN vote per reference view
        k = len(sim_maps)
        bw_top_k = _BACKWARD_TOP_K
        votes = torch.zeros((h, w), dtype=torch.int32, device=sim_maps[0].device)
        for m, sim_m in enumerate(sim_maps):
            sim0 = sim_m[0]                       # (Hs, Ws, h, w)
            Hs, Ws = sim0.shape[:2]
            sim_t_to_r = sim0.permute(2, 3, 0, 1)  # (h, w, Hs, Ws)
            ref_mask_m = downsample_mask(ref_masks[m:m + 1], Hs, Ws).squeeze(0)
            flat = sim_t_to_r.reshape(h, w, -1)
            K_eff = min(bw_top_k, flat.shape[-1])
            _, top_idx = flat.topk(K_eff, dim=-1)
            rows = top_idx // Ws
            cols = top_idx % Ws
            fg_votes = ref_mask_m[rows, cols].sum(dim=-1)
            votes += (fg_votes >= (K_eff // 2 + 1)).to(torch.int32)
        backward_mask = votes >= math.ceil(k / 2)
        return forward_mask & backward_mask

    # ──────── full forward ────────
    @torch.no_grad()
    def predict_mask(self, ref_images: torch.Tensor, ref_masks: torch.Tensor,
                     tgt_image: torch.Tensor) -> torch.Tensor:
        """Segment the target given reference image(s)+mask(s). Returns (H, W) bool."""
        # A2: flip-augmentation view expansion
        ref_images, ref_masks = self._flip_augment(ref_images, ref_masks)
        S = ref_images.shape[0]
        imgs = torch.cat([ref_images, tgt_image], dim=0).unsqueeze(0)  # (1, S+1, C, H, W)

        # A1: frozen DINOv3 features
        fmaps = self._extract_features(imgs)
        fmaps_norm = F.normalize(fmaps, p=2, dim=2)
        _, _, C, h, w = fmaps_norm.shape
        ref_masks = ref_masks.unsqueeze(1)
        feat_tgt = fmaps_norm[:, S]               # raw-normalized target (for bilateral feature kernel)

        # A3: SVD positional debias
        fmaps_debiased = self._debias_features(fmaps_norm)
        feat_refs_deb = fmaps_debiased[:, :S]
        feat_tgt_deb = fmaps_debiased[:, S]

        # A4: within-class full-covariance shrinkage whitening (lambda=0.95)
        feat_refs_deb, feat_tgt_deb = self._shrinkage_whiten(
            feat_refs_deb, feat_tgt_deb, ref_masks, S, h, w)

        # FG/BG prototypes from whitened references (for the candidate + bilateral prototypes)
        ref_prototypes = []
        for s in range(S):
            mask_s = downsample_mask(ref_masks[s:s + 1], h, w)
            fg = feat_refs_deb[0, s, :, mask_s]
            if fg.shape[1] > 0:
                ref_prototypes.append(fg.mean(dim=1))
        if not ref_prototypes:
            return self._finalize_mask(
                torch.zeros(h, w, dtype=torch.bool, device=feat_tgt.device), tgt_image)
        p_fg_raw = torch.stack(ref_prototypes).mean(dim=0)
        ref_prototype = F.normalize(p_fg_raw, p=2, dim=0).unsqueeze(1)

        # C2 (part 1): candidate via forward + backward matching
        sim_maps = []
        for m in range(S):
            feat_ref_m = feat_refs_deb[:, m]
            sim_m = torch.einsum("bchw,bcxy->bhwxy", feat_ref_m, feat_tgt_deb)
            sim_maps.append(sim_m)
        candidate_mask = self._locate_candidates(
            sim_maps, ref_masks, feat_tgt_deb, ref_prototype, h, w)
        if candidate_mask.sum() == 0:
            return self._finalize_mask(candidate_mask, tgt_image)

        # STAGE B: build KDE FG/BG anchors from whitened references
        fg_collect, bg_collect = [], []
        for s_ in range(S):
            feat_s = feat_refs_deb[0, s_]          # (C, h, w)
            mask_s = downsample_mask(ref_masks[s_:s_ + 1], h, w)
            fg_s = feat_s[:, mask_s]
            bg_s = feat_s[:, ~mask_s]
            if fg_s.shape[1] > 0:
                fg_collect.append(fg_s.T)
            if bg_s.shape[1] > 0:
                bg_collect.append(bg_s.T)
        if not (fg_collect and bg_collect):
            return self._finalize_mask(candidate_mask, tgt_image)
        fg_anchors = F.normalize(torch.cat(fg_collect, dim=0).float(), p=2, dim=1)
        bg_anchors = F.normalize(torch.cat(bg_collect, dim=0).float(), p=2, dim=1)
        feat_t_white = F.normalize(
            feat_tgt_deb[0].reshape(C, -1).permute(1, 0).float(), p=2, dim=1)
        # graph features for the bilateral feature kernel use the raw-normalized target
        feat_t_bil = F.normalize(
            feat_tgt[0].reshape(C, -1).permute(1, 0).float(), p=2, dim=1)

        # per-patch RGB: avg-pool the denormalized target to the (h, w) grid
        tgt_rgb01 = denormalize(tgt_image[0]).clamp(0.0, 1.0).unsqueeze(0)  # (1,3,H,W)
        rgb_grid = F.adaptive_avg_pool2d(tgt_rgb01, (h, w))[0]              # (3,h,w)
        bil_img_rgb = rgb_grid.permute(1, 2, 0).reshape(-1, 3)             # (h*w, 3)

        # STAGE B+C1: KDE posterior + bilateral propagation
        pred_mask, info = density_ratio_predict(
            feat_tgt_deb_flat=feat_t_white,
            feat_ref_fg=fg_anchors,
            feat_ref_bg=bg_anchors,
            h=h, w=w,
            feat_tgt_bil_flat=feat_t_bil,
            bil_img_rgb=bil_img_rgb,
            sigma_grid=_SIGMA_GRID,
            bil_tau_f=_BILATERAL_TAU_F, bil_tau_I=_BILATERAL_TAU_I, bil_d_max=_BILATERAL_D_MAX,
            bil_tau_m=_BILATERAL_TAU_M, bil_kappa=_BILATERAL_KAPPA,
            bil_n_iters=_BILATERAL_N_ITERS, bil_alpha=_BILATERAL_ALPHA,
            threshold=_THRESHOLD,
        )

        # stash the continuous smoothed posterior for the upsample-then-threshold step (STAGE C3)
        self._post_ell = info.get("ell_smooth", None)
        if self._post_ell is not None:
            self._post_ell = self._post_ell.view(h, w)
        self._post_tau = float(info.get("tau", 0.0))
        self._post_cand = None

        # C2 (part 2): support-adaptive candidate dilation, then intersect
        if candidate_mask.sum() > 0:
            cand = candidate_mask
            p = float(ref_masks.float().mean().item())   # support FG fraction
            r = int(round(_CAND_ADAPT_RMAX * p))
            if r > 0:
                _cm = cand.float().view(1, 1, h, w)
                _cm = F.max_pool2d(_cm, kernel_size=2 * r + 1, stride=1, padding=r)
                cand = (_cm.view(h, w) > 0.5)
            self._post_cand = cand
            pred_mask = pred_mask & cand

        return self._finalize_mask(pred_mask, tgt_image)

    # ──────── A4: within-class full-covariance shrinkage whitening ────────
    @torch.no_grad()
    def _shrinkage_whiten(self, feat_refs_deb, feat_tgt_deb, ref_masks, S, h, w):
        """f' = Sigma_W^{-1/2} f with Ledoit-Wolf-style shrinkage lambda=0.95.

        Sigma_W = within-class pooled covariance of (debiased) FG/BG support
        patches; lambda blends it toward (mean diagonal)*I for full rank. No
        post-whitening L2 renorm (v3 default) -- the whitening scale is kept;
        downstream KDE anchors are re-normalized in density_ratio_predict regardless.
        """
        fg_list, bg_list = [], []
        for s_ in range(S):
            mask_s = downsample_mask(ref_masks[s_:s_ + 1], h, w)
            fg_s = feat_refs_deb[0, s_, :, mask_s]
            bg_s = feat_refs_deb[0, s_, :, ~mask_s]
            if fg_s.shape[1] > 0:
                fg_list.append(fg_s)
            if bg_s.shape[1] > 0:
                bg_list.append(bg_s)
        if not (fg_list and bg_list):
            return feat_refs_deb, feat_tgt_deb
        fg_all = torch.cat(fg_list, dim=1).float()   # (C, N_fg)
        bg_all = torch.cat(bg_list, dim=1).float()   # (C, N_bg)
        C = fg_all.shape[0]
        Xf = fg_all - fg_all.mean(dim=1, keepdim=True)
        Xb = bg_all - bg_all.mean(dim=1, keepdim=True)
        nrm = max(fg_all.shape[1] + bg_all.shape[1] - 2, 1)
        Sig = (Xf @ Xf.T + Xb @ Xb.T) / nrm          # (C, C)
        mu = Sig.diagonal().mean()
        lam = _SHRINK_LAMBDA
        Sig = (1 - lam) * Sig + lam * mu * torch.eye(C, device=Sig.device, dtype=Sig.dtype)
        evals, evecs = torch.linalg.eigh(Sig)
        inv_sqrt = evals.clamp(min=1e-6).rsqrt()
        W = (evecs * inv_sqrt.unsqueeze(0)) @ evecs.T  # Sigma^{-1/2} (symmetric)
        W = W.to(feat_refs_deb.dtype)
        fr = feat_refs_deb.permute(0, 1, 3, 4, 2) @ W  # (B,n,h,w,C)
        ft = feat_tgt_deb.permute(0, 2, 3, 1) @ W      # (B,h,w,C)
        # v3 default: NO post-whitening L2 renorm
        return fr.permute(0, 1, 4, 2, 3), ft.permute(0, 3, 1, 2)

    # ──────── C3: upsample-then-threshold finalize ────────
    @torch.no_grad()
    def _finalize_mask(self, mask: torch.Tensor, tgt_image: torch.Tensor) -> torch.Tensor:
        """Upsample the CONTINUOUS  posterior to image res, threshold there, ∩ candidate."""
        H, W = tgt_image.shape[-2:]
        ell = getattr(self, "_post_ell", None)
        if ell is not None:
            ell = ell.to(tgt_image.device).float()
            ell_up = F.interpolate(ell[None, None], size=(H, W),
                                   mode="bilinear", align_corners=False)[0, 0]
            up = ell_up > float(self._post_tau)
            cand = getattr(self, "_post_cand", None)
            if cand is not None:
                up = up & upsample_mask(cand, H, W)
            self._post_ell = None
            self._post_cand = None
            if self.resize_to_orig_size and self._orig_tgt_size is not None:
                up = upsample_mask(up, self._orig_tgt_size[0], self._orig_tgt_size[1])
            return up
        # fallback: bilinear-upsample the binary patch mask
        up = upsample_mask(mask, H, W)
        if self.resize_to_orig_size and self._orig_tgt_size is not None:
            up = upsample_mask(up, self._orig_tgt_size[0], self._orig_tgt_size[1])
        return up
