"""KDE posterior + bilateral propagation (paper stage B & C core).

The nonparametric heart of FROST. Given L2-normalized, debiased + whitened
target patches and FG/BG support anchors, compute a per-patch log-posterior
ratio via a von Mises-Fisher (cosine) Parzen kernel, pick the kernel bandwidth
sigma by a leave-one-out support margin, then smooth the continuous posterior
with a 5-kernel bilateral label-propagation on the 64x64 grid.

  1. KDE Bayes posterior (Parzen-Rosenblatt over cosine sims):
        l(t) = LSE_i((<t, f_i^FG> - 1)/sigma) - log N_FG
             - LSE_j((<t, f_j^BG> - 1)/sigma) + log N_BG
     Uses ALL support FG/BG patches as anchors (no k-means, no eigengap).
     l(t) > 0 is the Bayes-optimal boundary under equal priors.

  2. Bandwidth sigma via leave-one-out margin on support
     (grid {0.05,0.1,0.2,0.5,1.0}); pick the sigma that best separates the
     support FG/BG log-posterior ratios.

  3. bilateral propagation: product-of-kernels random-walk smoothing of the
     continuous posterior l. With kappa=0 (FROST default) the class-prototype
     kernel is identity, so smoothing is feature + RGB + spatial + score driven.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def _chunked_lse_sim(A: torch.Tensor, B: torch.Tensor, sigma: float,
                     log_count, self_diag: bool = False) -> torch.Tensor:
    """Memory-safe ``logsumexp((A @ B.T - 1)/sigma, dim=1) - log_count``.

    Splits A's ROWS so the (rows x anchors) block stays within a fixed element
    budget; logsumexp reduces over the full anchor dim per chunk, so the result
    is bit-identical to the un-chunked form. ``self_diag=True`` masks each row's
    own column (leave-one-out self-exclusion), needed for the LOO bandwidth and
    for high-shot anchor pools that would otherwise OOM at once.
    """
    Nt = A.shape[0]
    Nc = max(B.shape[0], 1)
    s = max(sigma, 1e-6)
    budget = 150_000_000  # ~600 MB fp32 per block
    chunk = max(64, min(Nt, int(budget // Nc)))
    if chunk >= Nt and not self_diag:
        sim = (A @ B.T - 1.0) / s
        return torch.logsumexp(sim, dim=1) - log_count
    parts = []
    for i in range(0, Nt, chunk):
        blk = (A[i:i + chunk] @ B.T - 1.0) / s
        if self_diag:
            r = blk.shape[0]
            idx = torch.arange(r, device=blk.device)
            blk[idx, i + idx] = float("-inf")
        parts.append(torch.logsumexp(blk, dim=1))
    return torch.cat(parts, 0) - log_count


@torch.no_grad()
def kde_log_posterior(
    feat_tgt: torch.Tensor,   # (N_t, D), L2-normalized
    feat_fg: torch.Tensor,    # (N_fg, D), L2-normalized
    feat_bg: torch.Tensor,    # (N_bg, D), L2-normalized
    sigma: float,
) -> torch.Tensor:
    """Per-patch log-posterior ratio l(t) = log p(t|FG) - log p(t|BG). Returns (N_t,)."""
    log_p_fg = _chunked_lse_sim(feat_tgt, feat_fg, sigma, math.log(max(feat_fg.shape[0], 1)))
    log_p_bg = _chunked_lse_sim(feat_tgt, feat_bg, sigma, math.log(max(feat_bg.shape[0], 1)))
    return log_p_fg - log_p_bg


@torch.no_grad()
def loo_margin_score(feat_fg: torch.Tensor, feat_bg: torch.Tensor, sigma: float) -> float:
    """Leave-one-out margin E[l(f_FG)] - E[l(f_BG)] on support. Higher = better sigma."""
    if feat_fg.shape[0] < 2 or feat_bg.shape[0] < 2:
        return 0.0
    # LLR on FG support (leave self out) -- should be > 0
    log_p_fg_loo = _chunked_lse_sim(feat_fg, feat_fg, sigma,
                                    math.log(feat_fg.shape[0] - 1), self_diag=True)
    log_p_bg_at_fg = _chunked_lse_sim(feat_fg, feat_bg, sigma, math.log(feat_bg.shape[0]))
    llr_fg = log_p_fg_loo - log_p_bg_at_fg
    # LLR on BG support (leave self out) -- should be < 0
    log_p_bg_loo = _chunked_lse_sim(feat_bg, feat_bg, sigma,
                                    math.log(feat_bg.shape[0] - 1), self_diag=True)
    log_p_fg_at_bg = _chunked_lse_sim(feat_bg, feat_fg, sigma, math.log(feat_fg.shape[0]))
    llr_bg = log_p_fg_at_bg - log_p_bg_loo
    return float(llr_fg.mean().item()) - float(llr_bg.mean().item())


@torch.no_grad()
def estimate_bandwidth(
    feat_fg: torch.Tensor,
    feat_bg: torch.Tensor,
    sigma_grid: tuple = (0.05, 0.10, 0.20, 0.50, 1.00),
) -> Tuple[float, float]:
    """Pick sigma from the grid that maximizes the LOO support margin."""
    best_sigma, best_score = sigma_grid[0], -float("inf")
    for s in sigma_grid:
        sc = loo_margin_score(feat_fg, feat_bg, s)
        if sc > best_score:
            best_score = sc
            best_sigma = s
    return best_sigma, best_score


@torch.no_grad()
def bg_heterogeneity(feat_bg: torch.Tensor) -> float:
    """BG feature heterogeneity = 1 - mean off-diagonal BG-BG cosine (diagnostic)."""
    n = feat_bg.shape[0]
    if n < 2:
        return 0.0
    if n <= 20000:
        sim = feat_bg @ feat_bg.T
        off_sum = sim.sum() - torch.diagonal(sim).sum()
    else:
        # memory-safe factored identity for high shot counts
        s = feat_bg.sum(dim=0)
        total = s @ s
        diag = (feat_bg * feat_bg).sum()
        off_sum = total - diag
    mean_off = float(off_sum.item()) / (n * (n - 1))
    return 1.0 - mean_off


@torch.no_grad()
def bilateral_propagate(
    m_kde: torch.Tensor,         # (N,)  KDE margin (= posterior l, tau subtracted at thresholding)
    feat_tgt: torch.Tensor,      # (N, D) L2-normalized features for the feature kernel
    img_rgb: torch.Tensor,       # (N, 3) per-patch mean RGB in [0,1]
    coords: torch.Tensor,        # (N, 2) patch-grid (row, col) coords
    fg_proto: torch.Tensor,      # (D,)  mean FG prototype (L2-normalized)
    bg_proto: torch.Tensor,      # (D,)  mean BG prototype (L2-normalized)
    tau_f: float = 0.20,         # feature kernel temperature
    tau_I: float = 0.05,         # RGB kernel temperature
    d_max: float = 16.0,         # spatial window radius (grid units)
    tau_m: float = 1.0,          # score-diff kernel temperature
    kappa: float = 0.0,          # class-proto kernel strength (FROST default 0 -> identity)
    n_iters: int = 10,
    alpha: float = 0.70,
) -> torch.Tensor:
    """bilateral-graph label propagation.

    Edge weight = product of 5 kernels:
      W_ab = exp(<F_a,F_b>/tau_f) * exp(-||I_a-I_b||^2/tau_I)
           * 1[|p_a-p_b| <= d_max] * exp(-(s_a-s_b)^2/tau_m)
           * exp(kappa * (FG_aff_a*FG_aff_b + BG_aff_a*BG_aff_b))
    Update: m^{t+1} = (1-alpha)*m^0 + alpha * D^{-1} W m^t (row-normalized).
    With kappa=0 the 5th (class-proto) kernel is exp(0)=1, so propagation routes
    only within feature/color/space-similar, score-smooth regions.
    """
    device = feat_tgt.device

    # 1) feature similarity (cosine; feat_tgt is L2-normalized)
    F_sim = feat_tgt @ feat_tgt.T
    W = torch.exp(F_sim / float(tau_f))

    # 2) RGB similarity
    dI2 = ((img_rgb.unsqueeze(1) - img_rgb.unsqueeze(0)) ** 2).sum(dim=-1)
    W = W * torch.exp(-dI2 / float(tau_I))

    # 3) spatial window (hard mask within d_max radius)
    dp = ((coords.unsqueeze(1).float() - coords.unsqueeze(0).float()) ** 2).sum(dim=-1).sqrt()
    W = W * (dp <= float(d_max)).to(W.dtype)

    # 4) score-difference kernel (within-confidence-band smoothing)
    s_norm = m_kde / (m_kde.std() + 1e-6)
    ds2 = (s_norm.unsqueeze(1) - s_norm.unsqueeze(0)) ** 2
    W = W * torch.exp(-ds2 / float(tau_m))

    # 5) class-prototype affinity (identity when kappa == 0)
    fg_aff = feat_tgt @ fg_proto
    bg_aff = feat_tgt @ bg_proto
    proto = (fg_aff.unsqueeze(1) * fg_aff.unsqueeze(0)
             + bg_aff.unsqueeze(1) * bg_aff.unsqueeze(0))
    W = W * torch.exp(float(kappa) * proto)

    W.fill_diagonal_(0.0)
    deg = W.sum(dim=1, keepdim=True).clamp(min=1e-6)
    P = W / deg

    m = m_kde.clone()
    a = float(alpha)
    for _ in range(int(n_iters)):
        m = (1.0 - a) * m_kde + a * (P @ m)
    return m


@torch.no_grad()
def density_ratio_predict(
    feat_tgt_deb_flat: torch.Tensor,  # (N_t, D), debiased+whitened, L2-normalized
    feat_ref_fg: torch.Tensor,        # (N_fg, D), same space
    feat_ref_bg: torch.Tensor,        # (N_bg, D), same space
    h: int,
    w: int,
    feat_tgt_bil_flat: torch.Tensor,  # (N_t, D) features for the bilateral feature kernel
    bil_img_rgb: torch.Tensor,          # (N_t, 3) per-patch RGB in [0,1]
    sigma_grid: tuple = (0.05, 0.10, 0.20, 0.50, 1.00),
    bil_tau_f: float = 0.20,
    bil_tau_I: float = 0.05,
    bil_d_max: float = 16.0,
    bil_tau_m: float = 1.0,
    bil_kappa: float = 0.0,
    bil_n_iters: int = 10,
    bil_alpha: float = 0.70,
    threshold: float = 0.0,
) -> Tuple[torch.Tensor, Dict]:
    """FROST KDE-posterior + bilateral smoothing.

    Returns (binary mask (h, w), info) where ``info['ell_smooth']`` is the
    continuous bilaterally-smoothed posterior at patch resolution (CPU), consumed by the
    upsample-then-threshold step in the model to place the boundary at sub-patch resolution.
    """
    if feat_ref_fg.shape[0] == 0 or feat_ref_bg.shape[0] == 0:
        return torch.zeros(h, w, dtype=torch.bool, device=feat_tgt_deb_flat.device), {}

    # 1) bandwidth via LOO support margin
    sigma, margin = estimate_bandwidth(feat_ref_fg, feat_ref_bg, sigma_grid)

    # 2) KDE log-posterior ratio on the target patches
    ell_tgt = kde_log_posterior(feat_tgt_deb_flat, feat_ref_fg, feat_ref_bg, sigma)

    # 3) bilateral propagation of the continuous posterior
    device = feat_tgt_bil_flat.device
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    coords = torch.stack([ys.flatten(), xs.flatten()], dim=-1)  # (N, 2)
    mu_fg = F.normalize(feat_ref_fg.mean(dim=0), p=2, dim=0)
    mu_bg = F.normalize(feat_ref_bg.mean(dim=0), p=2, dim=0)
    bg_het = bg_heterogeneity(feat_ref_bg)  # diagnostic only
    ell_smooth = bilateral_propagate(
        ell_tgt, feat_tgt_bil_flat, bil_img_rgb, coords,
        fg_proto=mu_fg, bg_proto=mu_bg,
        tau_f=bil_tau_f, tau_I=bil_tau_I, d_max=bil_d_max,
        tau_m=bil_tau_m, kappa=bil_kappa,
        n_iters=bil_n_iters, alpha=bil_alpha,
    )

    # 4) threshold (Bayes-optimal tau = 0)
    tau = float(threshold)
    mask = (ell_smooth > tau).view(h, w)

    info = {
        "sigma": float(sigma),
        "loo_margin": float(margin),
        "bg_het": float(bg_het),
        "tau": float(tau),
        "n_pos": int(mask.sum().item()),
        # continuous posterior for sub-patch (sub-patch boundary placement
        "ell_smooth": ell_smooth.view(h, w).cpu(),
    }
    return mask, info
