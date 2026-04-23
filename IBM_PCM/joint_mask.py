"""Joint XRF + ptycho thresholding for cleaner instance boundaries.

Motivation: the XRF-only mask produces boundaries that are soft (XRF is ~1.6x
coarser than ptycho) and includes regions that aren't visible in ptycho at all.
Since the model is trained on ptycho, anything in the label mask that has no
ptycho signal is effectively noise: the model can't learn to reproduce it and
will get penalized for correctly predicting nothing there.

Approach: intersect the XRF union mask with a ptycho intensity threshold, so a
voxel is labeled foreground only if BOTH (a) some aligned XRF channel says
"metal is here" AND (b) ptycho shows density above background. Boundaries now
follow the crisp ptycho edges, and XRF-only components (no ptycho signal)
disappear naturally when CC labeling drops below the min-size floor.

Ptycho thresholding options
---------------------------
- "otsu":        global Otsu on the whole volume. Parameter-free, sharp split
                 between background and features. Aggressive (~7.6% fg).
- "percentile":  global percentile threshold (e.g. P85 → top 15%). Tunable,
                 softer cutoff than Otsu.
- "bg_quantile": compute ptycho values on XRF-negative voxels (known
                 background), threshold at `q`-th percentile of that
                 distribution. Most principled: "above what background
                 reaches". Equivalent to a false-positive-rate cap.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu


def ptycho_threshold(
    ptycho: np.ndarray,
    xrf_mask: np.ndarray,
    method: str = "otsu",
    percentile: float = 85.0,
    bg_quantile: float = 99.0,
) -> float:
    """Return a single threshold value for ptycho intensity."""
    if method == "otsu":
        return float(threshold_otsu(ptycho))
    if method == "percentile":
        return float(np.percentile(ptycho, percentile))
    if method == "bg_quantile":
        bg_vals = ptycho[~xrf_mask]
        return float(np.percentile(bg_vals, bg_quantile))
    raise ValueError(f"unknown method {method!r}")


def build_joint_labels(
    aligned_xrf: np.ndarray,
    ptycho: np.ndarray,
    *,
    xrf_percentile: float = 95.0,
    xrf_closing_iters: int = 1,
    ptycho_method: str = "otsu",
    ptycho_percentile: float = 85.0,
    ptycho_bg_quantile: float = 99.0,
    post_closing_iters: int = 1,
    min_component_voxels: int = 100,
) -> tuple[np.ndarray, dict]:
    """Build instance labels from joint XRF ∩ ptycho thresholding.

    Returns (labels3d, info) where info records the intermediate masks' fg%
    and the chosen ptycho threshold.
    """
    # 1) XRF union mask (same as build_instance_labels)
    def norm01(a):
        lo, hi = np.percentile(a, (1, 99.9))
        return np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1).astype(np.float32)

    chans = np.stack([norm01(c) for c in aligned_xrf])
    thr_xrf = np.array([np.percentile(c, xrf_percentile) for c in chans])
    xrf_mask = (chans > thr_xrf[:, None, None, None]).any(axis=0)
    if xrf_closing_iters > 0:
        xrf_mask = ndi.binary_closing(xrf_mask, iterations=xrf_closing_iters)

    # 2) Ptycho threshold
    thr_pty = ptycho_threshold(
        ptycho, xrf_mask,
        method=ptycho_method,
        percentile=ptycho_percentile,
        bg_quantile=ptycho_bg_quantile,
    )
    pty_mask = ptycho > thr_pty

    # 3) Intersect
    joint = xrf_mask & pty_mask

    # 4) Post-closing to fill ptycho-noise-induced single-voxel gaps
    if post_closing_iters > 0:
        joint = ndi.binary_closing(joint, iterations=post_closing_iters)
        # Don't let closing expand outside the XRF ROI — keeps joint ⊆ xrf_mask
        joint = joint & xrf_mask

    # 5) CC label + min-size filter
    lbl, _ = ndi.label(joint, structure=np.ones((3, 3, 3), dtype=np.uint8))
    sizes = np.bincount(lbl.ravel())
    keep = sizes >= min_component_voxels
    keep[0] = False
    remap = np.zeros_like(sizes)
    remap[keep] = np.arange(1, keep.sum() + 1)
    labels = remap[lbl].astype(np.int32)

    info = {
        "xrf_fg_pct": float(100.0 * xrf_mask.mean()),
        "pty_fg_pct": float(100.0 * pty_mask.mean()),
        "joint_fg_pct": float(100.0 * joint.mean()),
        "final_fg_pct": float(100.0 * (labels > 0).mean()),
        "ptycho_threshold": thr_pty,
        "n_instances": int(labels.max()),
    }
    return labels, info
