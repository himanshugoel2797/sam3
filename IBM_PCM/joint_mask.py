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

Two labelers on top of the joint mask:
- ``build_joint_labels``: 26-connected CC + min-size filter. One label per
  spatially-contiguous blob. Gates and the contacts/PCM stack they touch end
  up in one component because XRF blur bridges them.
- ``build_material_split_labels``: cuts each joint blob at the per-voxel XRF
  argmax boundary before CC labeling. Voxels where W is the dominant element
  (gate metal) split off from voxels where Ti / Ge / Te dominates (contacts,
  chalcogenide stack), so a physically-fused blob becomes per-material
  instances. This is the production labeler for SAM 3 export.

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

from typing import Sequence

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


def _norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, (1, 99.9))
    return np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1).astype(np.float32)


def _resolve_xrf_pcts(
    xrf_percentile: float | Sequence[float], n_chans: int
) -> np.ndarray:
    if np.isscalar(xrf_percentile):
        return np.full(n_chans, float(xrf_percentile))
    pcts = np.asarray(xrf_percentile, dtype=float)
    if pcts.shape != (n_chans,):
        raise ValueError(
            f"xrf_percentile must be scalar or length-{n_chans} sequence, "
            f"got shape {pcts.shape}"
        )
    return pcts


def build_joint_mask(
    aligned_xrf: np.ndarray,
    ptycho: np.ndarray,
    *,
    xrf_percentile: float | Sequence[float] = 95.0,
    xrf_closing_iters: int = 1,
    ptycho_method: str = "otsu",
    ptycho_percentile: float = 85.0,
    ptycho_bg_quantile: float = 99.0,
    post_closing_iters: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (joint_mask, xrf_chans_norm, info) without CC labeling.

    Shared by both `build_joint_labels` (CC-only labeler) and
    `build_material_split_labels` (material-aware labeler). `xrf_chans_norm`
    is the per-channel P1/P99.9-normalized XRF stack — handed back so
    downstream labelers can compute per-voxel argmax without re-normalizing.
    """
    n_chans = len(aligned_xrf)
    xrf_pcts = _resolve_xrf_pcts(xrf_percentile, n_chans)

    chans = np.stack([_norm01(c) for c in aligned_xrf])
    thr_xrf = np.array([np.percentile(c, p) for c, p in zip(chans, xrf_pcts)])
    per_chan_masks = chans > thr_xrf[:, None, None, None]
    xrf_mask = per_chan_masks.any(axis=0)
    if xrf_closing_iters > 0:
        xrf_mask = ndi.binary_closing(xrf_mask, iterations=xrf_closing_iters)

    thr_pty = ptycho_threshold(
        ptycho, xrf_mask,
        method=ptycho_method,
        percentile=ptycho_percentile,
        bg_quantile=ptycho_bg_quantile,
    )
    pty_mask = ptycho > thr_pty

    joint = xrf_mask & pty_mask

    if post_closing_iters > 0:
        joint = ndi.binary_closing(joint, iterations=post_closing_iters)
        # Don't let closing expand outside the XRF ROI — keeps joint ⊆ xrf_mask
        joint = joint & xrf_mask

    info = {
        "xrf_fg_pct": float(100.0 * xrf_mask.mean()),
        "pty_fg_pct": float(100.0 * pty_mask.mean()),
        "joint_fg_pct": float(100.0 * joint.mean()),
        "ptycho_threshold": thr_pty,
        "xrf_percentiles": xrf_pcts.tolist(),
        "xrf_thresholds": thr_xrf.tolist(),
        "per_channel_fg_pct": [float(100.0 * m.mean()) for m in per_chan_masks],
    }
    return joint, chans, info


def _filter_and_remap(
    lbl: np.ndarray, min_voxels: int, start_id: int = 1
) -> tuple[np.ndarray, int]:
    sizes = np.bincount(lbl.ravel())
    keep = sizes >= min_voxels
    keep[0] = False
    kept = int(keep.sum())
    remap = np.zeros_like(sizes)
    remap[keep] = np.arange(start_id, start_id + kept)
    return remap[lbl].astype(np.int32), kept


def build_joint_labels(
    aligned_xrf: np.ndarray,
    ptycho: np.ndarray,
    *,
    xrf_percentile: float | Sequence[float] = 95.0,
    xrf_closing_iters: int = 1,
    ptycho_method: str = "otsu",
    ptycho_percentile: float = 85.0,
    ptycho_bg_quantile: float = 99.0,
    post_closing_iters: int = 1,
    min_component_voxels: int = 100,
) -> tuple[np.ndarray, dict]:
    """Build instance labels from joint XRF ∩ ptycho thresholding.

    `xrf_percentile` is either a scalar (broadcast to every channel) or a
    sequence of length N matching `aligned_xrf`'s channel axis — useful when
    one element fingerprints a structurally weaker feature (e.g. W in
    FinFET gates) and warrants a softer threshold than bright contact-metal
    channels.

    Returns (labels3d, info) where info records the intermediate masks' fg%
    and the chosen thresholds (per-channel for XRF).
    """
    joint, _, info = build_joint_mask(
        aligned_xrf, ptycho,
        xrf_percentile=xrf_percentile,
        xrf_closing_iters=xrf_closing_iters,
        ptycho_method=ptycho_method,
        ptycho_percentile=ptycho_percentile,
        ptycho_bg_quantile=ptycho_bg_quantile,
        post_closing_iters=post_closing_iters,
    )

    lbl, _ = ndi.label(joint, structure=np.ones((3, 3, 3), dtype=np.uint8))
    labels, kept = _filter_and_remap(lbl, min_component_voxels)

    info["final_fg_pct"] = float(100.0 * (labels > 0).mean())
    info["n_instances"] = kept
    return labels, info


def build_material_split_labels(
    aligned_xrf: np.ndarray,
    ptycho: np.ndarray,
    *,
    xrf_percentile: float | Sequence[float] = 95.0,
    xrf_closing_iters: int = 1,
    ptycho_method: str = "otsu",
    ptycho_percentile: float = 85.0,
    ptycho_bg_quantile: float = 99.0,
    post_closing_iters: int = 1,
    smooth_iters: int = 0,
    min_component_voxels: int = 100,
) -> tuple[np.ndarray, dict]:
    """Joint mask + per-voxel XRF argmax → CC per material class.

    Each voxel's "material" is the index of the XRF channel with the highest
    normalized intensity (over the P1/P99.9-normalized stack). Cuts joint
    components at the argmax boundary so a fused gate-plus-contact blob
    becomes one CC per dominant element.

    `smooth_iters` runs that many 3x3x3 mode-filter passes on the material
    volume to suppress speckle near boundaries before CC labeling.

    Returns (labels3d, info). `info` adds `per_material_n_instances` (kept
    instances per channel, indexed to `aligned_xrf`'s channel axis) on top
    of the joint-mask info.
    """
    joint, chans, info = build_joint_mask(
        aligned_xrf, ptycho,
        xrf_percentile=xrf_percentile,
        xrf_closing_iters=xrf_closing_iters,
        ptycho_method=ptycho_method,
        ptycho_percentile=ptycho_percentile,
        ptycho_bg_quantile=ptycho_bg_quantile,
        post_closing_iters=post_closing_iters,
    )

    n_chans = chans.shape[0]
    material = np.argmax(chans, axis=0).astype(np.int8)
    for _ in range(smooth_iters):
        counts = np.stack([
            ndi.uniform_filter((material == m).astype(np.float32), size=3)
            for m in range(n_chans)
        ])
        material = np.argmax(counts, axis=0).astype(np.int8)

    out = np.zeros_like(joint, dtype=np.int32)
    next_id = 1
    per_material = []
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    for m in range(n_chans):
        sub = joint & (material == m)
        if not sub.any():
            per_material.append(0)
            continue
        lbl, _ = ndi.label(sub, structure=structure)
        sub_labels, kept = _filter_and_remap(lbl, min_component_voxels, start_id=next_id)
        per_material.append(kept)
        if kept == 0:
            continue
        nonzero = sub_labels > 0
        out[nonzero] = sub_labels[nonzero]
        next_id += kept

    info["final_fg_pct"] = float(100.0 * (out > 0).mean())
    info["n_instances"] = next_id - 1
    info["per_material_n_instances"] = per_material
    info["material_smooth_iters"] = int(smooth_iters)
    return out, info
