#!/usr/bin/env python3
"""Sweep morphological-closing iterations on the XRF union mask to see how
instance definitions change, and render side-by-side comparisons on the same
val frames used by run_inference_vis.py.

Reads aligned XRF volumes from ./aligned/ (no re-registration).
Writes visualizations + stats to ./relabel_sweep/.
"""

import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
HERE = Path(__file__).parent
REPO = HERE.parent
ALIGNED_DIR = HERE / "aligned"
XRF_CHANNELS = ["Ge_K_fista", "Te_L_fista", "Ti_K_fista", "W_L_fista"]
VAL_ANN = HERE / "sam3_export/coco/annotations_val.json"
IMG_ROOT = HERE / "sam3_export"
OUT_DIR = HERE / "relabel_sweep"

MASK_PERCENTILE = 95.0
MIN_COMPONENT_VOXELS = 100
CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)

CLOSING_ITERS = [0, 1, 2, 3]   # variants to sweep
N_SAMPLES = 8                  # must match run_inference_vis.py
SEED = 42                      # must match run_inference_vis.py

VIDEO_AXES = {"xy": 0, "xz": 1, "yz": 2}


# --------------------------------------------------------------------------- #
# Mask building (mirrors export_sam3.build_instance_labels, with closing)
# --------------------------------------------------------------------------- #
def build_union_mask(aligned_xrf: np.ndarray) -> np.ndarray:
    def norm01(a):
        lo, hi = np.percentile(a, (1, 99.9))
        return np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1).astype(np.float32)

    chans = np.stack([norm01(c) for c in aligned_xrf])
    thr = np.array([np.percentile(c, MASK_PERCENTILE) for c in chans])
    return (chans > thr[:, None, None, None]).any(axis=0)


def label_instances(union_mask: np.ndarray, closing_iters: int) -> tuple[np.ndarray, dict]:
    """Apply closing (if >0), CC label, drop tiny components, return labels + stats."""
    mask = union_mask
    if closing_iters > 0:
        mask = ndi.binary_closing(mask, iterations=closing_iters)

    lbl, _ = ndi.label(mask, structure=CONNECTIVITY)
    sizes = np.bincount(lbl.ravel())
    keep = sizes >= MIN_COMPONENT_VOXELS
    keep[0] = False
    remap = np.zeros_like(sizes)
    remap[keep] = np.arange(1, keep.sum() + 1)
    labels = remap[lbl].astype(np.int32)

    kept_sizes = sizes[keep]
    stats = {
        "closing_iters": closing_iters,
        "n_instances": int(labels.max()),
        "foreground_voxels": int(mask.sum()),
        "foreground_pct": float(100.0 * mask.sum() / mask.size),
        "min_size": int(kept_sizes.min()) if kept_sizes.size else 0,
        "median_size": int(np.median(kept_sizes)) if kept_sizes.size else 0,
        "max_size": int(kept_sizes.max()) if kept_sizes.size else 0,
        "p90_size": int(np.percentile(kept_sizes, 90)) if kept_sizes.size else 0,
    }
    return labels, stats


# --------------------------------------------------------------------------- #
# Visualization helpers
# --------------------------------------------------------------------------- #
def instance_slice_to_masks(slice2d: np.ndarray) -> list[np.ndarray]:
    """Split a 2D instance-ID slice into per-instance binary masks."""
    present = np.unique(slice2d)
    present = present[present > 0]
    return [(slice2d == i).astype(np.uint8) for i in present]


def make_color_overlay(masks: list[np.ndarray], alpha: float = 0.5) -> np.ndarray | None:
    if not masks:
        return None
    h, w = masks[0].shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    cmap = plt.cm.get_cmap("tab20")
    for i, m in enumerate(masks):
        color = np.array(cmap(i % 20))
        color[3] = alpha
        overlay[m > 0] = color
    return overlay


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading aligned XRF volumes …")
    aligned_xrf = np.stack([
        tifffile.imread(ALIGNED_DIR / f"{n}_aligned.tif").astype(np.float32)
        for n in XRF_CHANNELS
    ])
    print(f"  xrf stack {aligned_xrf.shape}")

    print(f"\nBuilding union mask at P{MASK_PERCENTILE} …")
    union = build_union_mask(aligned_xrf)
    print(f"  foreground: {union.sum()} voxels ({100.0*union.sum()/union.size:.2f}%)")

    # Build labels for each closing iteration
    label_volumes = {}
    all_stats = []
    print("\nSweeping closing iterations:")
    for n_iter in CLOSING_ITERS:
        labels, stats = label_instances(union, n_iter)
        label_volumes[n_iter] = labels
        all_stats.append(stats)
        print(f"  closing={n_iter}  instances={stats['n_instances']}  "
              f"fg%={stats['foreground_pct']:.2f}  "
              f"median={stats['median_size']}  max={stats['max_size']}  "
              f"p90={stats['p90_size']}")

    # Save stats table
    with open(OUT_DIR / "stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    # Load val annotations + pick same samples as run_inference_vis.py
    print("\nSelecting val samples (same seed as run_inference_vis.py) …")
    with open(VAL_ANN) as f:
        coco = json.load(f)
    img_id_to_info = {img["id"]: img for img in coco["images"]}

    random.seed(SEED)
    sample_ids = random.sample(
        list(img_id_to_info.keys()), min(N_SAMPLES, len(img_id_to_info))
    )

    # Render comparison plots
    print(f"\nRendering {len(sample_ids)} comparison plots …")
    n_variants = len(CLOSING_ITERS)
    n_cols = 1 + n_variants   # input + one panel per variant

    for idx, img_id in enumerate(sample_ids):
        info = img_id_to_info[img_id]
        video = info["video"]          # "xy" / "xz" / "yz"
        frame = info["frame"]
        axis = VIDEO_AXES[video]

        img_path = IMG_ROOT / info["file_name"]
        image = Image.open(img_path).convert("RGB")

        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.5))

        axes[0].imshow(image)
        axes[0].set_title(f"Input\n{info['file_name']}", fontsize=9)
        axes[0].axis("off")

        for col, n_iter in enumerate(CLOSING_ITERS, start=1):
            slice2d = np.take(label_volumes[n_iter], frame, axis=axis)
            masks = instance_slice_to_masks(slice2d)

            axes[col].imshow(image)
            overlay = make_color_overlay(masks)
            if overlay is not None:
                axes[col].imshow(overlay)
            total_inst = label_volumes[n_iter].max()
            axes[col].set_title(
                f"closing={n_iter}  ({len(masks)} in slice / {total_inst} total)",
                fontsize=9,
            )
            axes[col].axis("off")

        out_path = OUT_DIR / f"sweep_{idx:02d}_img{img_id}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{idx+1}/{len(sample_ids)}] {info['file_name']} → {out_path}")

    # Summary table to stdout
    print("\n" + "=" * 72)
    print(f"{'closing':>8}  {'n_inst':>7}  {'fg%':>6}  "
          f"{'median':>8}  {'p90':>8}  {'max':>10}")
    print("-" * 72)
    for s in all_stats:
        print(f"{s['closing_iters']:>8}  {s['n_instances']:>7}  "
              f"{s['foreground_pct']:>6.2f}  {s['median_size']:>8}  "
              f"{s['p90_size']:>8}  {s['max_size']:>10}")
    print("=" * 72)
    print(f"\nWrote {len(sample_ids)} comparison PNGs + stats.json to {OUT_DIR}/")


if __name__ == "__main__":
    main()
