#!/usr/bin/env python3
"""Sweep joint XRF + ptycho thresholding strategies and render side-by-side
comparisons with the XRF-only baseline on the same val frames used by
run_inference_vis.py.

Writes visualizations + stats to ./joint_threshold_sweep/.
"""

import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi

from joint_mask import build_joint_labels

# --------------------------------------------------------------------------- #
# Paths & config
# --------------------------------------------------------------------------- #
HERE = Path(__file__).parent
ALIGNED_DIR = HERE / "aligned"
PTYCHO_PATH = HERE / "Recon_obj_arg_fista.tiff"
XRF_CHANNELS = ["Ge_K_fista", "Te_L_fista", "Ti_K_fista", "W_L_fista"]
VAL_ANN = HERE / "sam3_export/coco/annotations_val.json"
IMG_ROOT = HERE / "sam3_export"
OUT_DIR = HERE / "joint_threshold_sweep"

MASK_PERCENTILE = 95.0
MASK_CLOSING_ITERS = 1
POST_CLOSING_ITERS = 1
MIN_COMPONENT_VOXELS = 100
CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)

N_SAMPLES = 8
SEED = 42
VIDEO_AXES = {"xy": 0, "xz": 1, "yz": 2}

# Variant definitions. name + kwargs passed to build_joint_labels;
# "baseline" = None params (use existing xrf-only code path).
VARIANTS = [
    {"name": "baseline"},
    {"name": "joint Otsu",
     "params": dict(ptycho_method="otsu")},
    {"name": "joint P85",
     "params": dict(ptycho_method="percentile", ptycho_percentile=85.0)},
    {"name": "joint P90",
     "params": dict(ptycho_method="percentile", ptycho_percentile=90.0)},
    {"name": "bg-quant P99",
     "params": dict(ptycho_method="bg_quantile", ptycho_bg_quantile=99.0)},
]


# --------------------------------------------------------------------------- #
# Baseline (XRF-only, closing=1)
# --------------------------------------------------------------------------- #
def build_baseline(aligned_xrf):
    def norm01(a):
        lo, hi = np.percentile(a, (1, 99.9))
        return np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1).astype(np.float32)

    chans = np.stack([norm01(c) for c in aligned_xrf])
    thr = np.array([np.percentile(c, MASK_PERCENTILE) for c in chans])
    mask = (chans > thr[:, None, None, None]).any(axis=0)
    if MASK_CLOSING_ITERS > 0:
        mask = ndi.binary_closing(mask, iterations=MASK_CLOSING_ITERS)
    lbl, _ = ndi.label(mask, structure=CONNECTIVITY)
    sizes = np.bincount(lbl.ravel())
    keep = sizes >= MIN_COMPONENT_VOXELS
    keep[0] = False
    remap = np.zeros_like(sizes)
    remap[keep] = np.arange(1, keep.sum() + 1)
    labels = remap[lbl].astype(np.int32)
    return labels, {
        "final_fg_pct": float(100.0 * (labels > 0).mean()),
        "n_instances": int(labels.max()),
    }


# --------------------------------------------------------------------------- #
# Visualization helpers
# --------------------------------------------------------------------------- #
def instance_slice_to_masks(slice2d):
    present = np.unique(slice2d)
    present = present[present > 0]
    return [(slice2d == i).astype(np.uint8) for i in present]


def make_color_overlay(masks, alpha=0.5):
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

    print("Loading ptycho + aligned XRF …")
    ptycho = tifffile.imread(PTYCHO_PATH).astype(np.float32)
    aligned_xrf = np.stack([
        tifffile.imread(ALIGNED_DIR / f"{n}_aligned.tif").astype(np.float32)
        for n in XRF_CHANNELS
    ])
    print(f"  ptycho {ptycho.shape}  |  xrf {aligned_xrf.shape}")

    label_volumes = {}
    all_info = []
    for v in VARIANTS:
        name = v["name"]
        t0 = time.time()
        if name == "baseline":
            labels, info = build_baseline(aligned_xrf)
        else:
            labels, info = build_joint_labels(
                aligned_xrf, ptycho,
                xrf_percentile=MASK_PERCENTILE,
                xrf_closing_iters=MASK_CLOSING_ITERS,
                post_closing_iters=POST_CLOSING_ITERS,
                min_component_voxels=MIN_COMPONENT_VOXELS,
                **v["params"],
            )
        elapsed = time.time() - t0
        info["name"] = name
        info["params"] = v.get("params")
        info["elapsed_sec"] = elapsed
        all_info.append(info)
        label_volumes[name] = labels
        print(f"  {name}: n={info['n_instances']}  fg%={info['final_fg_pct']:.2f}  "
              f"pty_thr={info.get('ptycho_threshold', 'n/a')}  "
              f"({elapsed:.1f}s)")

    with open(OUT_DIR / "stats.json", "w") as f:
        json.dump(all_info, f, indent=2, default=float)

    # Pick val samples (match run_inference_vis.py)
    with open(VAL_ANN) as f:
        coco = json.load(f)
    img_id_to_info = {img["id"]: img for img in coco["images"]}
    random.seed(SEED)
    sample_ids = random.sample(
        list(img_id_to_info.keys()), min(N_SAMPLES, len(img_id_to_info))
    )

    n_cols = 1 + len(VARIANTS)
    print(f"\nRendering {len(sample_ids)} comparison plots …")
    for idx, img_id in enumerate(sample_ids):
        info = img_id_to_info[img_id]
        video, frame = info["video"], info["frame"]
        axis = VIDEO_AXES[video]
        image = Image.open(IMG_ROOT / info["file_name"]).convert("RGB")

        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.5))
        axes[0].imshow(image)
        axes[0].set_title(f"Input\n{info['file_name']}", fontsize=9)
        axes[0].axis("off")

        for col, v in enumerate(VARIANTS, start=1):
            labels = label_volumes[v["name"]]
            slice2d = np.take(labels, frame, axis=axis)
            masks = instance_slice_to_masks(slice2d)
            axes[col].imshow(image)
            overlay = make_color_overlay(masks)
            if overlay is not None:
                axes[col].imshow(overlay)
            total = int(labels.max())
            axes[col].set_title(
                f"{v['name']}\n{len(masks)} in slice / {total} total",
                fontsize=9,
            )
            axes[col].axis("off")

        out_path = OUT_DIR / f"joint_{idx:02d}_img{img_id}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{idx+1}/{len(sample_ids)}] {info['file_name']} → {out_path}")

    print("\n" + "=" * 80)
    print(f"{'variant':>18}  {'n_inst':>7}  {'fg%':>6}  "
          f"{'pty_thr':>10}  {'sec':>6}")
    print("-" * 80)
    for s in all_info:
        thr = s.get("ptycho_threshold")
        thr_s = f"{thr:.5f}" if thr is not None else "—"
        print(f"{s['name']:>18}  {s['n_instances']:>7}  "
              f"{s['final_fg_pct']:>6.2f}  {thr_s:>10}  {s['elapsed_sec']:>6.1f}")
    print("=" * 80)
    print(f"\nWrote {len(sample_ids)} PNGs + stats.json to {OUT_DIR}/")


if __name__ == "__main__":
    main()
