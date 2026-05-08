#!/usr/bin/env python3
"""Run the fine-tuned SAM3 model over the full ptycho volume along all 3
axes (xy / xz / yz), vote across the per-axis binary predictions, and
stitch the voted volume into 3D instance labels.

Voting drops voxels only seen from a single viewpoint (typical 2D false
positives) while keeping voxels the model locates from multiple directions.

Outputs go to IBM_PCM/vis_3d/:
- pred_per_axis.npz           — raw binary {xy, xz, yz} predictions
- pred_labels3d.npy           — (Z, Y, X) int32 instance labels after vote+CC
- mip_xyz.png                 — 3-panel max-intensity projection along each axis
- slice_grid_xy.png           — 12 evenly-spaced xy slices with overlay

Compute: ~1218 forward passes (148 + 535 + 535). ≈10-20 min on 1× A100.

Run on a GPU node:
    python3 IBM_PCM/visualize_3d_segmentation.py \\
        --ckpt runs/<run_name>/checkpoints/checkpoint.pt

    # Re-render from cached predictions (no GPU)
    python3 IBM_PCM/visualize_3d_segmentation.py --skip-inference
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

HERE = Path(__file__).parent
REPO = HERE.parent

DEFAULT_CKPT = REPO / "runs/ibm_pcm_ft_short/checkpoints/checkpoint.pt"
DEFAULT_EXPORT_ROOT = HERE / "sam3_export"
DEFAULT_BACKDROP_ROOT = HERE / "sam3_export"
DEFAULT_OUT_DIR = HERE / "vis_3d"

# Each axis produces one binary vote per voxel. A voxel is kept if at least
# VOTE_THRESHOLD of the 3 axis predictions agree — 2-of-3 drops single-axis
# false positives; 1-of-3 keeps small features only one axis happens to see.
AXES = {"xy": 0, "xz": 1, "yz": 2}
DEFAULT_VOTE_THRESHOLD = 2

CONF_THRESH = 0.3           # model detection threshold (needs re-inference to change)
DEFAULT_TEXT_PROMPT = "IC feature"
DEFAULT_MIN_COMPONENT_VOXELS = 50
N_SLICE_GRID = 12


# --------------------------------------------------------------------------- #
def _predict_frame(processor: Sam3Processor, img: Image.Image, text_prompt: str) -> np.ndarray:
    """Text-prompted inference on one PIL frame → binary (H, W) mask."""
    state = processor.set_image(img)
    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(prompt=text_prompt, state=state)
    H, W = img.height, img.width
    out = np.zeros((H, W), dtype=bool)
    if "masks" not in state or state["masks"] is None:
        return out
    pm = state["masks"]
    if torch.is_tensor(pm):
        pm = pm.cpu().numpy()
    for j in range(pm.shape[0]):
        m = pm[j]
        while m.ndim > 2:
            m = m[0] if m.shape[0] == 1 else m.any(axis=0)
        out |= m.astype(bool)
    return out


def run_axis_inference(
    processor: Sam3Processor,
    axis_name: str,
    axis: int,
    shape_full: tuple,
    export_root: Path,
    text_prompt: str = DEFAULT_TEXT_PROMPT,
) -> np.ndarray:
    """Run the model on every frame of the given axis; stitch each frame's
    binary mask back into a full (Z, Y, X) volume by indexing along `axis`."""
    frames_dir = export_root / "video" / axis_name / "frames"
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    assert frame_paths, f"no frames in {frames_dir}"
    vol = np.zeros(shape_full, dtype=np.uint8)
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i, p in enumerate(frame_paths):
            mask = _predict_frame(processor, Image.open(p).convert("RGB"), text_prompt)
            # Slot this 2D prediction back into its slice of the 3D volume.
            sl = [slice(None)] * 3
            sl[axis] = i
            vol[tuple(sl)] = mask.astype(np.uint8)
            if (i + 1) % 50 == 0 or i == len(frame_paths) - 1:
                print(f"    [{axis_name}] {i+1}/{len(frame_paths)}  "
                      f"({time.time()-t0:.1f}s)")
    return vol


def run_inference_all_axes(
    ckpt: Path, export_root: Path, text_prompt: str = DEFAULT_TEXT_PROMPT,
) -> dict[str, np.ndarray]:
    """Run per-axis inference on xy/xz/yz. Returns {axis_name: binary (Z,Y,X)}."""
    print(f"Loading model from {ckpt} …")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = build_sam3_image_model(
        checkpoint_path=str(ckpt), load_from_HF=False,
        enable_segmentation=True, device="cuda", eval_mode=True,
    )
    processor = Sam3Processor(model, confidence_threshold=CONF_THRESH)

    # Probe volume shape from the xy stack.
    xy_frames = sorted((export_root / "video/xy/frames").glob("*.jpg"))
    assert xy_frames, f"no xy frames under {export_root}"
    img0 = Image.open(xy_frames[0])
    shape_full = (len(xy_frames), img0.height, img0.width)
    print(f"  export root: {export_root}")
    print(f"  volume: {shape_full}")
    print(f"  text prompt: {text_prompt!r}")

    out = {}
    for name, axis in AXES.items():
        print(f"\n  [{name}] axis={axis}")
        out[name] = run_axis_inference(
            processor, name, axis, shape_full, export_root, text_prompt
        )
    return out


def vote(per_axis: dict[str, np.ndarray], threshold: int) -> np.ndarray:
    """Sum the 3 binary per-axis predictions, threshold at `threshold`."""
    stack = np.stack(list(per_axis.values()), axis=0).astype(np.uint8)
    votes = stack.sum(axis=0)
    for k in range(1, 4):
        print(f"  voxels with ≥{k} axis agreement: "
              f"{int((votes >= k).sum()):>10}  "
              f"({100*(votes >= k).mean():.2f}%)")
    return (votes >= threshold).astype(np.uint8)


def stitch_instances(binary_vol: np.ndarray, min_voxels: int) -> np.ndarray:
    """3D connected components on the binary prediction volume."""
    print("3D connected-component labeling …")
    lbl, n = ndi.label(binary_vol, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lbl.ravel())
    keep = sizes >= min_voxels
    keep[0] = False
    remap = np.zeros_like(sizes)
    remap[keep] = np.arange(1, keep.sum() + 1)
    labels = remap[lbl].astype(np.int32)
    print(f"  {n} raw components → {int(labels.max())} after size filter "
          f"(min={min_voxels} vox)  fg%={100*(labels>0).mean():.2f}")
    return labels


# --------------------------------------------------------------------------- #
# Visualizations
# --------------------------------------------------------------------------- #
def colorize(labels2d: np.ndarray) -> np.ndarray:
    """Map instance IDs → RGB, background → black."""
    cmap = plt.cm.get_cmap("tab20")
    rgb = np.zeros((*labels2d.shape, 3), dtype=np.float32)
    n = int(labels2d.max())
    for i in range(1, n + 1):
        c = np.array(cmap((i - 1) % 20))[:3]
        rgb[labels2d == i] = c
    return rgb


def render_mips(labels: np.ndarray, out_path: Path) -> None:
    """Max-projection of labels along Z / Y / X axes."""
    print("Rendering MIPs …")
    mips = []
    for axis, title in [(0, "XY (∥Z)"), (1, "XZ (∥Y)"), (2, "YZ (∥X)")]:
        # Project by taking the first nonzero label along the axis — gives
        # a colored projection rather than a binary silhouette.
        mip_ids = np.argmax(labels > 0, axis=axis)
        mip_vals = np.take_along_axis(
            labels, mip_ids[(slice(None),) * axis + (None,)], axis=axis
        ).squeeze(axis)
        # Fallback: where no foreground on this ray, show 0
        any_fg = (labels > 0).any(axis=axis)
        mip_vals = mip_vals * any_fg
        mips.append((title, mip_vals))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, m) in zip(axes, mips):
        ax.imshow(colorize(m))
        ax.set_title(f"{title}  —  {int(m.max())} instances visible")
        ax.axis("off")
    fig.suptitle(
        f"3D segmentation MIP  (volume {labels.shape}, "
        f"{int(labels.max())} instances)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def render_slice_grid(
    labels: np.ndarray, out_path: Path, backdrop_root: Path
) -> None:
    """12 evenly-spaced xy slices with the ptycho image behind instance overlay.

    `backdrop_root` is the original grayscale export — keeping it separate from
    the (possibly 3-slice-stack) model input makes the overlay readable."""
    print("Rendering slice grid …")
    Z = labels.shape[0]
    zs = np.linspace(0, Z - 1, N_SLICE_GRID).astype(int)
    frames_dir = backdrop_root / "video/xy/frames"

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for ax, z in zip(axes.ravel(), zs):
        img = Image.open(frames_dir / f"{z:05d}.jpg").convert("RGB")
        ax.imshow(img, cmap="gray")
        slc = labels[z]
        present = np.unique(slc); present = present[present > 0]
        if len(present):
            rgb = colorize(slc)
            alpha = np.where(slc > 0, 0.5, 0.0)
            rgba = np.dstack([rgb, alpha])
            ax.imshow(rgba)
        ax.set_title(f"z={z}  ({len(present)} instances)", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"XY slices through the predicted 3D segmentation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                    help="model checkpoint to run inference with")
    ap.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT,
                    help="directory with video/{xy,xz,yz}/frames the model sees")
    ap.add_argument("--backdrop-root", type=Path, default=DEFAULT_BACKDROP_ROOT,
                    help="grayscale export used only as slice-grid backdrop")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="where to write npz/npy + visualization images")
    ap.add_argument("--skip-inference", action="store_true",
                    help="reuse existing pred_per_axis.npz (skip GPU)")
    ap.add_argument("--vote-threshold", type=int, default=DEFAULT_VOTE_THRESHOLD,
                    choices=[1, 2, 3],
                    help="min axes that must agree on a voxel (1 = recall small "
                         "features visible from one view; 3 = strictest, only "
                         "voxels all 3 axes confirm)")
    ap.add_argument("--min-component-voxels", type=int,
                    default=DEFAULT_MIN_COMPONENT_VOXELS,
                    help="min size of a kept 3D connected component")
    ap.add_argument("--text-prompt", type=str, default=DEFAULT_TEXT_PROMPT,
                    help="text query for SAM 3 (e.g. 'tungsten grain' for the "
                         "per-material checkpoint)")
    args = ap.parse_args()

    args.out_dir.mkdir(exist_ok=True, parents=True)
    per_axis_npy = args.out_dir / "pred_per_axis.npz"
    labels_npy = args.out_dir / "pred_labels3d.npy"

    if args.skip_inference and per_axis_npy.exists():
        print(f"Loading cached {per_axis_npy} …")
        loaded = np.load(per_axis_npy)
        per_axis = {k: loaded[k] for k in loaded.files}
    else:
        per_axis = run_inference_all_axes(args.ckpt, args.export_root, args.text_prompt)
        np.savez_compressed(per_axis_npy, **per_axis)
        print(f"  saved per-axis predictions → {per_axis_npy}")

    print(f"\nVoting (threshold={args.vote_threshold}-of-{len(per_axis)}) …")
    binary = vote(per_axis, args.vote_threshold)
    labels = stitch_instances(binary, args.min_component_voxels)
    np.save(labels_npy, labels)
    print(f"  saved {labels_npy}")

    render_mips(labels, args.out_dir / "mip_xyz.png")
    render_slice_grid(labels, args.out_dir / "slice_grid_xy.png", args.backdrop_root)

    print(f"\nAll outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
