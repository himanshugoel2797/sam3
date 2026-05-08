#!/usr/bin/env python3
"""Run the fine-tuned SAM3 model directly over a 3D TIFF volume — no
pre-export step required.  Same vote-then-CC pipeline as
visualize_3d_segmentation.py, but RGB frames are built in-memory from
the tiff instead of being read off disk.

Frame encoding matches export_sam3.py: each 2D frame is the prev/curr/next
slice along the active axis stacked into R/G/B (edges clamped), giving the
model 2.5D context in a single forward pass.

Outputs go to --out-dir:
- pred_per_axis.npz   — raw binary {xy, xz, yz} predictions
- pred_labels3d.npy   — (Z, Y, X) int32 instance labels after vote+CC
- mip_xyz.png         — 3-panel max-intensity projection along each axis
- slice_grid_xy.png   — 12 evenly-spaced xy slices with overlay

Run on a GPU node:
    python3 IBM_PCM/visualize_3d_segmentation_tiff.py \\
        --tiff IBM_PCM/Recon_obj_arg_fista.tiff \\
        --ckpt runs/<run_name>/checkpoints/checkpoint.pt

    # Re-render from cached predictions (no GPU)
    python3 IBM_PCM/visualize_3d_segmentation_tiff.py \\
        --tiff IBM_PCM/Recon_obj_arg_fista.tiff --skip-inference
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from PIL import Image
from scipy import ndimage as ndi

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

HERE = Path(__file__).parent
REPO = HERE.parent

DEFAULT_CKPT = REPO / "runs/ibm_pcm_ft_short/checkpoints/checkpoint.pt"
DEFAULT_TIFF = HERE / "Recon_obj_arg_fista.tiff"
DEFAULT_OUT_DIR = HERE / "vis_3d_tiff"

AXES = {"xy": 0, "xz": 1, "yz": 2}
DEFAULT_VOTE_THRESHOLD = 2

CONF_THRESH = 0.3
DEFAULT_TEXT_PROMPT = "IC feature"
DEFAULT_MIN_COMPONENT_VOXELS = 50
N_SLICE_GRID = 12

# Match the percentile normalization used in export_sam3.py so the model
# sees the same intensity distribution it was trained on.
NORM_LO_PCT, NORM_HI_PCT = 1.0, 99.9


# --------------------------------------------------------------------------- #
def norm_to_uint8(vol: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(vol, (NORM_LO_PCT, NORM_HI_PCT))
    return np.clip(255 * (vol - lo) / max(hi - lo, 1e-12), 0, 255).astype(np.uint8)


def make_rgb_frame(vol_u8: np.ndarray, axis: int, i: int) -> Image.Image:
    """3-slice RGB stack (prev/curr/next) along `axis`, edges clamped."""
    n = vol_u8.shape[axis]
    prev_i = max(0, i - 1)
    next_i = min(n - 1, i + 1)
    r = np.take(vol_u8, prev_i, axis=axis)
    g = np.take(vol_u8, i, axis=axis)
    b = np.take(vol_u8, next_i, axis=axis)
    return Image.fromarray(np.stack([r, g, b], axis=-1), "RGB")


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
    vol_u8: np.ndarray,
    axis_name: str,
    axis: int,
    text_prompt: str,
) -> np.ndarray:
    """Run the model on every slice of the given axis; stitch each frame's
    binary mask back into a full (Z, Y, X) volume by indexing along `axis`."""
    n_frames = vol_u8.shape[axis]
    vol = np.zeros(vol_u8.shape, dtype=np.uint8)
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(n_frames):
            mask = _predict_frame(processor, make_rgb_frame(vol_u8, axis, i), text_prompt)
            sl = [slice(None)] * 3
            sl[axis] = i
            vol[tuple(sl)] = mask.astype(np.uint8)
            if (i + 1) % 50 == 0 or i == n_frames - 1:
                print(f"    [{axis_name}] {i+1}/{n_frames}  ({time.time()-t0:.1f}s)")
    return vol


def run_inference_all_axes(
    ckpt: Path, vol_u8: np.ndarray, text_prompt: str,
) -> dict[str, np.ndarray]:
    print(f"Loading model from {ckpt} …")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = build_sam3_image_model(
        checkpoint_path=str(ckpt), load_from_HF=False,
        enable_segmentation=True, device="cuda", eval_mode=True,
    )
    processor = Sam3Processor(model, confidence_threshold=CONF_THRESH)
    print(f"  volume: {vol_u8.shape}")
    print(f"  text prompt: {text_prompt!r}")

    out = {}
    for name, axis in AXES.items():
        print(f"\n  [{name}] axis={axis}")
        out[name] = run_axis_inference(processor, vol_u8, name, axis, text_prompt)
    return out


def vote(per_axis: dict[str, np.ndarray], threshold: int) -> np.ndarray:
    stack = np.stack(list(per_axis.values()), axis=0).astype(np.uint8)
    votes = stack.sum(axis=0)
    for k in range(1, 4):
        print(f"  voxels with ≥{k} axis agreement: "
              f"{int((votes >= k).sum()):>10}  "
              f"({100*(votes >= k).mean():.2f}%)")
    return (votes >= threshold).astype(np.uint8)


def stitch_instances(binary_vol: np.ndarray, min_voxels: int) -> np.ndarray:
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
def colorize(labels2d: np.ndarray) -> np.ndarray:
    cmap = plt.cm.get_cmap("tab20")
    rgb = np.zeros((*labels2d.shape, 3), dtype=np.float32)
    n = int(labels2d.max())
    for i in range(1, n + 1):
        c = np.array(cmap((i - 1) % 20))[:3]
        rgb[labels2d == i] = c
    return rgb


def render_mips(labels: np.ndarray, out_path: Path) -> None:
    print("Rendering MIPs …")
    mips = []
    for axis, title in [(0, "XY (∥Z)"), (1, "XZ (∥Y)"), (2, "YZ (∥X)")]:
        mip_ids = np.argmax(labels > 0, axis=axis)
        mip_vals = np.take_along_axis(
            labels, mip_ids[(slice(None),) * axis + (None,)], axis=axis
        ).squeeze(axis)
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


def render_slice_grid(labels: np.ndarray, vol_u8: np.ndarray, out_path: Path) -> None:
    """12 evenly-spaced xy slices with the ptycho slice behind instance overlay."""
    print("Rendering slice grid …")
    Z = labels.shape[0]
    zs = np.linspace(0, Z - 1, N_SLICE_GRID).astype(int)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for ax, z in zip(axes.ravel(), zs):
        ax.imshow(vol_u8[z], cmap="gray")
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
def load_volume(tiff_path: Path) -> np.ndarray:
    print(f"Loading volume from {tiff_path} …")
    vol = tifffile.imread(str(tiff_path))
    if vol.ndim != 3:
        raise ValueError(f"expected 3D tiff, got shape {vol.shape}")
    vol_u8 = norm_to_uint8(vol.astype(np.float32))
    print(f"  shape={vol_u8.shape}  dtype={vol.dtype}→uint8")
    return vol_u8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiff", type=Path, default=DEFAULT_TIFF,
                    help="3D TIFF volume (Z, Y, X)")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                    help="model checkpoint to run inference with")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="where to write npz/npy + visualization images")
    ap.add_argument("--skip-inference", action="store_true",
                    help="reuse existing pred_per_axis.npz (skip GPU)")
    ap.add_argument("--vote-threshold", type=int, default=DEFAULT_VOTE_THRESHOLD,
                    choices=[1, 2, 3],
                    help="min axes that must agree on a voxel")
    ap.add_argument("--min-component-voxels", type=int,
                    default=DEFAULT_MIN_COMPONENT_VOXELS,
                    help="min size of a kept 3D connected component")
    ap.add_argument("--text-prompt", type=str, default=DEFAULT_TEXT_PROMPT,
                    help="text query for SAM 3")
    args = ap.parse_args()

    args.out_dir.mkdir(exist_ok=True, parents=True)
    per_axis_npy = args.out_dir / "pred_per_axis.npz"
    labels_npy = args.out_dir / "pred_labels3d.npy"

    vol_u8 = load_volume(args.tiff)

    if args.skip_inference and per_axis_npy.exists():
        print(f"Loading cached {per_axis_npy} …")
        loaded = np.load(per_axis_npy)
        per_axis = {k: loaded[k] for k in loaded.files}
    else:
        per_axis = run_inference_all_axes(args.ckpt, vol_u8, args.text_prompt)
        np.savez_compressed(per_axis_npy, **per_axis)
        print(f"  saved per-axis predictions → {per_axis_npy}")

    print(f"\nVoting (threshold={args.vote_threshold}-of-{len(per_axis)}) …")
    binary = vote(per_axis, args.vote_threshold)
    labels = stitch_instances(binary, args.min_component_voxels)
    np.save(labels_npy, labels)
    print(f"  saved {labels_npy}")

    render_mips(labels, args.out_dir / "mip_xyz.png")
    render_slice_grid(labels, vol_u8, args.out_dir / "slice_grid_xy.png")

    print(f"\nAll outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
