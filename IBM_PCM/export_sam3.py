"""Export aligned ptycho + XRF-derived mask as a SAM 3 fine-tuning dataset.

Produces two parallel representations in ./sam3_export/:

  1. video/   — SAM 2/3 video-tracking layout (JPEG frames + instance-ID PNG
                masks, one sub-directory per scan axis XY/XZ/YZ).  Each 3D
                connected component of the mask keeps a stable instance ID
                across every slice it appears in, which is exactly what SAM 3
                video/streaming fine-tuning expects.
  2. coco/    — COCO-format annotations.json with RLE segmentations + boxes,
                referencing the same JPEGs.  Useful for image-mode fine-tuning
                or for COCO-compatible augmentation pipelines (Roboflow-100-VL
                / ODinW style, which is what SAM 3 training code consumes).

Plus prompts.json with per-instance, per-frame bounding boxes and sampled
positive/negative point prompts (derived from the mask) so you can train in
prompt mode without recomputing them.

Run: `python3 export_sam3.py`.  Tunable constants at the top.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from pycocotools import mask as mask_utils


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
REF_VOLUME = 'Recon_obj_arg_fista.tiff'
ALIGNED_DIR = Path('aligned')
XRF_CHANNELS = ['Ge_K_fista', 'Te_L_fista', 'Ti_K_fista', 'W_L_fista']
OUT_DIR = Path('sam3_export')

MASK_PERCENTILE = 95.0   # user-validated threshold
MIN_COMPONENT_VOXELS = 100
CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)   # 26-connectivity
# Morphological closing fills tiny gaps from XRF-blur before CC labeling so
# fragments of the same physical feature aren't split into separate instances.
# Validated via relabel_sweep.py: iters=1 absorbs ~7 noise fragments without
# merging semantically distinct regions; iters>=3 collapses real structure.
MASK_CLOSING_ITERS = 1

# Per-frame prompt sampling
NUM_POS_POINTS = 5
NUM_NEG_POINTS = 3
EROSION_ITERS = 1        # interior-point sampling uses the eroded mask
POINT_RNG_SEED = 0

# Normalization of the ptycho reference for JPEG output
NORM_LO_PCT, NORM_HI_PCT = 1.0, 99.9

# Which axes to export as video sequences.  Each entry is (name, axis-index).
# axis 0 = Z → XY frames,  axis 1 = Y → XZ frames,  axis 2 = X → YZ frames.
VIDEOS = [('xy', 0), ('xz', 1), ('yz', 2)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_volumes() -> tuple[np.ndarray, np.ndarray]:
    ptycho = tifffile.imread(REF_VOLUME).astype(np.float32)
    chans = np.stack([
        tifffile.imread(ALIGNED_DIR / f'{n}_aligned.tif').astype(np.float32)
        for n in XRF_CHANNELS
    ])
    return ptycho, chans


def norm_to_uint8(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, (NORM_LO_PCT, NORM_HI_PCT))
    return np.clip(255 * (a - lo) / max(hi - lo, 1e-12), 0, 255).astype(np.uint8)


def build_instance_labels(aligned_xrf: np.ndarray) -> np.ndarray:
    """Threshold each aligned channel at its P95, OR into a union mask,
    drop tiny components, then label the 3D volume with stable instance IDs."""
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
    # remap labels to a contiguous range starting at 1
    remap = np.zeros_like(sizes)
    remap[keep] = np.arange(1, keep.sum() + 1)
    return remap[lbl].astype(np.int32)


def sample_points(slice_mask: np.ndarray, rng: np.random.Generator) -> dict:
    """Return prompt points for a single-instance 2D mask."""
    interior = ndi.binary_erosion(slice_mask, iterations=EROSION_ITERS) if slice_mask.any() else slice_mask
    pts = interior if interior.any() else slice_mask
    ys, xs = np.where(pts)
    if len(ys) == 0:
        return {'pos_points': [], 'neg_points': [], 'com': None}

    pos_idx = rng.choice(len(ys), size=min(NUM_POS_POINTS, len(ys)), replace=False)
    pos = [[int(xs[i]), int(ys[i])] for i in pos_idx]

    # negatives: random points outside a dilated mask (so they're clearly BG)
    dil = ndi.binary_dilation(slice_mask, iterations=3)
    nys, nxs = np.where(~dil)
    if len(nys) > 0:
        neg_idx = rng.choice(len(nys), size=min(NUM_NEG_POINTS, len(nys)), replace=False)
        neg = [[int(nxs[i]), int(nys[i])] for i in neg_idx]
    else:
        neg = []

    com = ndi.center_of_mass(slice_mask)   # (y, x)
    com_pt = [int(round(com[1])), int(round(com[0]))] if not np.isnan(com[0]) else None
    return {'pos_points': pos, 'neg_points': neg, 'com': com_pt}


def rle_encode(mask2d: np.ndarray) -> dict:
    """pycocotools RLE with bytes→ASCII str so it's JSON-safe."""
    rle = mask_utils.encode(np.asfortranarray(mask2d.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def instance_bbox(mask2d: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask2d)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------
def export() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ptycho, aligned_xrf = load_volumes()
    print(f'ptycho  {ptycho.shape}  |  xrf stack  {aligned_xrf.shape}')

    labels3d = build_instance_labels(aligned_xrf)
    n_instances = int(labels3d.max())
    assert n_instances <= 255, f'{n_instances} instances → switch mask dtype to uint16'
    print(f'3D instances: {n_instances}  (stable IDs across all slices)')

    ptycho_u8 = norm_to_uint8(ptycho)

    coco = {
        'info': {
            'description': 'Ptycho-tomo + XRF-derived union mask (aligned)',
            'source_volume': REF_VOLUME,
            'mask_percentile': MASK_PERCENTILE,
            'num_3d_instances': n_instances,
        },
        'categories': [{'id': 1, 'name': 'IC feature', 'supercategory': 'integrated_circuit'}],
        'images': [], 'annotations': [],
    }
    prompts = {'videos': {}}
    rng = np.random.default_rng(POINT_RNG_SEED)
    image_id = 0
    ann_id = 0

    for vid_name, axis in VIDEOS:
        vid_dir = OUT_DIR / 'video' / vid_name
        (vid_dir / 'frames').mkdir(parents=True, exist_ok=True)
        (vid_dir / 'masks').mkdir(parents=True, exist_ok=True)
        n_frames = ptycho_u8.shape[axis]
        height, width = [s for i, s in enumerate(ptycho_u8.shape) if i != axis]
        print(f'\n[{vid_name}] axis={axis}  n_frames={n_frames}  frame_size={(height, width)}')

        prompts['videos'][vid_name] = {
            'axis': axis, 'n_frames': n_frames, 'frame_height': height, 'frame_width': width,
            'frames': [],
        }

        for i in range(n_frames):
            frame = np.take(ptycho_u8, i, axis=axis)
            mask_slice = np.take(labels3d, i, axis=axis).astype(np.uint8)

            fname = f'{i:05d}.jpg'
            mname = f'{i:05d}.png'
            Image.fromarray(frame, 'L').convert('RGB').save(
                vid_dir / 'frames' / fname, quality=92, optimize=True,
            )
            Image.fromarray(mask_slice, 'L').save(vid_dir / 'masks' / mname, optimize=True)

            img_entry = {
                'id': image_id, 'file_name': f'video/{vid_name}/frames/{fname}',
                'width': width, 'height': height, 'video': vid_name, 'frame': i,
            }
            coco['images'].append(img_entry)

            frame_prompts = {'frame_index': i, 'image_id': image_id, 'objects': []}
            present = np.unique(mask_slice)
            present = present[present > 0]
            for inst_id in present.tolist():
                m = (mask_slice == inst_id)
                bbox = instance_bbox(m)
                if bbox is None:
                    continue
                rle = rle_encode(m)
                area = int(mask_utils.area(mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))))
                coco['annotations'].append({
                    'id': ann_id, 'image_id': image_id, 'category_id': 1,
                    'segmentation': rle, 'bbox': list(bbox), 'area': area,
                    'iscrowd': 0, 'instance_id': int(inst_id),
                })
                ann_id += 1

                pts = sample_points(m, rng)
                frame_prompts['objects'].append({
                    'instance_id': int(inst_id),
                    'bbox_xywh': list(bbox),
                    **pts,
                })

            prompts['videos'][vid_name]['frames'].append(frame_prompts)
            image_id += 1

        print(f'  wrote {n_frames} frames + masks → {vid_dir}')

    (OUT_DIR / 'coco').mkdir(exist_ok=True)
    with open(OUT_DIR / 'coco' / 'annotations.json', 'w') as f:
        json.dump(coco, f)
    with open(OUT_DIR / 'prompts.json', 'w') as f:
        json.dump(prompts, f)

    meta = {
        'total_images': image_id, 'total_annotations': ann_id,
        'num_3d_instances': n_instances,
        'video_sequences': [
            {'name': n, 'axis': a, 'n_frames': int(ptycho_u8.shape[a])} for n, a in VIDEOS
        ],
        'mask_percentile': MASK_PERCENTILE,
        'mask_closing_iters': MASK_CLOSING_ITERS,
        'min_component_voxels': MIN_COMPONENT_VOXELS,
        'ptycho_shape_zyx': list(ptycho.shape),
        'normalization': {'lo_pct': NORM_LO_PCT, 'hi_pct': NORM_HI_PCT, 'output': 'uint8 RGB JPEG'},
        'mask_format': 'uint8 PNG; pixel value = 3D instance ID (0 = background)',
        'notes': [
            'Instance IDs are stable across frames within a video (same 3D component).',
            'IDs are NOT stable across videos (xy / xz / yz share underlying 3D labels '
            'but the instance set visible in each slice differs).',
            'Prompts were sampled from the XRF-derived mask and inherit its ~1.6x coarser '
            'XY resolution — see README in project root.',
        ],
    }
    with open(OUT_DIR / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\n{OUT_DIR}/ written:')
    print(f'  images: {image_id}')
    print(f'  annotations: {ann_id}')
    print(f'  videos: {[n for n, _ in VIDEOS]}')


if __name__ == '__main__':
    export()
