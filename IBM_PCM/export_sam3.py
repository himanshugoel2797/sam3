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

Each frame is encoded as a 3-slice RGB stack so the model gets per-pixel
2.5D context in a single 2D forward pass:

    R = ptycho slice [i-1]   (clamped at the low edge)
    G = ptycho slice [i]
    B = ptycho slice [i+1]   (clamped at the high edge)

Plus prompts.json with per-instance, per-frame bounding boxes and sampled
positive/negative point prompts (derived from the mask) so you can train in
prompt mode without recomputing them.

Run: `python3 export_sam3.py`.  Tunable constants at the top.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from pycocotools import mask as mask_utils

from joint_mask import build_material_split_labels


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
REF_VOLUME = 'Recon_obj_arg_fista.tiff'
ALIGNED_DIR = Path('aligned')
XRF_CHANNELS = ['Ge_K_fista', 'Te_L_fista', 'Ti_K_fista', 'W_L_fista']
# Material names aligned to XRF_CHANNELS — used as text prompts when emitting
# per-material categories. Pattern is "<material> grain" for SAM 3 text queries.
XRF_MATERIAL_NAMES = ['germanium', 'tellurium', 'titanium', 'tungsten']
OUT_DIR = Path('sam3_export')

UMBRELLA_CATEGORY_NAME = 'IC feature'

# Rich prompt aliases per XRF channel — every instance is annotated under
# *every* alias that applies. Lets the text encoder see multiple paraphrases
# for the same concept (element name vs symbol vs class umbrella vs density
# tier) so it can answer queries phrased differently from training. Indexed
# to XRF_CHANNELS order: [Ge, Te, Ti, W].
#
# Excludes structural/functional prompts ("gate metal", "via", etc.) — those
# would require manual labeling per instance, the XRF-derived material
# identity alone doesn't tell us whether a W blob is a gate or a via.
RICH_PROMPTS_PER_CHANNEL: list[list[str]] = [
    # Ge — chalcogenide, low-Z.
    ['germanium grain', 'chalcogenide', 'low-Z grain'],
    # Te — chalcogenide, low-Z.
    ['tellurium grain', 'chalcogenide', 'low-Z grain'],
    # Ti — mid-Z metal.
    ['titanium grain', 'metal'],
    # W — high-Z refractory metal.
    ['tungsten grain', 'metal', 'dense metal'],
]
# Generic prompts that apply to every instance regardless of material.
RICH_PROMPTS_GENERIC = [UMBRELLA_CATEGORY_NAME]

# Per-channel XRF percentiles, indexed to match XRF_CHANNELS above:
#   Ge_K, Te_L, Ti_K — PCM chalcogenide / liner channels: keep tight at P95.
#   W_L              — gate metal in the FinFET stack. Per-voxel W density in
#                      a gate stripe is lower than in via plugs, so P95 admits
#                      contacts but clips gate structure. P90 doubles the
#                      admitted W volume; the ptycho ∩ AND still gates against
#                      pure-XRF noise.
MASK_PERCENTILES = (95.0, 95.0, 95.0, 90.0)
MIN_COMPONENT_VOXELS = 100
# Morphological closing fills tiny gaps from XRF-blur before CC labeling so
# fragments of the same physical feature aren't split into separate instances.
# iters=1 absorbs noise fragments without merging semantically distinct
# regions; iters>=3 starts collapsing real structure.
MASK_CLOSING_ITERS = 1

# Joint XRF ∩ ptycho thresholding: XRF selects where metal is present; ptycho
# selects where density is above background. Intersecting snaps boundaries to
# the crisp ptycho edges and drops XRF-only regions (no visible ptycho signal
# → model can't learn them). P85 on ptycho preserves enough structure to
# avoid fragmenting traces while trimming ~40% of fg voxels that the model
# was being penalized for.
PTYCHO_METHOD = "percentile"
PTYCHO_PERCENTILE = 85.0
# Post-closing was bridging thin necks between physically distinct features —
# adjacent vias and parallel metal lines were getting fused into one CC. With
# the looser W=P90 mask there's enough connectivity that the gate stripes
# stay contiguous without help; turning closing off recovers ~30 instances
# at the same fg% by un-fusing adjacent components.
POST_CLOSING_ITERS = 0

# Material-split labeling: cut joint CCs at the per-voxel XRF-argmax boundary
# so a fused gate-plus-contact blob becomes one instance per dominant element
# (W → gate stack, Ti / Ge / Te → contacts / chalcogenide / liner). 0 smooth
# passes keeps argmax boundaries crisp; >0 runs a 3x3x3 mode filter on the
# material volume to suppress speckle.
MATERIAL_SMOOTH_ITERS = 0

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


def build_instance_labels(
    aligned_xrf: np.ndarray, ptycho: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Joint XRF ∩ ptycho mask, split per material — see
    `joint_mask.build_material_split_labels`. Cuts gates (W-pure) off from
    the contacts/PCM stack they touch so each material becomes a separate
    instance.
    """
    return build_material_split_labels(
        aligned_xrf, ptycho,
        xrf_percentile=MASK_PERCENTILES,
        xrf_closing_iters=MASK_CLOSING_ITERS,
        ptycho_method=PTYCHO_METHOD,
        ptycho_percentile=PTYCHO_PERCENTILE,
        post_closing_iters=POST_CLOSING_ITERS,
        smooth_iters=MATERIAL_SMOOTH_ITERS,
        min_component_voxels=MIN_COMPONENT_VOXELS,
    )


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
def _build_instance_to_material(per_material_n_instances: list[int]) -> dict[int, int]:
    """Map instance_id (1..N) -> material channel index (0..C-1).

    `build_material_split_labels` assigns IDs sequentially per channel, so the
    first per_material[0] IDs are channel 0, the next per_material[1] are
    channel 1, etc. We rebuild that mapping for category lookup at annotation
    emission time.
    """
    out = {}
    next_id = 1
    for chan_idx, count in enumerate(per_material_n_instances):
        for inst_id in range(next_id, next_id + count):
            out[inst_id] = chan_idx
        next_id += count
    return out


def _build_rich_categories(
    instance_to_chan: dict[int, int],
) -> tuple[list[dict], dict[int, list[int]]]:
    """Build category list and per-instance category-id list for rich-prompt
    mode. Each XRF channel index (0=Ge, 1=Te, 2=Ti, 3=W) gets the union of its
    channel-specific aliases plus the generic prompts. Categories are
    deduplicated across channels (e.g. "metal" applies to both Ti and W; both
    materials' instances will be annotated under it).
    """
    name_to_id: dict[str, int] = {}
    categories: list[dict] = []

    def _add(name: str) -> int:
        if name not in name_to_id:
            cat_id = len(categories) + 1
            name_to_id[name] = cat_id
            categories.append({
                'id': cat_id, 'name': name,
                'supercategory': 'integrated_circuit',
            })
        return name_to_id[name]

    # Ensure umbrella + generics are registered first so their IDs are stable.
    for name in RICH_PROMPTS_GENERIC:
        _add(name)

    chan_cat_ids: list[list[int]] = []
    for chan_idx in range(len(XRF_MATERIAL_NAMES)):
        ids = [_add(p) for p in RICH_PROMPTS_PER_CHANNEL[chan_idx]]
        ids += [name_to_id[g] for g in RICH_PROMPTS_GENERIC]
        # Dedupe while preserving order.
        seen = set()
        ids = [i for i in ids if not (i in seen or seen.add(i))]
        chan_cat_ids.append(ids)

    instance_cat_ids: dict[int, list[int]] = {
        inst_id: chan_cat_ids[chan_idx]
        for inst_id, chan_idx in instance_to_chan.items()
    }
    return categories, instance_cat_ids


def export(
    out_dir: Path = OUT_DIR,
    per_material: bool = False,
    rich_prompts: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ptycho, aligned_xrf = load_volumes()
    print(f'ptycho  {ptycho.shape}  |  xrf stack  {aligned_xrf.shape}')

    labels3d, label_info = build_instance_labels(aligned_xrf, ptycho)
    n_instances = int(labels3d.max())
    assert n_instances <= 255, f'{n_instances} instances → switch mask dtype to uint16'
    print(
        f'3D instances: {n_instances}  (stable IDs across all slices, '
        f'material-split)  fg%={label_info["final_fg_pct"]:.2f}  '
        f'pty_thr={label_info["ptycho_threshold"]:.5f}  '
        f'smooth={label_info["material_smooth_iters"]}'
    )
    print('per-channel XRF (P, thr, fg%, n_inst):')
    per_mat = label_info["per_material_n_instances"]
    for name, p, thr, fg, nm in zip(
        XRF_CHANNELS,
        label_info["xrf_percentiles"],
        label_info["xrf_thresholds"],
        label_info["per_channel_fg_pct"],
        per_mat,
    ):
        print(f'  {name:14s}  P{p:>5.2f}  thr={thr:.4f}  fg={fg:.2f}%  n_inst={nm}')

    ptycho_u8 = norm_to_uint8(ptycho)

    # Categories. Three modes:
    #   default        — single "IC feature" umbrella (legacy).
    #   per_material   — umbrella + 4 element-specific categories. Each
    #                    instance double-annotated (umbrella + its material).
    #   rich_prompts   — rich alias map (per-element + class umbrellas + density
    #                    tier + generics). Each instance annotated under every
    #                    alias that applies to its XRF channel; sets up the
    #                    text encoder to handle paraphrased queries.
    if rich_prompts and per_material:
        raise ValueError("rich_prompts and per_material are mutually exclusive")

    chan_idx_to_cat_id: dict[int, int] = {}
    instance_to_cat_ids: dict[int, list[int]] = {}
    if rich_prompts:
        instance_to_chan = _build_instance_to_material(
            label_info['per_material_n_instances']
        )
        categories, instance_to_cat_ids = _build_rich_categories(instance_to_chan)
    elif per_material:
        categories = [{
            'id': 1, 'name': UMBRELLA_CATEGORY_NAME,
            'supercategory': 'integrated_circuit',
        }]
        for chan_idx, mat_name in enumerate(XRF_MATERIAL_NAMES):
            cat_id = chan_idx + 2
            chan_idx_to_cat_id[chan_idx] = cat_id
            categories.append({
                'id': cat_id,
                'name': f'{mat_name} grain',
                'supercategory': 'integrated_circuit',
            })
        instance_to_chan = _build_instance_to_material(
            label_info['per_material_n_instances']
        )
    else:
        categories = [{
            'id': 1, 'name': UMBRELLA_CATEGORY_NAME,
            'supercategory': 'integrated_circuit',
        }]
        instance_to_chan = {}

    coco = {
        'info': {
            'description': 'Ptycho-tomo + joint XRF∩ptycho mask, material-split (aligned)',
            'source_volume': REF_VOLUME,
            'xrf_channels': XRF_CHANNELS,
            'xrf_percentiles': list(MASK_PERCENTILES),
            'ptycho_method': PTYCHO_METHOD,
            'ptycho_percentile': PTYCHO_PERCENTILE,
            'material_smooth_iters': MATERIAL_SMOOTH_ITERS,
            'num_3d_instances': n_instances,
            'per_material_categories': per_material,
            'rich_prompts': rich_prompts,
        },
        'categories': categories,
        'images': [], 'annotations': [],
    }
    prompts = {'videos': {}}
    rng = np.random.default_rng(POINT_RNG_SEED)
    image_id = 0
    ann_id = 0

    for vid_name, axis in VIDEOS:
        vid_dir = out_dir / 'video' / vid_name
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
            # 3-slice RGB encoding: R = i-1, G = i, B = i+1 (edges clamped).
            # Gives the model 2.5D context in a single 2D forward pass.
            prev_i = max(0, i - 1)
            next_i = min(n_frames - 1, i + 1)
            r = np.take(ptycho_u8, prev_i, axis=axis)
            g = np.take(ptycho_u8, i, axis=axis)
            b = np.take(ptycho_u8, next_i, axis=axis)
            rgb = np.stack([r, g, b], axis=-1)
            mask_slice = np.take(labels3d, i, axis=axis).astype(np.uint8)

            fname = f'{i:05d}.jpg'
            mname = f'{i:05d}.png'
            Image.fromarray(rgb, 'RGB').save(
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
                # Emit one annotation per category this instance satisfies.
                if rich_prompts:
                    cat_ids = instance_to_cat_ids.get(int(inst_id), [1])
                elif per_material:
                    chan_idx = instance_to_chan.get(int(inst_id))
                    cat_ids = [1] + (
                        [chan_idx_to_cat_id[chan_idx]] if chan_idx is not None else []
                    )
                else:
                    cat_ids = [1]
                for cat_id in cat_ids:
                    coco['annotations'].append({
                        'id': ann_id, 'image_id': image_id, 'category_id': cat_id,
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

    (out_dir / 'coco').mkdir(exist_ok=True)
    with open(out_dir / 'coco' / 'annotations.json', 'w') as f:
        json.dump(coco, f)
    with open(out_dir / 'prompts.json', 'w') as f:
        json.dump(prompts, f)

    meta = {
        'total_images': image_id, 'total_annotations': ann_id,
        'num_3d_instances': n_instances,
        'video_sequences': [
            {'name': n, 'axis': a, 'n_frames': int(ptycho_u8.shape[a])} for n, a in VIDEOS
        ],
        'xrf_channels': XRF_CHANNELS,
        'mask_percentiles': list(MASK_PERCENTILES),
        'mask_closing_iters': MASK_CLOSING_ITERS,
        'min_component_voxels': MIN_COMPONENT_VOXELS,
        'ptycho_method': PTYCHO_METHOD,
        'ptycho_percentile': PTYCHO_PERCENTILE,
        'post_closing_iters': POST_CLOSING_ITERS,
        'material_smooth_iters': MATERIAL_SMOOTH_ITERS,
        'labeler': 'material_split',
        'label_info': label_info,
        'ptycho_shape_zyx': list(ptycho.shape),
        'normalization': {'lo_pct': NORM_LO_PCT, 'hi_pct': NORM_HI_PCT, 'output': 'uint8 RGB JPEG'},
        'frame_encoding': '3-slice RGB stack: R=ptycho[i-1], G=ptycho[i], B=ptycho[i+1] (edges clamped).',
        'mask_format': 'uint8 PNG; pixel value = 3D instance ID (0 = background)',
        'notes': [
            'Instance IDs are stable across frames within a video (same 3D component).',
            'IDs are NOT stable across videos (xy / xz / yz share underlying 3D labels '
            'but the instance set visible in each slice differs).',
            'Labels are XRF ∩ ptycho>P85 split by per-voxel XRF argmax — fused '
            'gate-plus-contact blobs become one instance per dominant element '
            '(W → gate metal; Ti / Ge / Te → contacts / chalcogenide / liner).',
        ],
    }
    meta['per_material_categories'] = per_material
    meta['rich_prompts'] = rich_prompts
    if per_material or rich_prompts:
        meta['category_names'] = [c['name'] for c in categories]
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\n{out_dir}/ written:')
    print(f'  images: {image_id}')
    print(f'  annotations: {ann_id}')
    print(f'  videos: {[n for n, _ in VIDEOS]}')
    print(f'  categories: {[c["name"] for c in categories]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, default=OUT_DIR,
                    help='where to write the SAM 3 export tree')
    ap.add_argument('--per-material', action='store_true',
                    help='emit per-material categories ("tungsten grain" etc.) '
                         'in addition to the umbrella "IC feature" class. '
                         'Each instance is double-annotated so a query for '
                         '"IC feature" still matches everything.')
    ap.add_argument('--rich-prompts', action='store_true',
                    help='emit the full rich-prompt alias map: per-element '
                         '+ class umbrellas (chalcogenide / metal) + density '
                         'tier + generic prompts. Each instance is annotated '
                         'under every alias that applies to its XRF channel. '
                         'Mutually exclusive with --per-material.')
    args = ap.parse_args()
    export(
        out_dir=args.out_dir,
        per_material=args.per_material,
        rich_prompts=args.rich_prompts,
    )
