from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Union, get_args, get_origin

import torch
import torch.nn.functional as F
from src.models.decoder.sam3.sam3.model.data_misc import BatchedFindTarget

import random

# -----------------------------
# Copied patterns from SAM3
# -----------------------------
MyTensor = Union[torch.Tensor, List[Any]]


def convert_my_tensors(obj):
    def is_optional_field(field) -> bool:
        return get_origin(field) is Union and type(None) in get_args(field)

    for field in fields(obj):
        v = getattr(obj, field.name)

        if is_dataclass(v):
            convert_my_tensors(v)
            continue

        field_type = field.type
        if is_optional_field(field.type):
            field_type = Union[get_args(field.type)[:-1]]

        if field_type != MyTensor or v is None:
            continue

        # list[Tensor] -> stack, else -> as_tensor
        if len(v) and isinstance(v[0], torch.Tensor):
            stack_dim = 0
            if field.name in ["input_boxes", "input_boxes_label"]:
                stack_dim = 1
            setattr(obj, field.name, torch.stack(v, dim=stack_dim).to(getattr(obj, field.name + "__type")))
        else:
            setattr(obj, field.name, torch.as_tensor(v, dtype=getattr(obj, field.name + "__type")))
    return obj


def packed_to_padded_naive(boxes_packed, num_boxes, fill_value=0):
    B = num_boxes.shape[0]
    Ns = num_boxes.tolist()
    boxes_padded = boxes_packed.new_zeros(B, max(Ns) if len(Ns) else 0, *boxes_packed.shape[1:])
    if fill_value != 0:
        boxes_padded[...] = fill_value
    prev_idx = 0
    for i in range(B):
        next_idx = prev_idx + Ns[i]
        boxes_padded[i, : Ns[i]] = boxes_packed[prev_idx:next_idx]
        prev_idx = next_idx
    return boxes_padded


def pad_tensor_list_to_longest(tensors: List[torch.Tensor], dim=0, pad_val=0) -> List[torch.Tensor]:
    if not tensors:
        return tensors
    pad_len = max(t.shape[dim] for t in tensors)
    for i in range(len(tensors)):
        n_dims = len(tensors[i].shape)
        n_right_dims = (n_dims - 1) - (n_dims + dim) % n_dims
        n_pad = pad_len - tensors[i].shape[dim]
        pad_tuple = tuple([0] * 2 * n_right_dims + [0, n_pad])
        tensors[i] = F.pad(tensors[i], pad_tuple, value=pad_val)
    return tensors


# -----------------------------
# InstructPart -> SAM3 helpers
# -----------------------------
def _pad_to_hw(x: torch.Tensor, out_h: int, out_w: int, *, pad_val: float = 0.0) -> torch.Tensor:
    # x: (...,H,W)
    H, W = int(x.shape[-2]), int(x.shape[-1])
    pad_h = out_h - H
    pad_w = out_w - W
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), value=pad_val)


def _resize_image_and_mask_square(
    img_chw: torch.Tensor,
    mask_1hw: torch.Tensor,
    *,
    resolution: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Resize (warp) image and mask to a fixed square resolution.

    SAM3's default preprocessing resizes images to (resolution, resolution) (no aspect preservation),
    and masks should follow the same geometric transform.
    """
    if int(img_chw.shape[-2]) == int(resolution) and int(img_chw.shape[-1]) == int(resolution):
        return img_chw, mask_1hw
    img_out = F.interpolate(
        img_chw.unsqueeze(0),
        size=(int(resolution), int(resolution)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    mask_out = F.interpolate(
        mask_1hw.unsqueeze(0),
        size=(int(resolution), int(resolution)),
        mode="nearest",
    ).squeeze(0)
    return img_out, mask_out


def _resize_seg_nearest(seg_bool_hw: torch.Tensor, resolution: int) -> torch.Tensor:
    """Nearest-neighbor resize of a (H, W) bool segment to (resolution, resolution)."""
    H, W = int(seg_bool_hw.shape[-2]), int(seg_bool_hw.shape[-1])
    if H == int(resolution) and W == int(resolution):
        return seg_bool_hw
    seg_float = seg_bool_hw.to(torch.float32).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(seg_float, size=(int(resolution), int(resolution)), mode="nearest")
    return (resized[0, 0] > 0.5).to(torch.bool)


def _normalize_image_chw(
    img_chw: torch.Tensor,
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    """Channel-wise normalize an image in CHW format."""
    mean_t = img_chw.new_tensor(mean).view(3, 1, 1)
    std_t = img_chw.new_tensor(std).view(3, 1, 1)
    return (img_chw - mean_t) / std_t


def _cc_boxes_and_segments_from_mask_bool(
    mask_bool_hw: torch.Tensor,
    *,
    connectivity: int = 8,
    min_area: int = 1,
    sort_by_area_desc: bool = True,
):
    """
    Returns:
      boxes_list:    list[Tensor(4,)] in cxcywh normalized to (H,W)
      segments_list: list[Tensor(H,W)] bool per component
    Notes:
      - Uses SciPy connected components if available; falls back to single-component box/mask.
      - Expects mask_bool_hw is on CPU (recommended for dataloader workers).
    """
    assert mask_bool_hw.ndim == 2
    H, W = map(int, mask_bool_hw.shape)

    if mask_bool_hw.sum().item() == 0:
        return [], []

    # Try SciPy CC
    try:
        import numpy as np
        from scipy.ndimage import label as cc_label
    except Exception:
        # Fallback: single component = whole mask
        ys, xs = torch.where(mask_bool_hw)
        y0, y1 = ys.min().float(), (ys.max() + 1).float()
        x0, x1 = xs.min().float(), (xs.max() + 1).float()
        cx = (x0 + x1) * 0.5 / float(W)
        cy = (y0 + y1) * 0.5 / float(H)
        bw = (x1 - x0) / float(W)
        bh = (y1 - y0) / float(H)
        box = torch.stack([cx, cy, bw, bh]).to(dtype=torch.float32)
        return [box], [mask_bool_hw.to(torch.bool)]

    mask_np = mask_bool_hw.numpy().astype("uint8")
    if connectivity == 8:
        structure = np.ones((3, 3), dtype=np.uint8)
    elif connectivity == 4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    else:
        raise ValueError("connectivity must be 4 or 8")

    lbl_np, num = cc_label(mask_np, structure=structure)
    if num == 0:
        return [], []

    lbl = torch.from_numpy(lbl_np)  # (H,W) int
    coords = torch.nonzero(lbl > 0, as_tuple=False)  # (M,2) [y,x]
    lab = (lbl[coords[:, 0], coords[:, 1]] - 1).to(torch.long)  # 0..num-1

    areas = torch.bincount(lab, minlength=num)

    y0 = torch.full((num,), H, dtype=torch.long)
    x0 = torch.full((num,), W, dtype=torch.long)
    y1 = torch.zeros((num,), dtype=torch.long)
    x1 = torch.zeros((num,), dtype=torch.long)

    ys = coords[:, 0].to(torch.long)
    xs = coords[:, 1].to(torch.long)

    y0.scatter_reduce_(0, lab, ys, reduce="amin", include_self=True)
    x0.scatter_reduce_(0, lab, xs, reduce="amin", include_self=True)
    y1.scatter_reduce_(0, lab, ys + 1, reduce="amax", include_self=True)  # exclusive
    x1.scatter_reduce_(0, lab, xs + 1, reduce="amax", include_self=True)  # exclusive

    keep = areas >= int(min_area)
    if keep.sum().item() == 0:
        return [], []

    kept_ids = torch.nonzero(keep, as_tuple=False).flatten()  # component ids 0..num-1

    # sort by area desc if requested
    if sort_by_area_desc:
        order = torch.argsort(areas[kept_ids], descending=True)
        kept_ids = kept_ids[order]

    boxes_list: List[torch.Tensor] = []
    segs_list: List[torch.Tensor] = []
    for cid in kept_ids.tolist():
        yy0, xx0 = y0[cid].float(), x0[cid].float()
        yy1, xx1 = y1[cid].float(), x1[cid].float()

        cx = (xx0 + xx1) * 0.5 / float(W)
        cy = (yy0 + yy1) * 0.5 / float(H)
        bw = (xx1 - xx0) / float(W)
        bh = (yy1 - yy0) / float(H)
        boxes_list.append(torch.stack([cx, cy, bw, bh]).to(torch.float32))

        segs_list.append((lbl == (cid + 1)).to(torch.bool))

    return boxes_list, segs_list


def _cc_boxes_and_segments_from_mask_bool_single(
    mask_bool_hw: torch.Tensor,
    *,
    connectivity: int = 8,          # kept for API-compat; unused
    min_area: int = 1,              # kept for API-compat; unused
    sort_by_area_desc: bool = True, # kept for API-compat; unused
):
    """
    This function assumes only one object in the mask.
    Returns:
      boxes_list:    list[Tensor(4,)] in cxcywh normalized to (H,W)
      segments_list: list[Tensor(H,W)] bool (here: single mask)
    """
    assert mask_bool_hw.ndim == 2
    H, W = map(int, mask_bool_hw.shape)
    mask = mask_bool_hw.to(torch.bool)

    if not mask.any().item():
        return [], []

    ys, xs = torch.where(mask)
    y0, y1 = ys.min().float(), (ys.max() + 1).float()  # exclusive
    x0, x1 = xs.min().float(), (xs.max() + 1).float()  # exclusive

    cx = (x0 + x1) * 0.5 / float(W)
    cy = (y0 + y1) * 0.5 / float(H)
    bw = (x1 - x0) / float(W)
    bh = (y1 - y0) / float(H)

    box = torch.stack([cx, cy, bw, bh]).to(torch.float32)
    return [box], [mask]

# -----------------------------
# The collate function you want
# -----------------------------
def _preprocess_one_sample(
    item: Dict[str, Any],
    *,
    image_resolution: int | None,
    normalize_image: bool,
    normalize_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    normalize_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    connectivity: int,
    min_area: int,
    sort_by_area_desc: bool,
    only_one_object: bool,
) -> Dict[str, Any]:
    """Per-sample preprocessing: resize, normalize, connected components."""
    img = item["image"]
    msk = item["mask"]
    H, W = int(img.shape[1]), int(img.shape[2])

    if img.dtype != torch.float32 and img.dtype != torch.float16 and img.dtype != torch.bfloat16:
        img = img.float()
    if msk.dtype != torch.float32 and msk.dtype != torch.float16 and msk.dtype != torch.bfloat16:
        msk = msk.float()

    img_unnormalized = None
    if image_resolution is not None:
        img, msk = _resize_image_and_mask_square(img, msk, resolution=int(image_resolution))
        img_unnormalized = img

    if normalize_image:
        img = _normalize_image_chw(img, mean=normalize_mean, std=normalize_std)

    # connected components on resized mask
    gt_bool_hw = (msk[0] > 0.5).to(torch.bool)
    if gt_bool_hw.is_cuda:
        gt_bool_hw = gt_bool_hw.cpu()

    if "precomputed_segs" in item and "precomputed_boxes" in item:
        precomp_segs = item["precomputed_segs"]
        precomp_boxes = item["precomputed_boxes"]
        assert len(precomp_segs) == len(precomp_boxes), (
            f"precomputed_segs (len {len(precomp_segs)}) and precomputed_boxes "
            f"(len {len(precomp_boxes)}) must have the same length"
        )
        if image_resolution is not None:
            segs_list = [_resize_seg_nearest(s, int(image_resolution)) for s in precomp_segs]
        else:
            segs_list = [s.to(torch.bool).cpu() for s in precomp_segs]
        boxes_list = [b.to(torch.float32) for b in precomp_boxes]
    elif only_one_object:
        boxes_list, segs_list = _cc_boxes_and_segments_from_mask_bool_single(
            gt_bool_hw, connectivity=connectivity, min_area=min_area,
            sort_by_area_desc=sort_by_area_desc,
        )
    else:
        boxes_list, segs_list = _cc_boxes_and_segments_from_mask_bool(
            gt_bool_hw, connectivity=connectivity, min_area=min_area,
            sort_by_area_desc=sort_by_area_desc,
        )

    result = {
        "proc_img": img,
        "proc_mask": msk,
        "img_unnormalized": img_unnormalized,
        "original_size": [H, W],
        "gt_bool_hw": gt_bool_hw,
        "boxes_list": boxes_list,
        "segs_list": segs_list,
        "image": item["image"],
        "query": item["query"],
    }
    for k in ("motion_directions", "depth", "traj_data", "head_intrinsic", "data_path", "motion_2d"):
        if k in item:
            result[k] = item[k]
    return result


def collate_to_sam3(
    batch: List[Dict[str, Any]],
    *,

    load_image_in_fp16: bool = False,
    # image preprocessing (match SAM3 defaults)
    image_resolution: int | None = 1008,
    normalize_image: bool = False,
    normalize_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    normalize_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    # CC params
    connectivity: int = 8,
    min_area: int = 1,
    sort_by_area_desc: bool = True,
    # labels
    is_exhaustive: bool = True,

    only_one_object: bool = False,
) -> Dict[str, Any]:
    """
    Batch (from your InstructPartDataset) item keys:
      - image: (3,H,W) float32 [0,1]
      - mask:  (1,H,W) float32 [0,1]
      - instructions: list[str]  (ignored for now; dummy text is used)

    Notes:
      - SAM3's reference preprocessing resizes to a fixed square `image_resolution` (default 1008)
        and normalizes with mean/std (default 0.5/0.5) -> roughly [-1, 1].
      - Set `image_resolution=None` to keep original sizes; set `normalize_image=False` to disable.
      - If samples are already pre-processed (contain "proc_img" key from
        _preprocess_one_sample called in __getitem__), the per-sample work is skipped.

    Returns:
      {dict_key: BatchedDatapoint(...)} matching SAM3 collate structure.
    """
    assert len(batch) > 0

    # --- per-sample preprocessing: use pre-computed results or run here ---
    already_preprocessed = "proc_img" in batch[0]

    if already_preprocessed:
        preprocessed = batch
    else:
        preprocessed = [
            _preprocess_one_sample(
                item,
                image_resolution=image_resolution,
                normalize_image=normalize_image,
                normalize_mean=normalize_mean,
                normalize_std=normalize_std,
                connectivity=connectivity,
                min_area=min_area,
                sort_by_area_desc=sort_by_area_desc,
                only_one_object=only_one_object,
            )
            for item in batch
        ]

    # --- pad images to same (Hmax,Wmax) and stack ---
    proc_imgs = [p["proc_img"] for p in preprocessed]
    img_unnormalized = [p["img_unnormalized"] for p in preprocessed if p["img_unnormalized"] is not None]

    max_H = max(int(x.shape[1]) for x in proc_imgs)
    max_W = max(int(x.shape[2]) for x in proc_imgs)
    image_batch = torch.stack([_pad_to_hw(img, max_H, max_W, pad_val=0.0) for img in proc_imgs], dim=0)  # (B,3,Hmax,Wmax)
    if load_image_in_fp16:
        image_batch = image_batch.half()


    # --- assemble BatchedFindTarget from pre-computed CC results ---
    find_targets = BatchedFindTarget(
            num_boxes=[],
            boxes=[],
            boxes_padded=[],
            is_exhaustive=[],
            segments=[],
            semantic_segments=[],
            is_valid_segment=[],
            repeated_boxes=[],
            object_ids=[],
            object_ids_padded=[],
            motion_directions=[],
        )

    for p in preprocessed:
        K = len(p["boxes_list"])
        find_targets.num_boxes.append(K)
        find_targets.is_exhaustive.append(bool(is_exhaustive))
        find_targets.boxes.extend(p["boxes_list"])
        find_targets.object_ids.extend(list(range(K)))
        find_targets.segments.extend(p["segs_list"])
        find_targets.is_valid_segment.extend([1] * K)
        find_targets.semantic_segments.append(p["gt_bool_hw"].to(torch.bool))

        if "motion_directions" in p and p["motion_directions"] is not None:
            md = p["motion_directions"]
            if torch.is_tensor(md):
                # Single pre-multi-instance tensor → treat as K=1 list.
                md = [md]
            # Per-instance: extend so the packed tensor aligns with boxes/segments.
            find_targets.motion_directions.extend(list(md))

    find_targets = convert_my_tensors(find_targets)
    find_targets.boxes_padded = packed_to_padded_naive(find_targets.boxes.view(-1, 4), find_targets.num_boxes)
    find_targets.object_ids_padded = packed_to_padded_naive(find_targets.object_ids, find_targets.num_boxes, fill_value=-1)

    result = {
        "image": image_batch,
        "image_unnormalized": img_unnormalized,
        "find_target":  find_targets,
        "query": [random.choice(item["query"]) for item in batch],  # list[list[str]]
        "original_sizes": [p["original_size"] for p in preprocessed],  # [[H, W], ...] before resize to 1008
    }

    # Batch depth: resize to fixed resolution, stack into (B, 1, H_d, W_d)
    DEPTH_RESOLUTION = (480, 640)
    if "depth" in preprocessed[0] and preprocessed[0]["depth"] is not None:
        depth_list = []
        for p in preprocessed:
            d = p["depth"]
            if not torch.is_tensor(d):
                d = torch.as_tensor(d, dtype=torch.float32)
            if d.ndim == 2:
                d = d.unsqueeze(0)  # (H, W) -> (1, H, W)
            d = F.interpolate(
                d.unsqueeze(0), size=DEPTH_RESOLUTION,
                mode="bilinear", align_corners=False,
            ).squeeze(0)
            depth_list.append(d)
        result["depth"] = torch.stack(depth_list, dim=0)  # (B, 1, 480, 640)

    # Batch intrinsics: convert dict -> (3,3) tensor, scale to DEPTH_RESOLUTION
    if "head_intrinsic" in preprocessed[0] and preprocessed[0]["head_intrinsic"] is not None:
        intrinsics_list = []
        for p in preprocessed:
            intr = p["head_intrinsic"]
            if isinstance(intr, dict):
                d_orig = p["depth"]
                H_d = d_orig.shape[-2] if torch.is_tensor(d_orig) else d_orig.shape[0]
                W_d = d_orig.shape[-1]
                scale_y = DEPTH_RESOLUTION[0] / H_d
                scale_x = DEPTH_RESOLUTION[1] / W_d
                K = torch.tensor([
                    [intr["fx"] * scale_x, 0, intr.get("ppx", intr.get("cx", 0)) * scale_x],
                    [0, intr["fy"] * scale_y, intr.get("ppy", intr.get("cy", 0)) * scale_y],
                    [0, 0, 1],
                ], dtype=torch.float32)
                intrinsics_list.append(K)
            elif torch.is_tensor(intr):
                intrinsics_list.append(intr)
        if intrinsics_list:
            result["head_intrinsic"] = torch.stack(intrinsics_list, dim=0)  # (B, 3, 3)

    # Remaining optional keys (kept as lists)
    for k in ("traj_data", "data_path", "motion_2d"):
        if k not in result and k in preprocessed[0] and preprocessed[0][k] is not None:
            result[k] = [p[k] for p in preprocessed]

    return result

