"""AFUN RGB-D samples with matched functional masks and 3D curve targets."""
from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base import DatasetBase
from .sam3_utils import collate_to_sam3, _cc_boxes_and_segments_from_mask_bool
from src.models.decoder.motion_decoder.curves import fsparams_to_tensor

logger = logging.getLogger(__name__)

def _imread_rgb(path: pathlib.Path) -> np.ndarray:
    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            arr = np.array(rgb)
    except Exception as e:
        raise FileNotFoundError(f"Could not read image: {path}") from e
    return arr


def _imread_gray(path: pathlib.Path) -> np.ndarray:
    try:
        with Image.open(path) as img:
            g = img.convert("L")
            arr = np.array(g)
    except Exception as e:
        raise FileNotFoundError(f"Could not read mask: {path}") from e
    return arr


def _resize_mask_nearest(mask: np.ndarray, *, out_h: int, out_w: int) -> np.ndarray:
    """Nearest resize using OpenCV if available, else torch interpolate."""
    if mask.shape[0] == out_h and mask.shape[1] == out_w:
        return mask
    if cv2 is not None:
        return cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    m = torch.from_numpy(mask).float()[None, None]
    m2 = F.interpolate(m, size=(out_h, out_w), mode="nearest")
    return m2[0, 0].cpu().numpy()


def _pixel_to_3d(
    depth: np.ndarray,
    uv,
    intrinsic: Dict[str, float],
):
    """Project a 2D pixel to 3D using a depth map and pinhole intrinsics.

    Returns None when the pixel is out of bounds, depth is missing / non-positive,
    or required intrinsic fields are absent.
    """
    try:
        u, v = float(uv[0]), float(uv[1])
    except (TypeError, IndexError, ValueError):
        return None
    H, W = int(depth.shape[0]), int(depth.shape[1])
    iu, iv_ = int(round(u)), int(round(v))
    if iu < 0 or iv_ < 0 or iu >= W or iv_ >= H:
        return None
    z = float(depth[iv_, iu])
    if not np.isfinite(z) or z <= 0.0:
        return None
    fx = float(intrinsic.get("fx", 0.0))
    fy = float(intrinsic.get("fy", 0.0))
    cx = float(intrinsic.get("ppx", intrinsic.get("cx", 0.0)))
    cy = float(intrinsic.get("ppy", intrinsic.get("cy", 0.0)))
    if fx <= 0.0 or fy <= 0.0:
        return None
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


def _segment_centroid_2d(seg_bool_hw: torch.Tensor):
    """Return (cx, cy) of the True region. Assumes at least one True pixel."""
    ys, xs = torch.where(seg_bool_hw)
    return float(xs.float().mean().item()), float(ys.float().mean().item())


def _match_segments_to_trajectories(
    segs: List[torch.Tensor],
    trajectories: List[Dict[str, Any]],
    depth: np.ndarray,
    head_intrinsic: Optional[Dict[str, float]],
    H: int,
    W: int,
    *,
    lam: float = 0.5,
    abort_frac: Optional[float] = 0.4,
    scene_scale: float = 1.0,
):
    """Match segments to trajectories via Hungarian assignment.

    Primary signal: start_pixel is inside a segment (cost 0). Fallback:
    normalized 2D pixel + 3D world distance to segment centroid.

    Returns:
        (match_order, reject_reason). match_order[t] = segment index matched
        to trajectory t. If rejected, returns (None, reason).
    """
    from scipy.optimize import linear_sum_assignment

    K = len(segs)
    T = len(trajectories)
    if K != T or K == 0:
        return None, "count_mismatch"

    diag = float(np.hypot(H, W))
    centroids_2d = [_segment_centroid_2d(s) for s in segs]
    if head_intrinsic is not None:
        centroids_3d = [_pixel_to_3d(depth, (cx, cy), head_intrinsic) for cx, cy in centroids_2d]
    else:
        centroids_3d = [None] * K

    cost = np.zeros((K, T), dtype=np.float64)
    for i, seg in enumerate(segs):
        cx_i, cy_i = centroids_2d[i]
        c3d_i = centroids_3d[i]
        for t, traj in enumerate(trajectories):
            start_px = (traj.get("motion_2d") or {}).get("start_pixel")
            if not start_px or len(start_px) < 2:
                cost[i, t] = 1e6
                continue
            u, v = float(start_px[0]), float(start_px[1])
            iu, iv = int(round(u)), int(round(v))
            if 0 <= iu < W and 0 <= iv < H and bool(seg[iv, iu].item()):
                cost[i, t] = 0.0
                continue
            d2d = float(np.hypot(u - cx_i, v - cy_i)) / (diag + 1e-9)
            start_3d = None
            traj_3d = traj.get("trajectory_3d") or []
            if traj_3d and "position_3d" in traj_3d[0]:
                start_3d = np.asarray(traj_3d[0]["position_3d"], dtype=np.float32)
            if start_3d is None:
                md3 = traj.get("motion_3d")
                if md3 is not None and len(md3) == 3:
                    start_3d = np.asarray(md3, dtype=np.float32)
            if start_3d is not None and c3d_i is not None:
                d3d = float(np.linalg.norm(start_3d - c3d_i)) / (scene_scale + 1e-9)
                cost[i, t] = lam * d2d + (1.0 - lam) * d3d
            else:
                cost[i, t] = d2d

    row_ind, col_ind = linear_sum_assignment(cost)
    order_by_traj = [0] * T
    for r, c in zip(row_ind, col_ind):
        order_by_traj[int(c)] = int(r)

    if abort_frac is not None and abort_frac > 0.0:
        for t in range(T):
            start_px = (trajectories[t].get("motion_2d") or {}).get("start_pixel")
            if not start_px or len(start_px) < 2:
                return None, "bad_geometry"
            u, v = float(start_px[0]), float(start_px[1])
            iu, iv = int(round(u)), int(round(v))
            inside_any = False
            for seg in segs:
                if 0 <= iu < W and 0 <= iv < H and bool(seg[iv, iu].item()):
                    inside_any = True
                    break
            if inside_any:
                continue
            d_min = min(float(np.hypot(u - cx_i, v - cy_i)) for cx_i, cy_i in centroids_2d)
            if d_min > float(abort_frac) * diag:
                return None, "bad_geometry"

    return order_by_traj, None


def _load_trajectories_list(traj_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a list of per-instance trajectory dicts.

    - New format (non-empty ``trajectories`` key): return as-is.
    - Old format (no key, or empty list): wrap top-level singular motion
      fields into a one-element list so legacy samples become K'=1.
    """
    if traj_data.get("trajectories"):
        return list(traj_data["trajectories"])
    legacy_keys = (
        "motion_3d",
        "motion_2d",
        "trajectory_3d",
        "trajectory_2d",
        "fsparams",
        "spline_params",
        "opd_params",
    )
    return [{k: traj_data[k] for k in legacy_keys if k in traj_data}]


def _motion_from_traj_dict(
    t: Dict[str, Any],
    *,
    motion_type: str,
    dtype: torch.dtype,
    curve_T: int,
) -> Optional[torch.Tensor]:
    """Per-instance variant of RobotDataset._load_motion."""
    if motion_type == "motion_3d":
        vec = t.get("motion_3d") or t.get("normalized_motion_vector")
        if vec is None:
            return None
        return torch.as_tensor(vec, dtype=dtype).contiguous()

    if motion_type == "fsparams":
        fs = t.get("fsparams")
        if fs is None:
            return None
        return fsparams_to_tensor(fs, dtype=dtype)

    if motion_type == "curve_samples":
        from src.models.decoder.motion_decoder.curves import (
            sample_fs_curve,
            fsparams_to_tensor as _fsparams_to_tensor,
        )
        fs = t.get("fsparams")
        if fs is not None:
            fs_t = _fsparams_to_tensor(fs, dtype=torch.float64)
            pts = sample_fs_curve(fs_t, int(curve_T))
            return pts.to(dtype=dtype).contiguous()
        spline = t.get("spline_params")
        if spline is not None and "ctrl" in spline:
            from src.models.decoder.motion_decoder.spline_processor import _binomial_coeffs
            ctrl = torch.as_tensor(spline["ctrl"], dtype=torch.float64)
            if ctrl.ndim == 1:
                ctrl = ctrl.reshape(ctrl.numel() // 3, 3)
            K = ctrl.shape[0]
            origin = torch.zeros(1, 3, dtype=ctrl.dtype)
            abs_ctrl = torch.cat([origin, ctrl], dim=0)
            ts = torch.linspace(0.0, 1.0, int(curve_T), dtype=torch.float64)
            ks = torch.arange(K + 1, dtype=torch.float64)
            binom = _binomial_coeffs(K, dtype=torch.float64)
            log_t = torch.log(ts.clamp_min(1e-30))
            log_um = torch.log((1.0 - ts).clamp_min(1e-30))
            log_B = binom.log()[:, None] + ks[:, None] * log_t[None, :] + (K - ks)[:, None] * log_um[None, :]
            B = log_B.exp()
            B[:, 0] = 0.0; B[0, 0] = 1.0
            B[:, -1] = 0.0; B[-1, -1] = 1.0
            pts = (B.T @ abs_ctrl)
            return pts.to(dtype=dtype).contiguous()
        trajectory = t.get("trajectory_3d") or []
        if len(trajectory) >= 2:
            points = np.asarray([point["position_3d"] for point in trajectory], dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise ValueError("Invalid camera-frame trajectory.")
            times = np.linspace(0, 1, len(points))
            sampled = np.stack([np.interp(np.linspace(0, 1, curve_T), times, points[:, dim]) for dim in range(3)], axis=1)
            return torch.as_tensor(sampled - sampled[0], dtype=dtype)
        return None

    if motion_type == "opd":
        opd = t.get("opd_params")
        if opd is None:
            return None
        return torch.as_tensor(
            [float(opd["type"])] + list(opd["axis"]) + list(opd["origin"]),
            dtype=dtype,
        ).contiguous()

    sub = t.get(motion_type)
    if isinstance(sub, dict) and "ctrl" in sub:
        return torch.as_tensor(sub["ctrl"], dtype=dtype).contiguous()
    return None


def _traj_data_from_dict(t: Dict[str, Any]) -> Dict[str, Any]:
    """Per-instance equivalent of the traj_data dict from _load_cam_traj_data."""
    trajectory_3d = t.get("trajectory_3d") or []
    trajectory_2d = t.get("trajectory_2d") or []
    positions_3d = [item["position_3d"] for item in trajectory_3d if "position_3d" in item]
    positions_2d = [item["position_pixel"] for item in trajectory_2d if "position_pixel" in item]
    frame_indices = [item["frame_idx"] for item in trajectory_3d if "frame_idx" in item]
    velocities_3d: List[List[float]] = []
    if len(positions_3d) >= 2:
        arr = np.asarray(positions_3d, dtype=np.float32)
        velocities_3d = (arr[1:] - arr[:-1]).tolist()
    return {
        "positions_3d": positions_3d,
        "positions_2d": positions_2d,
        "velocities_3d": velocities_3d,
        "frame_indices": frame_indices,
    }


class UnmatchedSample(ValueError):
    """The sample's mask components cannot be paired with its trajectories."""


class RobotDataset(DatasetBase):
    """Read the public release directly from its manifest; depth is in millimeters."""

    def __init__(self, data_dir, *, split="train", subset=None, dataset_cfg=None,
                 loader_cfg=None, strict=True, cache=False, split_file=None,
                 max_samples=None, **kwargs):
        super().__init__(data_dir=data_dir, split=split, dtype=torch.float32)
        self.dataset_cfg = dataset_cfg or {}
        self.loader_cfg = loader_cfg
        self.strict = bool(strict)
        manifest = self.data_dir / "manifest.json"
        with manifest.open() as handle:
            records = json.load(handle)["samples"]
        if subset:
            sources = {subset} if isinstance(subset, str) else set(subset)
            records = [r for r in records if r["path"].split("/")[0] in sources]
        if split_file:
            with pathlib.Path(split_file).open() as handle:
                splits = json.load(handle)
            held_out = set(splits.get("test", [])) | set(splits.get("val", []))
            records = [r for r in records if (r["path"] not in held_out if split == "train"
                                             else r["path"] in splits[split])]
        elif split not in ("train", "all", ""):
            raise ValueError("For evaluation use src.eval and AFUN_eval; AFUN_train contains training samples only.")
        self.samples = records[:max_samples] if max_samples else records
        if not self.samples:
            raise ValueError(f"No AFUN samples found in {self.data_dir}.")
        for record in self.samples:
            path = pathlib.PurePosixPath(record["path"])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Invalid manifest path: {path}")
        logger.info("AFUN: %d samples from %s", len(self), self.data_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # A bounded fallback is available for legacy imperfect data, never for evaluation.
        for attempt in range(min(32, len(self))):
            current = (index + attempt) % len(self)
            try:
                return self._load_sample(current)
            except UnmatchedSample as error:
                # 94 of the 44,749 released samples (0.2%).
                logger.debug("Skipping %s: %s", self.samples[current]["path"], error)
            except (OSError, ValueError, KeyError) as error:
                if self.strict:
                    raise ValueError(f"Invalid AFUN sample {self.samples[current]['path']}: {error}") from error
                logger.warning("Skipping %s: %s", self.samples[current]["path"], error)
        raise ValueError(f"No valid AFUN sample in {min(32, len(self))} attempts.")

    def _load_sample(self, index):
        record = self.samples[index]
        directory = self.data_dir / record["path"]
        image = _imread_rgb(directory / "obs_frame.png").astype(np.float32) / 255
        depth = np.load(directory / "obs_frame_depth.npy", allow_pickle=False).astype(np.float32)
        mask = _imread_gray(directory / "sam_mask.png") > 0
        if depth.shape != image.shape[:2] or mask.shape != image.shape[:2]:
            raise ValueError("RGB, depth, and mask resolutions must match.")
        with (directory / self.dataset_cfg.get("traj_file_name", "trajectory.json")).open() as handle:
            raw = json.load(handle)
        trajectories = _load_trajectories_list(raw)
        intr = raw["camera_info"]["intrinsics"]
        intr = {"fx": float(intr["fx"]), "fy": float(intr["fy"]),
                "cx": float(intr.get("cx", intr.get("ppx", 0))),
                "cy": float(intr.get("cy", intr.get("ppy", 0)))}
        if min(intr["fx"], intr["fy"]) <= 0:
            raise ValueError("Camera focal lengths must be positive.")
        # Components below min_area count only when the size filter leaves the sample unpaired
        # (e.g. a small target), as such samples were part of AFUN training.
        for min_area in (int(self.dataset_cfg.get("min_area", 200)), 1):
            boxes, segments = _cc_boxes_and_segments_from_mask_bool(
                torch.from_numpy(mask), connectivity=8, min_area=min_area, sort_by_area_desc=False,
            )
            if len(trajectories) == 1 and segments:
                # A single interaction has no assignment ambiguity; retain its largest component.
                order = [max(range(len(segments)), key=lambda i: int(segments[i].sum()))]
                break
            order, reason = _match_segments_to_trajectories(
                segments, trajectories, depth * 0.001, intr, *depth.shape,
                lam=float(self.dataset_cfg.get("match_lambda_2d", 0.5)),
                abort_frac=self.dataset_cfg.get("match_cost_abort_frac_of_diag", 0.4),
            )
            if order is not None:
                break
        else:
            raise UnmatchedSample(f"Cannot match masks to trajectories: {reason}")
        segments, boxes = [segments[i] for i in order], [boxes[i] for i in order]
        motion = [_motion_from_traj_dict(t, motion_type=self.dataset_cfg.get("motion_type", "curve_samples"),
                                        dtype=self.dtype, curve_T=int(self.dataset_cfg.get("curve_T", 16)))
                  for t in trajectories]
        if not motion or any(m is None or not torch.isfinite(m).all() for m in motion):
            raise ValueError("Missing or non-finite motion targets.")
        queries = record.get("queries") or [record.get("language", "")]
        if isinstance(queries, str):
            queries = [queries]
        queries = [q for q in queries if isinstance(q, str) and q.strip()]
        if not queries:
            raise ValueError("Missing language instruction.")
        return {
            "image": torch.from_numpy(image).permute(2, 0, 1).contiguous(),
            "mask": torch.stack(segments).any(0).float()[None],
            "depth": torch.from_numpy(depth)[None], "query": queries,
            "motion_directions": motion, "motion_2d": [t.get("motion_2d") for t in trajectories],
            "traj_data": [_traj_data_from_dict(t) for t in trajectories],
            "head_intrinsic": {"fx": intr["fx"], "fy": intr["fy"], "ppx": intr["cx"], "ppy": intr["cy"]},
            "data_path": str(directory), "precomputed_segs": segments, "precomputed_boxes": boxes,
        }

    @staticmethod
    def collate_function(batch, **kwargs):
        result = collate_to_sam3(batch, **kwargs)
        result["has_motion"] = True
        return result
