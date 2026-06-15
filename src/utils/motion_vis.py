# src/utils/motion_vis.py
from __future__ import annotations

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import torch
from PIL import Image
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def get_mask_centroid(mask_uint8: np.ndarray) -> Tuple[int, int]:
    """
    mask_uint8: (H,W) uint8 where foreground > 0
    Returns (cx, cy) in pixel coordinates.
    """
    ys, xs = np.nonzero(mask_uint8 > 0)
    if len(xs) == 0:
        h, w = mask_uint8.shape
        return w // 2, h // 2
    # Simple centroid (no cv2 required)
    return int(xs.mean()), int(ys.mean())


def project_3d_to_2d(point_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    x, y, z = point_3d
    if z <= 0:
        return np.array([np.nan, np.nan])
    u = K[0, 0] * (x / z) + K[0, 2]
    v = K[1, 1] * (y / z) + K[1, 2]
    return np.array([u, v])


def compute_velocity_direction_from_3d(
    positions_3d: List[List[float]],
    intrinsic: Dict[str, Any],
    num_frames: int = 20,
) -> Tuple[Tuple[float, float], np.ndarray]:
    scale_x = 640.0 / (intrinsic["ppx"] * 2)
    scale_y = 480.0 / (intrinsic["ppy"] * 2)
    K = np.array(
        [
            [intrinsic["fx"] * scale_x, 0, 320.0],
            [0, intrinsic["fy"] * scale_y, 240.0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    if len(positions_3d) < 1:
        return (1.0, 0.0), np.array([1.0, 0.0, 0.0], dtype=np.float64)

    end_idx = min(len(positions_3d) - 1, num_frames - 1)

    start_pos = np.array(positions_3d[0], dtype=np.float64)
    end_pos = np.array(positions_3d[end_idx], dtype=np.float64)
    displacement_3d = end_pos - start_pos

    norm_3d = np.linalg.norm(displacement_3d)
    if norm_3d > 0:
        v3 = displacement_3d / norm_3d
    else:
        v3 = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    origin_3d = start_pos
    offset_distance = float(origin_3d[2]) * 0.1
    end_3d = origin_3d + v3 * offset_distance

    origin_2d = project_3d_to_2d(origin_3d, K)
    end_2d = project_3d_to_2d(end_3d, K)
    if np.isnan(origin_2d).any() or np.isnan(end_2d).any():
        return (1.0, 0.0), v3

    direction_2d = end_2d - origin_2d
    norm_2d = np.linalg.norm(direction_2d)
    if norm_2d > 0:
        direction_2d = direction_2d / norm_2d
    else:
        direction_2d = np.array([1.0, 0.0], dtype=np.float64)

    return (float(direction_2d[0]), float(direction_2d[1])), v3


def visualize_trajectory_on_mask(
    image_tensor: torch.Tensor | Image.Image,
    mask_tensor: torch.Tensor,
    motion_direction_tensor: torch.Tensor,
    traj_data: Optional[dict],
    head_intrinsic: dict,
    num_frames: int = 60,
    arrow_length: int = 80,
    figsize: Optional[tuple] = None,
):
    from matplotlib import pyplot as plt

    if cv2 is None:
        raise ImportError("cv2 is required for visualize_trajectory_on_mask (pip install opencv-python).")


    if isinstance(image_tensor, torch.Tensor):
        image_np = image_tensor.detach().cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))
        image_np = np.clip(image_np, 0, 1)
    else:
        image_np = np.array(image_tensor)

    mask_np = mask_tensor.detach().cpu().numpy()
    if mask_np.ndim == 2:
        mask_np = mask_np[None, ...]

    image_uint8 = (image_np * 255).astype(np.uint8) if image_np.max() <= 1 else image_np.astype(np.uint8)
    mask_uint8 = (mask_np * 255).astype(np.uint8) if mask_np.max() <= 1 else mask_np.astype(np.uint8)

    H, W = image_uint8.shape[:2]
    if figsize is None:
        fig_width = 7
        fig_height = fig_width * (H / W)
        figsize = (fig_width, fig_height)

    vis_image = image_uint8.copy()

    positions_2d = traj_data.get("positions_2d", []) if traj_data is not None else []
    base_colors = [
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
        (255, 255, 0),  # cyan
        (0, 128, 255),  # orange
    ]

    for obj_idx in range(mask_uint8.shape[0]):
        mask_i = mask_uint8[obj_idx]
        cx, cy = get_mask_centroid(mask_i)

        # mask overlay
        mask_colored = np.zeros_like(vis_image)
        mask_colored[mask_i > 0] = base_colors[obj_idx % len(base_colors)]
        vis_image = cv2.addWeighted(vis_image, 1.0, mask_colored, 0.3, 0)

        # velocity arrow
        if motion_direction_tensor.ndim == 1:
            dx, dy, _ = motion_direction_tensor
        else:
            dx, dy, _ = motion_direction_tensor[min(obj_idx, motion_direction_tensor.shape[0] - 1)]
        ex = int(cx + float(dx) * arrow_length)
        ey = int(cy + float(dy) * arrow_length)
        cv2.arrowedLine(vis_image, (cx, cy), (ex, ey), (0, 0, 255), 3, tipLength=0.3)

    # trajectory (shift to centroid) - draw once if available
    if len(positions_2d) > 0 and mask_uint8.shape[0] > 0:
        cx0, cy0 = get_mask_centroid(mask_uint8[0])
        traj_start = np.array(positions_2d[0])
        shift = np.array([cx0, cy0]) - traj_start
        shifted = [np.array(p) + shift for p in positions_2d]

        for i in range(len(shifted) - 1):
            pt1 = tuple(shifted[i].astype(int))
            pt2 = tuple(shifted[i + 1].astype(int))
            progress = i / max(len(shifted) - 1, 1)
            color = (int(255 * progress), int(255 * (1 - progress)), 0)
            cv2.line(vis_image, pt1, pt2, color, 2)

        cv2.circle(vis_image, tuple(shifted[0].astype(int)), 6, (0, 255, 0), -1)
        cv2.circle(vis_image, tuple(shifted[-1].astype(int)), 6, (255, 0, 0), -1)

    # show
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    return vis_image
