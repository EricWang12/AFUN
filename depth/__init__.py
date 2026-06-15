"""Depth wrappers for the AFUN demo.

- `da3.generate_depth(rgb)`         → predicts depth + intrinsics from RGB alone
- `lingbot.refine_depth(rgb, d, K)` → refines noisy sensor depth using RGB + K
- `prepare_depth(rgb, depth_dir)`   → picks the right path: DA3 if no depth_dir,
                                      lingbot if depth_dir has depth.npy + cam_K.txt

Both underlying models load lazily on first call and are cached as singletons.
"""
from __future__ import annotations

import pathlib

import numpy as np

from .da3 import generate_depth
from .lingbot import refine_depth

__all__ = ["generate_depth", "refine_depth", "prepare_depth"]


def prepare_depth(rgb: np.ndarray, depth_dir=None, device: str = "cuda", refine: bool = True):
    """Resolve depth for one RGB image.

    Parameters
    ----------
    rgb : np.ndarray
        (H, W, 3) uint8 image.
    depth_dir : str | pathlib.Path | None
        - ``None``  → Depth-Anything-3 generates depth + intrinsics from the RGB.
        - a folder  → must contain ``depth.npy`` (mm) + ``cam_K.txt`` (3x3); the
          raw depth is refined with lingbot (unless ``refine=False``).
    device : str
    refine : bool
        Only relevant when *depth_dir* is given. ``True`` (default) refines the
        sensor depth with lingbot; ``False`` returns the raw sensor depth as-is
        (skips lingbot). Ignored on the DA3 path.

    Returns
    -------
    (depth_mm, K, source) : (np.ndarray (H,W), np.ndarray (3,3), str)
        ``source`` is ``"da3"``, ``"lingbot"``, or ``"raw"`` (depth_dir + refine=False).
    """
    if depth_dir is None:
        depth_mm, K = generate_depth(rgb, device=device)
        return depth_mm, K, "da3"

    depth_dir = pathlib.Path(depth_dir)
    depth_path = depth_dir / "depth.npy"
    cam_path = depth_dir / "cam_K.txt"
    if not depth_path.exists():
        raise FileNotFoundError(f"missing depth.npy in {depth_dir}")
    if not cam_path.exists():
        raise FileNotFoundError(f"missing cam_K.txt in {depth_dir}")
    raw = np.load(depth_path).astype(np.float32)
    K = np.loadtxt(cam_path).astype(np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"{cam_path}: expected (3, 3) intrinsic, got {K.shape}")
    if not refine:
        return raw, K, "raw"
    depth_mm = refine_depth(rgb, raw, K, device=device)
    return depth_mm, K, "lingbot"
