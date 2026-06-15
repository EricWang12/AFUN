"""Depth-Anything-3 wrapper: RGB image → metric depth (mm) + intrinsics.

Pure in-memory: takes an RGB array and returns depth + intrinsics (no file I/O).
Model is loaded lazily on first call and cached as a module-level singleton.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Tuple

import cv2
import numpy as np
import torch

# Vendored DA3 source tree (patched in-tree; see extern/README.md)
_DA3_ROOT = pathlib.Path(__file__).resolve().parent.parent / "extern" / "Depth-Anything-3"

# HuggingFace model id — weights cache to ~/.cache/huggingface/hub/ on first call
MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"

# Module-level singleton — one DA3 load per Python process
_DA3 = None


def _ensure_paths() -> None:
    """Add the vendored DA3 source tree to sys.path. Idempotent."""
    for p in (_DA3_ROOT, _DA3_ROOT / "src"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _load_model(device: torch.device):
    """Load DA3 once, cache, return."""
    global _DA3
    if _DA3 is not None:
        return _DA3
    _ensure_paths()
    try:
        from depth_anything_3.api import DepthAnything3  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            f"Depth-Anything-3 not importable from {_DA3_ROOT}. "
            "Check that extern/Depth-Anything-3/ is intact and the env "
            "has the required deps. See extern/README.md."
        ) from e
    print(f"[depth.da3] loading {MODEL_ID} on {device} (one-time; weights to ~/.cache/huggingface/hub/)")
    _DA3 = DepthAnything3.from_pretrained(MODEL_ID).to(device).eval()
    return _DA3


def generate_depth(
    rgb: np.ndarray,
    device: str | torch.device = "cuda",
    process_res: int = 504,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict metric depth + intrinsics from a single RGB image.

    Parameters
    ----------
    rgb : np.ndarray
        (H, W, 3) uint8 RGB image.
    device : str | torch.device
        Device for inference; falls back to CPU if CUDA is unavailable.
    process_res : int
        DA3's internal processing resolution (default 504).

    Returns
    -------
    depth_mm : np.ndarray
        (H, W) float32 metric depth in **millimetres**, image-resolution.
    K : np.ndarray
        (3, 3) float64 intrinsic matrix, scaled to image resolution.
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"rgb must be (H, W, 3) uint8; got shape={rgb.shape} dtype={rgb.dtype}"
        )
    H, W = rgb.shape[:2]

    device = torch.device(device) if not isinstance(device, torch.device) else device
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    model = _load_model(device)
    with torch.no_grad():
        pred = model.inference(image=[rgb], process_res=process_res)

    depth_m = np.asarray(pred.depth, dtype=np.float32)
    if depth_m.ndim == 3:
        depth_m = depth_m[0]
    K = np.asarray(pred.intrinsics, dtype=np.float64)
    if K.ndim == 3:
        K = K[0]

    # Upsample depth + scale K to image resolution
    if depth_m.shape != (H, W):
        sy = H / depth_m.shape[0]
        sx = W / depth_m.shape[1]
        depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_LINEAR)
        K = K.copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

    depth_mm = (depth_m * 1000.0).astype(np.float32)
    return depth_mm, K
