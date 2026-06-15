"""LingBot-Depth wrapper: refine noisy RealSense raw depth using RGB + intrinsics.

Pure in-memory (no file I/O). The model is loaded lazily on first call and
cached as a module-level singleton.

Requires `xformers >= 0.0.35` (installed by install.sh).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch
from PIL import Image

# Vendored lingbot source tree (patched in-tree; see extern/README.md)
_LINGBOT_ROOT = pathlib.Path(__file__).resolve().parent.parent / "extern" / "lingbot-depth"

MODEL_ID = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"

# Module-level singleton — one MDM load per Python process
_MDM = None


def _ensure_paths() -> None:
    s = str(_LINGBOT_ROOT)
    if s not in sys.path:
        sys.path.insert(0, s)


def _load_model(device: torch.device):
    """Load LingBot once, cache, return."""
    global _MDM
    if _MDM is not None:
        return _MDM
    _ensure_paths()
    try:
        from mdm.model.v2 import MDMModel  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            f"lingbot-depth (mdm) not importable from {_LINGBOT_ROOT}. "
            "Check that extern/lingbot-depth/ is intact and the env "
            "has xformers>=0.0.35. See extern/README.md."
        ) from e
    print(f"[depth.lingbot] loading {MODEL_ID} on {device} (one-time; weights to ~/.cache/huggingface/hub/)")
    _MDM = MDMModel.from_pretrained(MODEL_ID).to(device).eval()
    return _MDM


def refine_depth(
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    device: str | torch.device = "cuda",
) -> np.ndarray:
    """Refine a noisy sensor depth map using RGB + intrinsics.

    Parameters
    ----------
    rgb : np.ndarray
        (H, W, 3) uint8 RGB image. Resized to depth dims if mismatched.
    depth_mm : np.ndarray
        (H, W) float32 depth in millimetres. Zero pixels are treated as invalid
        and filled by the model.
    K : np.ndarray
        (3, 3) intrinsic matrix in pixel units, at the depth resolution.
    device : str | torch.device
        Device for inference; falls back to CPU if CUDA is unavailable.

    Returns
    -------
    refined_depth_mm : np.ndarray
        (H, W) float32 refined depth in millimetres, same shape as input depth.
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"rgb must be (H, W, 3) uint8; got shape={rgb.shape} dtype={rgb.dtype}"
        )
    if depth_mm.ndim != 2:
        raise ValueError(f"depth_mm must be (H, W); got shape={depth_mm.shape}")
    if K.shape != (3, 3):
        raise ValueError(f"K must be (3, 3); got shape={K.shape}")

    depth_mm = np.nan_to_num(depth_mm.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    H_d, W_d = depth_mm.shape

    # Resize RGB to match depth if needed
    if rgb.shape[:2] != (H_d, W_d):
        rgb = np.asarray(Image.fromarray(rgb).resize((W_d, H_d), Image.BILINEAR))

    device = torch.device(device) if not isinstance(device, torch.device) else device
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    # Build the three tensors lingbot expects
    image_t = (
        torch.from_numpy(rgb.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    depth_t = torch.from_numpy(depth_mm / 1000.0).to(device)  # mm → m

    K_norm = K.astype(np.float32).copy()
    K_norm[0, 0] /= W_d
    K_norm[0, 2] /= W_d
    K_norm[1, 1] /= H_d
    K_norm[1, 2] /= H_d
    intrinsics_t = torch.from_numpy(K_norm).unsqueeze(0).to(device)

    model = _load_model(device)
    with torch.no_grad():
        out = model.infer(
            image_t,
            depth_in=depth_t,
            apply_mask=True,
            intrinsics=intrinsics_t,
        )
    refined_m = out["depth"].squeeze().detach().cpu().numpy().astype(np.float32)
    refined_mm = (refined_m * 1000.0).astype(np.float32)
    return refined_mm
