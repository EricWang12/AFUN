from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import matplotlib.figure
import torch
from typing import Union

def _to_pil(im):
    im = im if isinstance(im, Image.Image) else Image.fromarray(np.asarray(im))
    return im.convert("RGB")


def update_cfg(cfg, attr, value):
    OmegaConf.set_struct(cfg, False)
    setattr(cfg, attr, value)
    OmegaConf.set_struct(cfg, True)
    return cfg

# potential input: PIL Image, matplotlib Figure, numpy array, torch tensor
def _to_hwc_u8(x : Union[Image.Image, matplotlib.figure.Figure, np.ndarray, torch.Tensor]) -> np.ndarray:
    if hasattr(x, "get_array"):
        x = np.asarray(x.get_array())
    elif hasattr(x, "canvas"):  # matplotlib Figure
        x.canvas.draw()
        x = np.asarray(x.canvas.buffer_rgba())[..., :3]  # HWC RGB uint8
    else:
        x = np.asarray(x)

    if x.ndim == 2:                 # HW -> HWC
        x = np.repeat(x[..., None], 3, axis=2)
    elif x.ndim == 3 and x.shape[2] == 4:  # RGBA -> RGB
        x = x[..., :3]

    if x.dtype != np.uint8:
        x = x.astype(np.uint8)
    return np.ascontiguousarray(x)


def concat_side_by_side(img_l, img_r, *, bgr=False, pad_value=0):
    a = _to_hwc_u8(img_l)
    b = _to_hwc_u8(img_r)

    # visualize_trajectory_on_mask returns BGR; wandb expects RGB
    if bgr:
        a = a[..., ::-1]
        b = b[..., ::-1]

    Ha, Wa = a.shape[:2]
    Hb, Wb = b.shape[:2]
    H = max(Ha, Hb)

    out = np.full((H, Wa + Wb, 3), pad_value, dtype=np.uint8)
    out[:Ha, :Wa] = a
    out[:Hb, Wa:Wa + Wb] = b
    return out