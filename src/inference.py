"""
Unified inference module for AFUN.

Usage as script:
    python -m src.inference \\
        --ckpt  outputs/model_step_72500.pt \\
        --config inference \\
        --image  door/color_00.png \\
        --query  "please open the door" \\
        [--depth door/depth_00.npy] \\
        [--output outputs/inference]

Usage as library:
    from src.inference import load_model, infer_image, visualize_2d, visualize_3d
    model  = load_model("model.pt", "inference")
    result = infer_image(model, "image.png", "open the door")
    visualize_2d(result)          # PIL.Image (mask overlay + 2D curve)
    visualize_3d(result)          # plotly figure (needs depth)
"""
from __future__ import annotations

import sys, os, io, pathlib, re
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.models.end_to_end import EndToEndModel
from src.models.decoder.motion_decoder.curves import (
    tensor_to_fsparams,
    compute_fsframe_curve,
    compute_spline_curve,
    plotly_pointcloud_and_curve,
)

# Default camera intrinsics (RealSense-like 640x480).
DEFAULT_CAM = {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0}

# 3D viewer defaults (kept in sync with eval_3d/render_row.py — yellow→blue
# trajectory + red mask blending). The start dot is drawn at
# ``VIS3D_TRAJ_START_SCALE × VIS3D_TRAJ_DOT_SIZE`` so the contact location
# stays readable without dominating the curve.
VIS3D_SCENE_STRIDE     = 1
VIS3D_MAX_POINTS       = 400_000
VIS3D_Z_MIN            = 0.05
VIS3D_MASK_POINTS      = 60_000
VIS3D_MASK_ALPHA       = 0.45
VIS3D_TRAJ_DOT_SIZE    = 14
VIS3D_TRAJ_DENSIFY     = 4
VIS3D_TRAJ_START_COLOR = "#ffff99"   # yellow (trajectory start)
VIS3D_TRAJ_END_COLOR   = "#3282bd"   # blue   (trajectory end)
VIS3D_TRAJ_START_SCALE = 1.35        # emphasize the contact (start) dot

# 2D mirror of the 3D viewer constants — same yellow→blue gradient + 0.45 red
# mask + 1.35× emphasized start dot, scaled for matplotlib scatter.
VIS2D_TRAJ_DOT_SIZE    = 12
VIS2D_TRAJ_START_SCALE = VIS3D_TRAJ_START_SCALE
VIS2D_TRAJ_START_COLOR = VIS3D_TRAJ_START_COLOR
VIS2D_TRAJ_END_COLOR   = VIS3D_TRAJ_END_COLOR
VIS3D_GREY_PAPER       = "#444444"
VIS3D_GREY_FONT        = "white"

_NAMED_COLORS = {
    "red": (255, 0, 0), "lime": (0, 255, 0), "blue": (0, 0, 255),
    "yellow": (255, 255, 0), "green": (0, 128, 0), "orange": (255, 165, 0),
    "purple": (128, 0, 128), "cyan": (0, 255, 255), "magenta": (255, 0, 255),
    "white": (255, 255, 255), "black": (0, 0, 0),
}


def _parse_color_to_rgb(c: str) -> tuple[int, int, int]:
    """Convert a color string (#xxxxxx, rgb(r,g,b), or named) to an (r,g,b) int triple."""
    s = str(c).strip()
    if s.startswith("#"):
        h = s[1:]
        if len(h) == 6:
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        if len(h) == 3:
            return tuple(int(ch * 2, 16) for ch in h)
    m = re.match(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", s)
    if m:
        return tuple(int(g) for g in m.groups())
    return _NAMED_COLORS.get(s.lower(), (200, 200, 200))


# ──────────────────────────────────────────────────────────────────────
# Visualization helpers (mask overlay + 2D/3D curve rendering).
# ──────────────────────────────────────────────────────────────────────
def _project_3d_to_2d(pts_3d: np.ndarray, fx, fy, cx, cy) -> np.ndarray:
    """Project (N,3) camera-frame points to (N,2) pixel coords."""
    if pts_3d is None or len(pts_3d) == 0:
        return np.zeros((0, 2), np.float32)
    z = pts_3d[:, 2].clip(min=1e-6)
    u = fx * pts_3d[:, 0] / z + cx
    v = fy * pts_3d[:, 1] / z + cy
    return np.stack([u, v], axis=1).astype(np.float32)


def _mask_to_image_shape(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    """Nearest-neighbour resize of a bool mask to (H,W); no-op if already sized."""
    if mask.shape[:2] == (H, W):
        return mask.astype(bool)
    return np.array(
        Image.fromarray((mask.astype(bool).astype(np.uint8) * 255), "L").resize(
            (W, H), Image.NEAREST)
    ) > 127


def _compute_motion_curve(result: dict) -> np.ndarray | None:
    """Reconstruct 3D motion curve from fsparams or spline_ctrl. Returns (N,3) or None."""
    motion_type = result.get("motion_type")
    if motion_type == "spline":
        spline_ctrl = result.get("spline_ctrl")
        if spline_ctrl is None:
            return None
        r0 = result.get("r0")
        r0_arr = np.zeros(3, dtype=np.float32) if r0 is None else np.asarray(r0, dtype=np.float32)
        r, _ = compute_spline_curve(
            {"ctrl": torch.from_numpy(spline_ctrl).float(),
             "r0": torch.from_numpy(r0_arr).float()},
        )
        return np.asarray(r, dtype=np.float32)
    if motion_type == "fsframe":
        fsp = result.get("fsparams")
        if fsp is None:
            return None
        r, _ = compute_fsframe_curve(fsp)
        return np.asarray(r, dtype=np.float32)
    return None


# ──────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────
def load_model(ckpt_path, config_name="inference", device="cuda", cfg_dir=None, config_path=None):
    """Load learned weights with the same model builder used for training."""
    from src.models.build import build_model
    from src.utils.checkpoint import load_weights

    if config_path:
        cfg = OmegaConf.load(config_path)
    else:
        with initialize_config_dir(version_base="1.3", config_dir=str(pathlib.Path(cfg_dir or ROOT / "configs").resolve())):
            cfg = compose(config_name=config_name)
    model = build_model(cfg, device, evaluation=True)
    load_weights(model, ckpt_path)
    return model.eval().to(device)


# ──────────────────────────────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────────────────────────────
_preprocess = transforms.Compose([
    transforms.Resize((1008, 1008)),
    transforms.ToTensor(),
])


def _prepare_image(
    image: str | Image.Image | np.ndarray,
    device: str,
) -> tuple[torch.Tensor, Image.Image]:
    """Return (tensor (1,3,1008,1008), original PIL image)."""
    if isinstance(image, (str, pathlib.Path)):
        img_pil = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        img_pil = Image.fromarray(image).convert("RGB")
    else:
        img_pil = image.convert("RGB")
    tensor = _preprocess(img_pil).unsqueeze(0).to(device)
    return tensor, img_pil


# ──────────────────────────────────────────────────────────────────────
# Core inference
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer_image(
    model: EndToEndModel,
    image: str | Image.Image | np.ndarray,
    query: str,
    device: str = "cuda",
    depth: str | np.ndarray | None = None,
    cam: dict | None = None,
) -> dict[str, Any]:
    """Run full inference on a single image + text query.

    Parameters
    ----------
    model : EndToEndModel
        Model returned by :func:`load_model`.
    image : str | PIL.Image | np.ndarray
        Path or raw image.
    query : str
        Natural language query, e.g. ``"open the door"``.
    device : str
        CUDA device string.
    depth : str | np.ndarray | None
        Optional depth map (path to ``.npy`` or array).  Required for 3-D
        motion curve visualization.
    cam : dict | None
        Camera intrinsics ``{fx, fy, cx, cy}``.  Defaults to
        :data:`DEFAULT_CAM`.

    Returns
    -------
    dict with keys:
        image_pil, image_tensor, query,
        mask (H_img×W_img bool, matches original image resolution), score,
        motion_8d, fsparams (dict), T0, r0,
        depth (if provided), cam.
    """
    if cam is None:
        cam = DEFAULT_CAM

    image_tensor, img_pil = _prepare_image(image, device)
    rgb_np = np.array(img_pil)
    W_img, H_img = img_pil.size  # PIL size is (W, H)

    model.eval()
    model.decoder.training = False
    if hasattr(model.decoder, "sam3_model"):
        model.decoder.sam3_model.training = False

    # Load depth early — needed both for the depth_encoder (if the decoder has
    # one) and for r0 back-projection later.
    depth_np = None
    if depth is not None:
        if isinstance(depth, (str, pathlib.Path)):
            depth_np = np.load(depth)
        else:
            depth_np = np.asarray(depth)

    # ── 1. LLM forward → metaquery embeddings ──
    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
        inputs_mq = model.llm.tokenize([query], image=image_tensor)
        llm_kwargs = dict(
            input_ids=inputs_mq["input_ids"].to(device),
            attention_mask=inputs_mq["attention_mask"].to(device),
            pixel_values=inputs_mq.get("pixel_values", torch.empty(0)).to(device),
            image_grid_thw=inputs_mq.get("image_grid_thw", torch.empty(0)).to(device),
            point_cloud_embeddings=None,
        )
        if "mm_token_type_ids" in inputs_mq:
            llm_kwargs["mm_token_type_ids"] = inputs_mq["mm_token_type_ids"].to(device)
        llm_out = model.llm(**llm_kwargs)

    mq_embeds = llm_out["metaquery_prompt_embeds"]
    if isinstance(mq_embeds, dict):
        semantic_embeds = mq_embeds["semantic"]
        motion_embeds = mq_embeds["motion"]
    else:
        semantic_embeds = mq_embeds
        motion_embeds = mq_embeds

    # ── 2. Decoder → segmentation ──
    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
        sam3_out, _ = model.decoder(
            images=image_tensor * 2 - 1,
            feature=semantic_embeds,
            motion_feature=motion_embeds,
            find_target=None,
        )

    # ── 3. Motion processor ──
    has_motion = hasattr(model.decoder, "motion_processor") and model.decoder.motion_processor is not None
    if has_motion:
        # Wire depth through the depth encoder when the decoder has one AND
        # we have a depth map. Required for checkpoints trained with
        # ``enable_geo=True`` (the geo cross-attention / mask-bias branch of
        # the motion processor is a no-op without geo_tokens).
        geo_tokens = None
        geo_shape = None
        depth_encoder = getattr(model.decoder, "depth_encoder", None)
        if depth_encoder is not None and depth_np is not None:
            depth_t = torch.from_numpy(np.asarray(depth_np, dtype=np.float32))
            if depth_t.ndim == 2:
                depth_t = depth_t.unsqueeze(0).unsqueeze(0)   # (1,1,H_d,W_d)
            elif depth_t.ndim == 3:
                depth_t = depth_t.unsqueeze(0)                # (1,1,H,W)
            # The encoder clamps at max_depth (default 10000, mm-scale);
            # keep units native (mm if raw npy is mm).
            depth_t = depth_t.to(device)
            intrinsic_t = torch.tensor(
                [[cam["fx"], 0.0, cam["cx"]],
                 [0.0, cam["fy"], cam["cy"]],
                 [0.0, 0.0, 1.0]],
                dtype=torch.float32, device=device,
            ).unsqueeze(0)  # (1, 3, 3) at depth resolution
            # Match the training collator's depth grid and intrinsic scaling.
            old_h, old_w = depth_t.shape[-2:]
            depth_t = F.interpolate(depth_t, size=(480, 640), mode="bilinear", align_corners=False)
            intrinsic_t[:, 0] *= 640 / old_w
            intrinsic_t[:, 1] *= 480 / old_h
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
                geo_tokens, geo_shape = depth_encoder(
                    image_tensor * 2 - 1, depth_t, intrinsic=intrinsic_t,
                )

        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
            motion_pred = model.decoder.motion_processor(
                sam3_out["queries"],
                motion_embeds,
                geo_tokens=geo_tokens,
                mask_logits=sam3_out.get("pred_masks"),
                geo_spatial_shape=geo_shape,
            )
            sam3_out["motion_directions"] = motion_pred.pred

    # ── 4. Postprocess ──
    # Use the original image size so SAM3 bilinearly upsamples mask logits
    # directly to (H_img, W_img) — avoids a 1008×1008 bool intermediate that
    # would otherwise need a second nearest-neighbour resize downstream.
    post = model.decoder._postprocess(
        sam3_out, orig_sizes=[(H_img, W_img)], has_motion=has_motion,
    )

    masks = post["masks"][0]       # (K, H_img, W_img) bool
    scores = post["scores"][0]     # (K,)

    if masks.shape[0] == 0:
        return {
            "image_pil": img_pil, "image_tensor": image_tensor, "query": query,
            "mask": np.zeros((H_img, W_img), dtype=bool), "score": 0.0,
            "motion_type": None, "motion_raw": None,
            "motion_8d": None, "fsparams": None,
            "spline_ctrl": None, "T0": None, "r0": None,
            "depth": None, "cam": cam, "num_detections": 0,
        }

    best_idx = int(scores.argmax().item())
    best_mask = masks[best_idx].cpu().numpy()
    best_score = float(scores[best_idx].item())

    # Motion — support both FSFrame (8D) and spline (9D Bezier control points)
    fsparams = None
    spline_ctrl = None
    T0 = None
    r0 = None
    motion_8d = None
    motion_raw = None
    motion_type = None
    if has_motion and "motion_directions" in post:
        motion_dirs = post["motion_directions"][0]
        # .float() because the motion processor runs under bfloat16 autocast
        # but numpy() doesn't support bfloat16.
        motion_t = motion_dirs[best_idx].detach().float().cpu()
        motion_raw = motion_t.numpy()
        motion_dim = motion_raw.shape[-1]

        if motion_dim == 8:
            motion_type = "fsframe"
            motion_8d = motion_raw
            fsparams = tensor_to_fsparams(motion_t)
            T0 = np.array(fsparams["T0"])
        elif motion_dim == 9:
            motion_type = "spline"
            # Bezier control points relative to P0=origin: [P1, P2, P3]
            spline_ctrl = motion_raw.reshape(3, 3)
            # Initial tangent (T0) = direction from origin (P0) to P1
            p1 = spline_ctrl[0]
            n = float(np.linalg.norm(p1))
            if n > 1e-8:
                T0 = p1 / n
            else:
                T0 = np.array([0.0, 0.0, 1.0])
        else:
            raise ValueError(f"Unsupported motion_directions dim {motion_dim} (expected 8 or 9)")

    # Depth → 3D curve origin (r0) via mask centroid back-projection.
    if depth_np is not None and motion_type is not None:
        r0 = _mask_centroid_3d(best_mask, depth_np, cam)
        if r0 is None:
            r0 = np.array([0.0, 0.0, 1.0])
        if fsparams is not None:
            fsparams["r0"] = np.array(r0)

    return {
        "image_pil": img_pil,
        "image_tensor": image_tensor,
        "rgb_np": rgb_np,
        "query": query,
        "mask": best_mask,
        "score": best_score,
        "motion_type": motion_type,
        "motion_raw": motion_raw,
        "motion_8d": motion_8d,
        "fsparams": fsparams,
        "spline_ctrl": spline_ctrl,
        "T0": T0,
        "r0": r0,
        "depth": depth_np,
        "cam": cam,
        "num_detections": int(masks.shape[0]),
    }


def _mask_centroid_3d(
    mask: np.ndarray,
    depth: np.ndarray,
    cam: dict,
) -> np.ndarray | None:
    """Back-project the mask centroid using the mean valid masked depth.

    ``mask`` is expected at the original image resolution. If depth has a
    different resolution, the mask is resized to match (nearest-neighbour).
    """
    H_d, W_d = depth.shape
    if mask.shape[:2] != (H_d, W_d):
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        mask_at_depth = F.interpolate(mask_t, size=(H_d, W_d), mode="nearest")[0, 0].numpy() > 0.5
    else:
        mask_at_depth = mask.astype(bool)

    ys, xs = np.nonzero(mask_at_depth)
    if len(xs) == 0:
        return None

    y_c, x_c = int(ys.mean()), int(xs.mean())
    masked_depth = np.asarray(depth[mask_at_depth], dtype=np.float32)
    masked_depth = masked_depth[np.isfinite(masked_depth) & (masked_depth > 0)]
    if masked_depth.size == 0:
        return None
    z = float(masked_depth.mean())

    if float(np.nanmax(depth)) > 100.0:
        z = z / 1000.0

    fx, fy = cam["fx"], cam["fy"]
    cx, cy = cam["cx"], cam["cy"]
    x_3d = (x_c - cx) * z / fx
    y_3d = (y_c - cy) * z / fy
    return np.array([x_3d, y_3d, z])


# ──────────────────────────────────────────────────────────────────────
# Visualization: 2D composite (mask overlay + projected 3D motion curve)
# ──────────────────────────────────────────────────────────────────────
def visualize_2d(result: dict, show: bool = False) -> Image.Image:
    """Build the 2D composite: RGB + red mask overlay (alpha 0.45) + yellow→blue
    trajectory dots, matching the eval_3d/render_row.py qualitative-row style.

    When ``result['detections']`` is non-empty, every detection is overlaid;
    otherwise we fall back to the top-1 keys. The first dot of each curve is
    drawn larger (white-edged) so the contact point is readable at a glance.

    Returns
    -------
    PIL.Image.Image
    """
    import matplotlib  # local import keeps script-mode import-light
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rgb = np.array(result["image_pil"].convert("RGB"))
    H, W = rgb.shape[:2]
    cam = result.get("cam") or DEFAULT_CAM
    fx, fy = float(cam["fx"]), float(cam["fy"])
    cx, cy = float(cam["cx"]), float(cam["cy"])

    detections = result.get("detections") or []
    if not detections:
        # Single-detection back-compat from the top-1 keys.
        detections = [{
            "mask": result.get("mask"),
            "score": result.get("score", 0.0),
            "motion_type": result.get("motion_type"),
            "spline_ctrl": result.get("spline_ctrl"),
            "fsparams": result.get("fsparams"),
            "r0": result.get("r0"),
        }]

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(rgb)
    for det in detections:
        m = det.get("mask")
        if m is None:
            continue
        m_img = _mask_to_image_shape(m, H, W)
        overlay = np.zeros((H, W, 4), dtype=np.float32)
        overlay[..., 0] = 1.0                                   # red
        overlay[..., 3] = m_img.astype(np.float32) * VIS3D_MASK_ALPHA
        ax.imshow(overlay)
    for det in detections:
        curve_3d = _compute_motion_curve(det)
        if curve_3d is None or len(curve_3d) < 2:
            continue
        pts2d = _project_3d_to_2d(curve_3d, fx, fy, cx, cy)
        if len(pts2d) < 1:
            continue
        n = len(pts2d)
        rgb1 = np.array(_parse_color_to_rgb(VIS2D_TRAJ_START_COLOR), dtype=np.float32) / 255.0
        rgb2 = np.array(_parse_color_to_rgb(VIS2D_TRAJ_END_COLOR),   dtype=np.float32) / 255.0
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
        colors = rgb1[None] * (1.0 - t) + rgb2[None] * t        # yellow→blue
        ax.scatter(pts2d[:, 0], pts2d[:, 1],
                   c=colors, s=VIS2D_TRAJ_DOT_SIZE ** 2, zorder=3, linewidths=0)
        ax.scatter([pts2d[0, 0]], [pts2d[0, 1]],
                   c=[colors[0]], s=(VIS2D_TRAJ_START_SCALE * VIS2D_TRAJ_DOT_SIZE) ** 2,
                   edgecolors="white", linewidths=1.2, zorder=4)
    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    fig.subplots_adjust(0, 0, 1, 1)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    out = Image.open(buf).convert("RGB").copy()
    buf.close()
    if show:
        out.show()
    return out


# ──────────────────────────────────────────────────────────────────────
# Visualization: 3D point cloud + motion curve (Plotly)
# ──────────────────────────────────────────────────────────────────────
def _compute_curve_and_frames(result: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Reconstruct (r_curve, R_curve) from result, matching make_wandb_vis."""
    motion_type = result.get("motion_type")
    r0 = result.get("r0")
    if r0 is None:
        return None
    r0_t = torch.as_tensor(np.asarray(r0, dtype=np.float32), dtype=torch.float32)

    if motion_type == "spline":
        spline_ctrl = result.get("spline_ctrl")
        if spline_ctrl is None:
            return None
        ctrl = torch.from_numpy(np.asarray(spline_ctrl, dtype=np.float32)).float()
        r, R = compute_spline_curve({"ctrl": ctrl, "r0": r0_t})
        return np.asarray(r, dtype=np.float32), np.asarray(R, dtype=np.float32)
    if motion_type == "fsframe":
        fsp = dict(result.get("fsparams") or {})
        if not fsp:
            return None
        fsp["r0"] = r0_t
        r, R = compute_fsframe_curve(fsp)
        return np.asarray(r, dtype=np.float32), np.asarray(R, dtype=np.float32)
    return None


def visualize_3d(result: dict, show: bool = False,
                 *, stride: int = VIS3D_SCENE_STRIDE,
                 max_points: int = VIS3D_MAX_POINTS,
                 z_min: float = VIS3D_Z_MIN, z_max: float = 2.0,
                 mask_points: int = VIS3D_MASK_POINTS,
                 mask_alpha: float = VIS3D_MASK_ALPHA,
                 traj_dot_size: int = VIS3D_TRAJ_DOT_SIZE,
                 traj_densify: int = VIS3D_TRAJ_DENSIFY,
                 traj_start_color: str = VIS3D_TRAJ_START_COLOR,
                 traj_end_color: str = VIS3D_TRAJ_END_COLOR):
    """Interactive 3D scene matching `eval_3d/view_npz.ipynb`.

    Renders, in order:
      1. Dense RGB-coloured point cloud (back-projected from depth + intrinsics).
      2. Predicted mask, alpha-blended with red so scene texture stays visible.
      3. Motion trajectory drawn as `scene.annotations` (image-plane bullets) so
         no point cloud or mask point can ever occlude the curve.

    The figure uses a flat grey paper, no axes / grid / ticks / legend.

    Returns
    -------
    plotly Figure
    """
    depth = result["depth"]
    if depth is None:
        raise ValueError("No depth provided — pass depth to infer_image() for 3D visualization.")

    rgb_np = result["rgb_np"]
    cam = result["cam"]
    motion_type = result.get("motion_type")
    fsparams = result.get("fsparams")
    spline_ctrl = result.get("spline_ctrl")
    T0 = result.get("T0")
    query = result["query"]
    score = float(result.get("score", 0.0))
    mask = result.get("mask")

    # Depth → meters
    depth_m = np.asarray(depth, dtype=np.float32)
    depth_m = np.where(np.isfinite(depth_m), depth_m, 0.0)
    if float(np.nanmax(depth_m)) > 100.0:
        depth_m = depth_m / 1000.0

    # If RGB and depth differ in resolution, resize RGB to match (shared intrinsics).
    H_d, W_d = depth_m.shape
    if rgb_np.shape[:2] != (H_d, W_d):
        rgb_np = np.array(Image.fromarray(rgb_np).resize((W_d, H_d), Image.BILINEAR))

    # Resize mask to depth resolution if needed (nearest-neighbour preserves binarity).
    if mask is not None:
        mask = np.asarray(mask).astype(bool)
        if mask.shape != (H_d, W_d):
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8) * 255)
                .resize((W_d, H_d), Image.NEAREST)
            ) > 127

    curve = _compute_curve_and_frames(result)
    if curve is None:
        raise ValueError(
            f"Cannot reconstruct motion curve (motion_type={motion_type!r}, r0={result.get('r0')!r})."
        )
    r_curve, _R_curve = curve

    fx, fy, cx, cy = float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])
    cam_intrinsic = {
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "scene_stride": int(stride),
        "z_min": float(z_min),
        "z_max": float(z_max),
        "max_points": int(max_points),
    }

    # Title with motion params.
    if motion_type == "spline" and spline_ctrl is not None:
        p1, p2, p3 = spline_ctrl[0], spline_ctrl[1], spline_ctrl[2]
        T0_arr = np.zeros(3) if T0 is None else T0
        title_str = (
            f'Spline — "{query}" (score={score:.3f})<br>'
            f"T0=[{T0_arr[0]:.3f}, {T0_arr[1]:.3f}, {T0_arr[2]:.3f}]  "
            f"P1=[{p1[0]:.2f}, {p1[1]:.2f}, {p1[2]:.2f}]  "
            f"P2=[{p2[0]:.2f}, {p2[1]:.2f}, {p2[2]:.2f}]  "
            f"P3=[{p3[0]:.2f}, {p3[1]:.2f}, {p3[2]:.2f}]"
        )
    elif motion_type == "fsframe" and fsparams is not None:
        kappa = float(fsparams["kappa"])
        tau = float(fsparams["tau"])
        velocity = float(fsparams["velocity"])
        T0_arr = np.zeros(3) if T0 is None else T0
        title_str = (
            f'FS-frame — "{query}" (score={score:.3f})<br>'
            f"T0=[{T0_arr[0]:.3f}, {T0_arr[1]:.3f}, {T0_arr[2]:.3f}]  "
            f"kappa={kappa:.3f}  tau={tau:.3f}  vel={velocity:.3f}"
        )
    else:
        title_str = f'"{query}" (score={score:.3f})'

    # Base scene: dense point cloud only (the trajectory is drawn as image-plane
    # annotations so it's always on top of any 3D primitive).
    fig = plotly_pointcloud_and_curve(
        depth_m, rgb_np,
        cam_intrinsic=cam_intrinsic,
        r_curve=None, R_curve=None,
        title=title_str,
    )

    # Mask overlay: blend underlying RGB with red so texture stays visible.
    if mask is not None and mask.shape == depth_m.shape:
        ys, xs = np.where(mask & (depth_m > z_min) & (depth_m < z_max))
        if ys.size:
            if ys.size > int(mask_points):
                sel = np.random.default_rng(0).choice(ys.size, int(mask_points), replace=False)
                ys, xs = ys[sel], xs[sel]
            zk = depth_m[ys, xs]
            X = (xs.astype(np.float32) - cx) * zk / fx
            Y = (ys.astype(np.float32) - cy) * zk / fy
            base = rgb_np[ys, xs].astype(np.float32)
            red  = np.array([255.0, 0.0, 0.0], dtype=np.float32)
            blended = (float(mask_alpha) * red
                       + (1.0 - float(mask_alpha)) * base).clip(0, 255).astype(np.uint8)
            colors = [f"rgb({r},{g},{b})" for r, g, b in blended]
            fig.add_trace(go.Scatter3d(
                x=X, y=Y, z=zk, mode="markers",
                marker=dict(size=3, color=colors, opacity=0.9),
                name="Predicted mask", hoverinfo="skip",
            ))

    # Trajectory: 2D image-plane annotations (always on top).
    scene_annotations = []
    if r_curve is not None and len(r_curve) >= 2:
        poly = np.asarray(r_curve, dtype=np.float32)
        df = max(1, int(traj_densify))
        if df > 1:
            n = len(poly)
            t_old = np.arange(n, dtype=np.float32)
            t_new = np.linspace(0.0, n - 1, (n - 1) * df + 1, dtype=np.float32)
            poly = np.column_stack([np.interp(t_new, t_old, poly[:, k]) for k in range(3)])
        rgb1 = np.array(_parse_color_to_rgb(traj_start_color), dtype=np.float32)
        rgb2 = np.array(_parse_color_to_rgb(traj_end_color),   dtype=np.float32)
        M = len(poly)
        for i, p in enumerate(poly):
            t = i / (M - 1) if M > 1 else 0.0
            c = (rgb1 * (1.0 - t) + rgb2 * t).clip(0, 255).astype(int)
            scene_annotations.append(dict(
                x=float(p[0]), y=float(p[1]), z=float(p[2]),
                text="●", showarrow=False,
                xanchor="center", yanchor="middle",
                font=dict(color=f"rgb({int(c[0])},{int(c[1])},{int(c[2])})",
                          size=int(traj_dot_size)),
            ))

    blank_axis = dict(visible=False, showbackground=False, showgrid=False,
                      zeroline=False, showticklabels=False, title="")
    fig.update_layout(
        height=750,
        showlegend=False,
        paper_bgcolor=VIS3D_GREY_PAPER,
        plot_bgcolor=VIS3D_GREY_PAPER,
        font=dict(color=VIS3D_GREY_FONT),
        scene=dict(
            xaxis=blank_axis, yaxis=blank_axis, zaxis=blank_axis,
            camera=dict(eye=dict(x=0, y=0, z=-2.0)),
            annotations=scene_annotations,
        ),
    )

    if show:
        fig.show()
    return fig


# ──────────────────────────────────────────────────────────────────────
# Save helpers (for script mode)
# ──────────────────────────────────────────────────────────────────────
def save_results(result: dict, output_dir: str, tag: str = "pred"):
    """Save prediction as .npz + composite PNG + optional 3D HTML.

    Output files (following the review-HTML convention):
      - ``{tag}.npz``     — mask, score, query, motion params
      - ``{tag}_seg.png`` — RGB + orange mask overlay + projected 2D curve
      - ``{tag}_3d.html`` — Plotly scene with first-person camera (if depth)
    """
    os.makedirs(output_dir, exist_ok=True)

    save_data = dict(
        mask=result["mask"],
        score=result["score"],
        query=result["query"],
    )
    motion_type = result.get("motion_type")
    if motion_type == "fsframe" and result.get("motion_8d") is not None:
        save_data.update(
            motion_type="fsframe",
            motion_8d=result["motion_8d"],
            T0=result["T0"],
            kappa=float(result["fsparams"]["kappa"]),
            tau=float(result["fsparams"]["tau"]),
            velocity=float(result["fsparams"]["velocity"]),
        )
    elif motion_type == "spline" and result.get("spline_ctrl") is not None:
        save_data.update(
            motion_type="spline",
            spline_ctrl=result["spline_ctrl"],
            T0=result["T0"],
        )
    if result["r0"] is not None:
        save_data["r0"] = result["r0"]

    # Uniformly-sampled points along the predicted curve (N ∈ {10..50}).
    # For spline, evaluate the cubic Bézier at t=linspace(0,1,N). For
    # fsframe we fall back to linear interp along the full 31-point curve.
    r0_arr = save_data.get("r0")
    if motion_type == "spline" and r0_arr is not None:
        sample_counts = [10, 20, 30, 40, 50]
        ctrl = np.asarray(result["spline_ctrl"], dtype=np.float32).reshape(3, 3)
        P0 = np.asarray(r0_arr, dtype=np.float32)
        P1, P2, P3 = P0 + ctrl[0], P0 + ctrl[1], P0 + ctrl[2]
        save_data["counts"] = np.array(sample_counts, dtype=np.int32)
        for N in sample_counts:
            t = np.linspace(0.0, 1.0, N, dtype=np.float32)[:, None]
            u = 1.0 - t
            pts = (u ** 3) * P0 + 3 * (u ** 2) * t * P1 + 3 * u * (t ** 2) * P2 + (t ** 3) * P3
            save_data[f"points_N{N}"] = pts.astype(np.float32)

    np.savez(os.path.join(output_dir, f"{tag}.npz"), **save_data)

    # 2D composite
    composite = visualize_2d(result, show=False)
    composite.save(os.path.join(output_dir, f"{tag}_seg.png"))

    # 3D HTML (if depth + motion available — visualize_3d needs both).
    # Use a scene-adaptive z_max so deep scenes (e.g. DA3-metric living rooms)
    # don't get all their points filtered out by the default 2.0m cap.
    if result.get("depth") is not None and result.get("motion_type") is not None:
        d_m = np.asarray(result["depth"], dtype=np.float32)
        d_m = np.where(np.isfinite(d_m), d_m, 0.0)
        if float(np.nanmax(d_m)) > 100.0:
            d_m = d_m / 1000.0
        valid = d_m[d_m > 0]
        z_max = float(np.percentile(valid, 99)) + 0.5 if valid.size else 5.0
        z_max = max(z_max, 2.0)
        fig3d = visualize_3d(result, show=False, z_max=z_max)
        fig3d.write_html(os.path.join(output_dir, f"{tag}_3d.html"))

    print(f"Results saved to {output_dir}/{tag}*")

# NOTE: the standalone CLI (`python -m src.inference ...`) was removed — use
# demo.py (single image) or run_examples.py (batch). This module is now a
# library: load_model / infer_image / save_results / visualize_2d / visualize_3d.
