# spline_processor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decoder.motion_decoder.motion_processor import (
    BaseMotionProcessor,
)
from src.models.decoder.motion_decoder.FSFrame import compute_spline_curve, plotly_pointcloud_and_curve
import wandb


def _binomial_coeffs(n: int, *, device=None, dtype=None) -> torch.Tensor:
    """Return binomial coefficients C(n,0..n) as a 1-D tensor."""
    vals = [1]
    for k in range(1, n + 1):
        vals.append(vals[-1] * (n - k + 1) // k)
    return torch.tensor(vals, device=device, dtype=dtype)


def _curve_frames_from_points(pts: torch.Tensor, r0=None):
    """Convert (T,3) sample points into (r_curve, R_curve) for plotting.

    Frames are a simple parallel-transport chain from a crude initial frame;
    only the arrow-tip tangent R_curve[-1,:,0] is used by the plotter.
    """
    if torch.is_tensor(r0):
        r0 = r0.to(dtype=pts.dtype, device=pts.device)
    elif r0 is not None:
        import numpy as _np
        r0 = torch.as_tensor(_np.asarray(r0), dtype=pts.dtype, device=pts.device)
    if r0 is not None and r0.shape[-1] == 3:
        pts = pts + r0[None, :]
    # Tangents from finite differences.
    deltas = torch.cat([pts[1:2] - pts[0:1], pts[2:] - pts[:-2], pts[-1:] - pts[-2:-1]], dim=0)
    T = deltas / (torch.linalg.norm(deltas, dim=-1, keepdim=True) + 1e-12)
    # Build any perpendicular axis
    up = torch.tensor([0.0, 0.0, 1.0], dtype=pts.dtype, device=pts.device)
    axis = torch.linalg.cross(T, up[None, :].expand_as(T), dim=-1)
    norm = torch.linalg.norm(axis, dim=-1, keepdim=True)
    fallback = torch.tensor([0.0, 1.0, 0.0], dtype=pts.dtype, device=pts.device)
    axis = torch.where(norm < 1e-6, torch.linalg.cross(T, fallback[None, :].expand_as(T), dim=-1), axis)
    N = axis / (torch.linalg.norm(axis, dim=-1, keepdim=True) + 1e-12)
    B = torch.linalg.cross(T, N, dim=-1)
    R = torch.stack([T, N, B], dim=-1)
    return pts.detach().cpu().numpy(), R.detach().cpu().numpy()


@dataclass
class SplineProcessorOutput:
    pred: torch.Tensor  # (B, N, 3*num_predicted_ctrl)  per token


class SplineProcessor(BaseMotionProcessor):
    """
    Inputs:
      - query_feats:   (B, 200, 256)
      - context_feats: (B, 32, 4096)

    Output:
      - spline control points: (B, N, 3*K) where K = num_predicted_ctrl.
        P0 is implicit at the origin; the K predicted points define a
        degree-K Bezier curve with K+1 total control points.
    """

    def __init__(
        self,
        model_cfg: Dict[str, Any],
        loss_cfg: Dict[str, Any],
    ) -> None:
        super().__init__(model_cfg, loss_cfg)

        from src.models.decoder.motion_decoder.curve_loss import CurveLoss
        curve_T = int(loss_cfg.get("curve_T", 16))
        self.curve_loss = CurveLoss(T=curve_T)

        self.num_predicted_ctrl = int(model_cfg.get("num_predicted_ctrl", 3))
        if self.num_predicted_ctrl < 2:
            raise ValueError(
                f"num_predicted_ctrl must be >= 2, got {self.num_predicted_ctrl}"
            )
        output_dim = int(model_cfg.get("output_dim", 3 * self.num_predicted_ctrl))
        if output_dim != 3 * self.num_predicted_ctrl:
            raise ValueError(
                f"output_dim ({output_dim}) must equal 3*num_predicted_ctrl "
                f"({3 * self.num_predicted_ctrl})"
            )
        self.output_dim = output_dim

        self.head = nn.Sequential(
            nn.LayerNorm(self.q_dim),
            nn.Linear(self.q_dim, self.head_hidden),
            nn.GELU(),
            nn.Linear(self.head_hidden, output_dim),
        )

    def forward(
        self,
        query_feats: torch.Tensor,    # (B, 200, 256)
        context_feats: torch.Tensor,  # (B, 32, 4096)
        *,
        geo_tokens: Optional[torch.Tensor] = None,
        mask_logits: Optional[torch.Tensor] = None,
        geo_spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> SplineProcessorOutput:
        x = self._run_transformer(
            query_feats, context_feats,
            geo_tokens=geo_tokens,
            mask_logits=mask_logits,
            geo_spatial_shape=geo_spatial_shape,
        )

        pred = self.head(x)  # (B, N, 3*K)
        return SplineProcessorOutput(pred=pred)

    def sample_curve(self, params: torch.Tensor, T: int) -> torch.Tensor:
        """Sample a degree-K Bezier curve at T evenly-spaced parameter values.

        Args:
            params: (M, 3*K) relative control points [P1..PK] with P0 fixed at the origin.
            T: number of sample points along the curve.

        Returns:
            (M, T, 3) sampled positions.
        """
        M = params.shape[0]
        K3 = params.shape[-1]
        if K3 % 3 != 0:
            raise ValueError(f"sample_curve: params last dim must be divisible by 3; got {K3}")
        K = K3 // 3
        pts = params.view(M, K, 3)                              # (M, K, 3)
        origin = torch.zeros_like(pts[:, :1, :])                # (M, 1, 3)
        ctrl = torch.cat([origin, pts], dim=1)                  # (M, K+1, 3)
        n = K                                                    # Bezier degree

        t = torch.linspace(0, 1, T, device=params.device, dtype=params.dtype)  # (T,)
        one_minus_t = 1.0 - t
        # Bernstein basis B_{k,n}(t) = C(n,k) * t^k * (1-t)^(n-k)  for k=0..n
        ks = torch.arange(n + 1, device=params.device, dtype=params.dtype)      # (n+1,)
        binom = _binomial_coeffs(n, device=params.device, dtype=params.dtype)  # (n+1,)
        log_t = torch.log(t.clamp_min(1e-30))                                   # (T,)
        log_um = torch.log(one_minus_t.clamp_min(1e-30))                        # (T,)
        # (n+1, T) Bernstein weights in log-space, exponentiate
        log_B = binom.log()[:, None] + ks[:, None] * log_t[None, :] + (n - ks)[:, None] * log_um[None, :]
        B = log_B.exp()                                                         # (n+1, T)
        # Clean endpoints (clamp_min replaced 0 with 1e-30, giving huge negatives
        # → near-0 after exp; but the 0^0 convention requires exact 1 at t=0/1).
        B[0, 0] = 1.0
        B[0, -1] = 0.0
        B[-1, 0] = 0.0
        B[-1, -1] = 1.0
        # samples[m, t, :] = sum_k B[k,t] * ctrl[m, k, :]
        samples = torch.einsum("kt,mkd->mtd", B, ctrl)
        return samples

    def compute_loss(
        self,
        pred_ctrl_dense: torch.Tensor,              # (B, N, 3*K)
        target: torch.Tensor,                        # (M, 3*Kg) legacy ctrl, or (M, T, 3) sampled pts
        indices: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        *,
        wDeg: float = 0.0,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> dict:
        """
        Two GT paths, selected by target.ndim:
          * ndim == 2: legacy Bezier-ctrl GT (M, 3*Kg). Pred and GT curves are both
            sampled with sample_curve and compared. Requires Kg == self.num_predicted_ctrl.
          * ndim == 3: pre-sampled curve GT (M, T, 3). Only the prediction is sampled.
        """
        weight = float(self.loss_cfg.get("weight", 1.0))

        out_dim = 3 * self.num_predicted_ctrl
        if pred_ctrl_dense.ndim != 3 or pred_ctrl_dense.size(-1) != out_dim:
            raise ValueError(
                f"pred_ctrl_dense must be (B,N,{out_dim}). Got {tuple(pred_ctrl_dense.shape)}"
            )

        batch_idx, token_idx, _ = indices
        if batch_idx is None or token_idx is None:
            raise ValueError("indices must provide (batch_idx, token_idx, _).")

        M = int(batch_idx.numel())
        if target.size(0) != M:
            raise ValueError(f"target first dim must match indices length M={M}.")
        if M == 0:
            z = pred_ctrl_dense.sum() * 0.0
            return {"loss": z, "Lcurve": z, "Ldeg": z}

        device = pred_ctrl_dense.device
        batch_idx = batch_idx.to(device=device)
        token_idx = token_idx.to(device=device)
        target = target.to(device=device)

        B, N, _ = pred_ctrl_dense.shape
        if (batch_idx.min() < 0) or (batch_idx.max() >= B):
            raise ValueError(f"batch_idx out of bounds for B={B}")
        if (token_idx.min() < 0) or (token_idx.max() >= N):
            raise ValueError(f"token_idx out of bounds for N={N}")

        pred_sparse = pred_ctrl_dense[batch_idx, token_idx, :]  # (M, 3*K)
        T = self.curve_loss.T
        pred_samples = self.sample_curve(pred_sparse, T)         # (M, T, 3)

        if target.ndim == 3:
            # Pre-sampled GT path: (M, T_gt, 3). Require matching T.
            if target.shape[1] != T or target.shape[2] != 3:
                raise ValueError(
                    f"target (M,T,3) has T={target.shape[1]} but curve_loss.T={T}."
                )
            gt_samples = target
        elif target.ndim == 2:
            # Legacy Bezier-ctrl GT path.
            if target.shape[-1] != out_dim:
                raise ValueError(
                    f"Legacy target shape {tuple(target.shape)} does not match "
                    f"num_predicted_ctrl={self.num_predicted_ctrl} (expected last dim {out_dim}). "
                    f"Either set num_predicted_ctrl to match or switch the data loader "
                    f"to produce (M,T,3) pre-sampled curves."
                )
            gt_samples = self.sample_curve(target, T)
        else:
            raise ValueError(
                f"target must be (M, 3*K) or (M, T, 3); got shape {tuple(target.shape)}"
            )

        Lcurve = self.curve_loss(pred_samples, gt_samples)

        # Degeneracy regularizer (legacy cubic-only): penalize collapsed curves.
        if wDeg != 0.0 and self.num_predicted_ctrl == 3:
            pred_pts = pred_sparse.view(M, 3, 3)
            p0, p1, p2 = pred_pts[:, 0], pred_pts[:, 1], pred_pts[:, 2]
            d02 = (p2 - p0).norm(dim=-1)
            d01 = (p1 - p0).norm(dim=-1)
            d12 = (p2 - p1).norm(dim=-1)
            Ldeg_vec = (F.relu(eps - d02) ** 2) + 0.5 * (F.relu(eps - d01) ** 2 + F.relu(eps - d12) ** 2)
            Ldeg = Ldeg_vec.mean() if reduction == "mean" else (Ldeg_vec.sum() if reduction == "sum" else Ldeg_vec)
        else:
            Ldeg = torch.zeros((), device=device)

        loss = (Lcurve + wDeg * Ldeg) * weight
        return {"loss": loss, "Lcurve": Lcurve, "Ldeg": Ldeg}


    def make_wandb_vis(self, batch: Dict[str, Any], pred_results: Dict[str, Any], gt_results: Dict[str, Any], *, step: int, idx: int = 0, output_path: str = None, pred_masks=None) -> Dict[str, Any]:
        import traceback as _tb
        try:
            return self._make_wandb_vis_impl(batch, pred_results, gt_results, step=step, idx=idx, output_path=output_path, pred_masks=pred_masks)
        except Exception as e:
            print(f"[SplineProcessor.make_wandb_vis] ERROR: {e}")
            _tb.print_exc()
            return {}

    def _make_wandb_vis_impl(self, batch: Dict[str, Any], pred_results: Dict[str, Any], gt_results: Dict[str, Any], *, step: int, idx: int = 0, output_path: str = None, pred_masks=None) -> Dict[str, Any]:

        # NOTE: assumes a single object per image.

        depth = batch['depth'][idx].cpu().numpy()
        dh, dw = depth.shape[-2], depth.shape[-1]

        # Use stacked batch["image"] tensor (always available, (B,3,H,W)) to avoid
        # list-of-tensors issues with image_raw / image_unnormalized.
        x = batch["image"][idx].detach().float()
        if x.ndim == 3 and x.shape[0] == 3:
            if x.min() < 0:
                x = x * 0.5 + 0.5  # SAM3 normalization: [-1,1] -> [0,1]
            x = x.clamp(0, 1)
            rgb = (x * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        else:
            raise ValueError("Input image tensor must be shape (3, H, W)")

        # Resize rgb to match depth (480x640) — intrinsics are already scaled to this resolution
        if rgb.shape[0] != dh or rgb.shape[1] != dw:
            from PIL import Image as _PILImage
            rgb = np.array(_PILImage.fromarray(rgb).resize((dw, dh), _PILImage.BILINEAR))
        raw_intrinsic = batch['head_intrinsic'][idx]
        if torch.is_tensor(raw_intrinsic) and raw_intrinsic.shape == (3, 3):
            cam_intrinsic = {
                'fx': raw_intrinsic[0, 0].item(), 'fy': raw_intrinsic[1, 1].item(),
                'ppx': raw_intrinsic[0, 2].item(), 'ppy': raw_intrinsic[1, 2].item(),
                'cx': raw_intrinsic[0, 2].item(), 'cy': raw_intrinsic[1, 2].item(),
                'z_min': 0.05, 'z_max': 10.0,  # wide range for robot scenes
            }
        else:
            cam_intrinsic = raw_intrinsic
            cam_intrinsic['cx'] = cam_intrinsic['ppx']; cam_intrinsic['cy'] = cam_intrinsic['ppy']
            cam_intrinsic.setdefault('z_min', 0.05)
            cam_intrinsic.setdefault('z_max', 10.0)

        def _get_r0(pred_masks, image, depth, fx, fy, cx, cy):
            # Resize mask to depth spatial dims so indices are in-bounds for depth lookup
            depth_hw = depth.shape[-2], depth.shape[-1]
            resized_mask = F.interpolate(pred_masks.unsqueeze(0), size=depth_hw, mode="nearest")[0, 0]

            mask_np = (resized_mask > 0.5).numpy().astype(np.uint8)
            if mask_np.sum() == 0:
                return None

            # Pole of inaccessibility: deepest in-mask pixel (argmax of distance transform).
            # Always lies inside the mask, even for concave / disconnected shapes.
            dist = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5)
            y_center, x_center = (int(c) for c in np.unravel_index(int(dist.argmax()), dist.shape))

            z = depth.squeeze(0)[y_center, x_center]
            if z <= 0 or not np.isfinite(z):
                return None
            if float(z) > 100.0:
                z = z / 1000.0

            point_3d = np.array([(x_center - cx) * z / fx, (y_center - cy) * z / fy, z])
            return point_3d
        spline_params = {"r0": None, "ctrl": None}


        idx = 0

        md = pred_results.get("motion_directions")
        if md is None or md.shape[0] == 0:
            return {}
        r0 = _get_r0(pred_results["masks"][idx].detach().cpu(), x, depth, cam_intrinsic['fx'], cam_intrinsic['fy'], cam_intrinsic['cx'], cam_intrinsic['cy'])
        spline_params['r0'] = r0
        # pred_results["motion_directions"] is (K, 3*num_predicted_ctrl); take first prediction
        spline_params['ctrl'] = md[0].detach().cpu()
        r_curve, R_curve = compute_spline_curve(spline_params)
        fig_pred = plotly_pointcloud_and_curve(depth.squeeze(0), rgb, cam_intrinsic=cam_intrinsic, r_curve=r_curve, R_curve=R_curve)

        # GT may be either Bezier ctrl (D,) or pre-sampled curve (T, 3); dispatch by ndim.
        gt_motion = batch['find_target'].motion_directions[idx].detach().cpu()
        r0_gt = _get_r0(gt_results["masks"][idx].detach().cpu(), x, depth, cam_intrinsic['fx'], cam_intrinsic['fy'], cam_intrinsic['cx'], cam_intrinsic['cy'])
        if gt_motion.ndim == 2 and gt_motion.shape[-1] == 3:
            r_curve_gt, R_curve_gt = _curve_frames_from_points(gt_motion.to(torch.float64), r0_gt)
        else:
            spline_params_gt = {"r0": r0_gt, "ctrl": gt_motion}
            r_curve_gt, R_curve_gt = compute_spline_curve(spline_params_gt)
        fig_gt = plotly_pointcloud_and_curve(depth.squeeze(0), rgb, cam_intrinsic=cam_intrinsic, r_curve=r_curve_gt, R_curve=R_curve_gt)


        wandb_images = {
            "pointcloud_and_spline_curve_pred": wandb.Plotly(fig_pred),
            "pointcloud_and_spline_curve_gt": wandb.Plotly(fig_gt),
        }

        return wandb_images
