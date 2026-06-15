"""Depth -> geo_tokens via a frozen Sonata point-cloud encoder.

Drop-in replacement for ``DFormerV2GeoEncoder`` (same forward signature,
same ``(geo_tokens, geo_spatial_shape)`` output). Unprojects depth to a
point cloud, runs Sonata, scatters voxel features back to a 2D grid.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _unproject(depth: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
    """Unproject a depth map to 3D point cloud using pinhole intrinsics.

    Args:
        depth:     (B, 1, H, W) raw metric depth (camera frame)
        intrinsic: (B, 3, 3)    per-batch intrinsic matrix, scaled to (H, W)
    Returns:
        points:    (B, 3, H, W) XYZ in camera frame
    """
    B, _, H, W = depth.shape
    dev, dtype = depth.device, depth.dtype

    v, u = torch.meshgrid(
        torch.arange(H, device=dev, dtype=dtype),
        torch.arange(W, device=dev, dtype=dtype),
        indexing="ij",
    )
    u = u.unsqueeze(0)
    v = v.unsqueeze(0)

    fx = intrinsic[:, 0, 0].view(B, 1, 1)
    fy = intrinsic[:, 1, 1].view(B, 1, 1)
    cx = intrinsic[:, 0, 2].view(B, 1, 1)
    cy = intrinsic[:, 1, 2].view(B, 1, 1)

    z = depth.squeeze(1)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return torch.stack([x, y, z], dim=1)


def _estimate_normals(points: torch.Tensor) -> torch.Tensor:
    """Per-pixel surface normals via finite-difference cross product.

    Args:
        points:  (B, 3, H, W) XYZ
    Returns:
        normals: (B, 3, H, W) unit normals. Last row and column are zero
                 (no neighbors available).
    """
    B, _, H, W = points.shape
    dx = points[..., :-1, 1:] - points[..., :-1, :-1]
    dy = points[..., 1:, :-1] - points[..., :-1, :-1]
    n = torch.cross(dx, dy, dim=1)
    n = F.normalize(n, dim=1, eps=1e-8)
    out = points.new_zeros(B, 3, H, W)
    out[..., :H - 1, :W - 1] = n
    return out


def _pack_valid(
    points: torch.Tensor,   # (B, 3, H, W)
    rgb: torch.Tensor,      # (B, 3, H, W)
    normals: torch.Tensor,  # (B, 3, H, W)
    valid: torch.Tensor,    # (B, H, W) bool
) -> List[Dict[str, torch.Tensor]]:
    """Gather valid pixels per batch item into (N_b, 3) tensors.

    Returns:
        list of length B, each item a dict with keys:
          "coord"     : (N_b, 3)
          "color"     : (N_b, 3)
          "normal"    : (N_b, 3)
          "pixel_idx" : (N_b,) flat index v*W + u in [0, H*W)
    """
    B, _, H, W = points.shape
    items: List[Dict[str, torch.Tensor]] = []
    for b in range(B):
        flat_mask = valid[b].reshape(-1)
        flat_idx = flat_mask.nonzero(as_tuple=False).squeeze(1)
        xyz = points[b].permute(1, 2, 0).reshape(-1, 3)[flat_idx]
        col = rgb[b].permute(1, 2, 0).reshape(-1, 3)[flat_idx]
        nrm = normals[b].permute(1, 2, 0).reshape(-1, 3)[flat_idx]
        items.append({
            "coord": xyz.contiguous(),
            "color": col.contiguous(),
            "normal": nrm.contiguous(),
            "pixel_idx": flat_idx,
        })
    return items


class SonataGeoEncoder(nn.Module):
    """Frozen Sonata + trainable linear head producing geo_tokens for motion processor.

    Forward signature matches ``DFormerV2GeoEncoder`` so this is a drop-in
    replacement at the ``depth_encoder:`` config slot.
    """

    def __init__(
        self,
        ckpt_path: str = "facebook/sonata",
        out_dim: int = 256,
        geo_spatial_shape: Tuple[int, int] = (30, 40),
        grid_size: float = 0.02,
        upcast_levels: int = 1,
        enable_fourier_encode: bool = True,
        max_depth: float = 10000.0,
        min_valid_points: int = 64,
        depth_scale: float = 1.0,  # multiply depth by this to reach meters (0.001 for mm-scale datasets like robot)
        **_ignored,  # absorbs DFormer-only keys leaked by Hydra dict-merge (e.g. pretrained_path, feature_stage, variant)
    ) -> None:
        super().__init__()
        # Deferred imports — keeps helper-only tests cheap.
        from src.models.encoder.sonata import sonata
        try:
            import flash_attn  # noqa: F401
            _has_flash = True
        except ImportError:
            _has_flash = False

        self.out_dim = int(out_dim)
        self.geo_spatial_shape = tuple(geo_spatial_shape)
        self.grid_size = float(grid_size)
        self.upcast_levels = int(upcast_levels)
        self.enable_fourier_encode = bool(enable_fourier_encode)
        self.max_depth = float(max_depth)
        self.min_valid_points = int(min_valid_points)
        self.depth_scale = float(depth_scale)

        # Load Sonata backbone (mirrors SonataWrapper.load_model at
        # src/models/encoder/sonata_wrapper.py:113-123).
        if _has_flash:
            self.sonata = sonata.load("sonata", repo_id=ckpt_path)
        else:
            self.sonata = sonata.load(
                "sonata",
                repo_id=ckpt_path,
                custom_config=dict(enc_patch_size=[1024] * 5, enable_flash=False),
            )
        self.transform = sonata.transform.default_torch()

        # Compute sonata output feature dim after upcast — same logic as
        # SonataWrapper.__init__ (sonata_wrapper.py:97-98).
        encs_stages = list(self.sonata.enc.children())
        sonata_feat_dim = sum(
            list(encs_stages[-i - 1].children())[-1].channels
            for i in range(self.upcast_levels + 1)
        )
        self._sonata_feat_dim = sonata_feat_dim

        # Trainable fourier-projection stage.
        #
        # Concatenates a 63-dim Fourier encoding of each voxel's normalized
        # grid coordinate with the Sonata feature, then projects back to
        # ``sonata_feat_dim``. Because this is a small Linear that sits on the
        # inference path (its output feeds the motion processor through
        # ``geo_tokens``), it MUST be trained and saved alongside the rest of
        # the trainable head — otherwise inference re-instantiates it with a
        # different random basis than training used (different RNG state),
        # which silently shifts motion predictions.
        if self.enable_fourier_encode:
            self.fourier_proj: Optional[nn.Linear] = nn.Linear(sonata_feat_dim + 63, sonata_feat_dim)
        else:
            self.fourier_proj = None

        # reduced_grid_size matches SonataWrapper (num_bins=1280, 4 downconvs).
        downconvs = len(encs_stages) - 1
        res_reduction = 2 ** downconvs
        self.reduced_grid_size = int(1280 / res_reduction)

        # Freeze Sonata backbone.
        for p in self.sonata.parameters():
            p.requires_grad = False

        # Trainable head — Linear(D_sonata -> out_dim) + LayerNorm.
        self.proj = nn.Linear(sonata_feat_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def _sonata_forward(self, items): 
        """Run valid per-item dicts through Sonata.

        Propagates coarse-voxel feats back up to the FINE (GridSample) voxel grid
        via the recorded pooling_inverses so that the per-item ``inverse`` captured
        at voxelization can be used directly for scatter-to-pixel.

        Args:
            items: list of dicts with keys "coord", "color", "normal" (all (N_b, 3)).

        Returns:
            per_item_vox_feats: list of (N_fine_b, D_sonata) tensors. Each row is
                                the feat for that fine voxel (propagated from its
                                coarse ancestor). Same order as ``items``.
            per_item_inverse:   list of (N_b,) int tensors mapping original-point
                                idx → fine-voxel idx *within that item*.
        """
        from src.models.encoder.sonata import sonata
        from src.models.encoder.sonata_wrapper import fourier_encode_vector

        # Voxelize each item; capture per-item fine-grid inverse before collate.
        voxel_dicts = [self.transform(it) for it in items]
        # inverse_fine[b]: (N_b,) maps original-point idx -> fine-voxel idx in [0, N_fine_b).
        per_item_inverse = [d["inverse"] for d in voxel_dicts]

        # Collate and run Sonata.
        batched = sonata.data.collate_fn(voxel_dicts)
        device = next(self.sonata.parameters()).device
        for k in list(batched.keys()):
            v = batched[k]
            if isinstance(v, torch.Tensor):
                batched[k] = v.to(device, non_blocking=True)

        point = self.sonata(batched)

        # Upcast loop WITH concat — matches SonataWrapper.forward (sonata_wrapper.py:140-145).
        # Runs self.upcast_levels times; feat dim grows as configured stages are concatenated.
        for _ in range(self.upcast_levels):
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent

        # Optional frozen fourier projection (mirrors sonata_wrapper.py:150-156).
        # Done at the CURRENT (coarse) stage — grid_coord lives here.
        context = point.feat  # (sum N_coarse_vox, D_sonata)
        if self.enable_fourier_encode and self.fourier_proj is not None:
            coords = point["grid_coord"]
            coords_n = coords / (self.reduced_grid_size - 1)
            encoded = fourier_encode_vector(coords_n)
            context = torch.cat([context, encoded], dim=-1)
            context = self.fourier_proj(context)

        # Continue upcasting (without concat) back to the FINE voxel grid. Each step
        # propagates the coarse feat vector to every fine-grid voxel that pools to it:
        #   parent.feat[i] = context[pooling_inverse[i]]
        # Replace ``context`` with the propagated parent-level feats. When the loop
        # ends, ``point`` sits at the fine grid (same N_vox as GridSample output).
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            # inverse has shape (N_parent,) with values in [0, N_child). Gather.
            context = context[inverse]
            point = parent

        # Now ``context`` has shape (sum N_fine_vox, D_sonata) and ``point["batch"]``
        # is the fine-grid batch index.
        batch_idx = point["batch"]
        num_items = len(items)
        per_item_vox_feats = [context[batch_idx == b] for b in range(num_items)]
        return per_item_vox_feats, per_item_inverse

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        intrinsic: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
            rgb:       (B, 3, H, W)        SAM3-normalized image (mean=0.5, std=0.5)
            depth:     (B, 1, H_d, W_d)    raw metric depth, scale matches ``intrinsic``
            intrinsic: (B, 3, 3)           pinhole intrinsics at (H_d, W_d)
        Returns:
            geo_tokens:        (B, H_geo*W_geo, out_dim)
            geo_spatial_shape: (H_geo, W_geo)
        """
        if intrinsic is None:
            raise ValueError("SonataGeoEncoder requires `intrinsic` (B, 3, 3).")

        B, _, H_d, W_d = depth.shape
        device = depth.device
        H_geo, W_geo = self.geo_spatial_shape

        # Step 1: Prep.
        # Clamp in dataset units first, then scale to meters (Sonata was trained on meter-scale scans;
        # the grid_size=0.02 m voxel quantization would overflow if fed mm-scale points).
        depth = depth.clamp(min=0.0, max=self.max_depth) * self.depth_scale
        if rgb.shape[-2:] != (H_d, W_d):
            rgb = F.interpolate(rgb, size=(H_d, W_d), mode="bilinear", align_corners=False)
        # SAM3 uses mean=0.5, std=0.5. Un-normalize to [0, 255] for Sonata's NormalizeColor.
        rgb_unn = (rgb * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0

        # Step 2–3: Unproject + normals.
        points = _unproject(depth, intrinsic.to(device))
        normals = _estimate_normals(points)

        # Step 4: Validity mask.
        valid = (depth.squeeze(1) > 0) & torch.isfinite(depth.squeeze(1))
        # Drop pixels where normals are all-zero (bottom row / right column).
        normal_ok = normals.abs().sum(dim=1) > 0
        valid = valid & normal_ok

        # Step 5: Pack per-item.
        items = _pack_valid(points, rgb_unn, normals, valid)

        # Allocate output upfront; fill only valid items.
        geo_out = points.new_zeros(B, self.out_dim, H_geo, W_geo, dtype=torch.float32)

        # Filter items with enough points.
        keep_b = [b for b, it in enumerate(items) if it["coord"].shape[0] >= self.min_valid_points]
        if len(keep_b) == 0:
            geo_tokens = geo_out.flatten(2).permute(0, 2, 1).contiguous()
            return geo_tokens, (H_geo, W_geo)

        kept_items = [{k: items[b][k] for k in ("coord", "color", "normal")} for b in keep_b]

        # Step 6: Sonata forward (under no_grad, eval).
        with torch.no_grad():
            self.sonata.eval()
            per_item_vox_feats, per_item_inverse = self._sonata_forward(kept_items)

        # Step 7: Scatter voxel feats back to pixel grid per item.
        D_sonata = self._sonata_feat_dim
        pixel_feats_all = points.new_zeros(B, D_sonata, H_d * W_d, dtype=torch.float32)
        for local_i, b in enumerate(keep_b):
            vox_feats = per_item_vox_feats[local_i]             # (N_vox_b, D)
            inv = per_item_inverse[local_i].to(device)          # (N_b,)
            pix_idx = items[b]["pixel_idx"].to(device)          # (N_b,)
            pt_feats = vox_feats[inv]                            # (N_b, D)
            pixel_feats_all[b].index_copy_(1, pix_idx, pt_feats.t().to(pixel_feats_all.dtype))

        pixel_feats_all = pixel_feats_all.view(B, D_sonata, H_d, W_d)

        # Step 8: 2D pool to (H_geo, W_geo).
        pooled = F.adaptive_avg_pool2d(pixel_feats_all, output_size=(H_geo, W_geo))
        pooled_flat = pooled.flatten(2).permute(0, 2, 1).contiguous()  # (B, N_spatial, D_sonata)

        # Step 9: Trainable head.
        geo_tokens = self.norm(self.proj(pooled_flat))
        return geo_tokens, (H_geo, W_geo)
