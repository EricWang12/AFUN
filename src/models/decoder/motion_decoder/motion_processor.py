# motion_processor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decoder.motion_decoder.attn import (
    SelfAttentionBlock,
    CrossAttentionBlock,
    GeoAttentionBlock,
)


def cosine_direction_loss(
    pred_dir: torch.Tensor,
    target_dir: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Cosine loss for direction vectors.

    pred_dir / target_dir:
      - (B, 3) or (B, N, 3)
    Returns:
      - scalar if reduction in {"mean","sum"}, else per-sample (B,) or per-token (B,N)
    """
    if pred_dir.shape != target_dir.shape:
        raise ValueError(
            f"Shape mismatch: pred {tuple(pred_dir.shape)} vs target {tuple(target_dir.shape)}"
        )
    if pred_dir.size(-1) != 3:
        raise ValueError(f"Last dim must be 3. Got {tuple(pred_dir.shape)}")

    pred = F.normalize(pred_dir, dim=-1, eps=eps)
    targ = F.normalize(target_dir, dim=-1, eps=eps)

    cos_sim = (pred * targ).sum(dim=-1)  # (B,) or (B,N)
    loss = 1.0 - cos_sim

    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unknown reduction: {reduction}")



def cosine_direction_loss_sparse(
    pred_dir_dense: torch.Tensor,            # (B, N, 3)
    target_dir_sparse: torch.Tensor,         # (M, 3)
    indices: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute cosine direction loss only on selected (batch, token) indices.

    indices example:
      (batch_idx, token_idx, None)
        batch_idx: (M,)
        token_idx: (M,)
    """
    if pred_dir_dense.ndim != 3 or pred_dir_dense.size(-1) != 3:
        raise ValueError(f"pred_dir_dense must be (B,N,3). Got {tuple(pred_dir_dense.shape)}")
    if target_dir_sparse.ndim != 2 or target_dir_sparse.size(-1) != 3:
        raise ValueError(f"target_dir_sparse must be (M,3). Got {tuple(target_dir_sparse.shape)}")

    batch_idx, token_idx, _ = indices
    if batch_idx is None or token_idx is None:
        raise ValueError("indices must provide (batch_idx, token_idx, _).")

    if batch_idx.ndim != 1 or token_idx.ndim != 1:
        raise ValueError(f"batch_idx and token_idx must be 1D. Got {batch_idx.ndim}D and {token_idx.ndim}D.")
    if batch_idx.numel() != token_idx.numel():
        raise ValueError("batch_idx and token_idx must have the same length.")
    M = batch_idx.numel()
    if target_dir_sparse.size(0) != M:
        raise ValueError(f"target_dir_sparse first dim must match indices length M={M}.")

    # Ensure indices are on same device
    device = pred_dir_dense.device
    batch_idx = batch_idx.to(device=device)
    token_idx = token_idx.to(device=device)

    # Bounds checks (cheap and helps catch silent bugs)
    B, N, _c = pred_dir_dense.shape
    if (batch_idx.min() < 0) or (batch_idx.max() >= B):
        raise ValueError(f"batch_idx out of bounds for B={B}: [{int(batch_idx.min())}, {int(batch_idx.max())}]")
    if (token_idx.min() < 0) or (token_idx.max() >= N):
        raise ValueError(f"token_idx out of bounds for N={N}: [{int(token_idx.min())}, {int(token_idx.max())}]")

    # Gather predicted directions at sparse locations -> (M, 3)
    pred_sparse = pred_dir_dense[batch_idx, token_idx, :]

    # Normalize both and compute loss (M,)
    pred_n = F.normalize(pred_sparse, dim=-1, eps=eps)
    targ_n = F.normalize(target_dir_sparse.to(device=device), dim=-1, eps=eps)
    cos_sim = (pred_n * targ_n).sum(dim=-1)  # (M,)
    loss_vec = 1.0 - cos_sim                # (M,)


    # unweighted reductions
    if reduction == "none":
        return loss_vec
    if reduction == "sum":
        return loss_vec.sum()
    if reduction == "mean":
        return loss_vec.mean()
    raise ValueError(f"Unknown reduction: {reduction}")


# ---------------------------------------------------------------------------
# MaskBiasComputer
# ---------------------------------------------------------------------------

class MaskBiasComputer(nn.Module):
    """Convert SAM3 mask logits to additive attention bias for geo cross-attention."""

    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(
        self,
        mask_logits: torch.Tensor,
        geo_spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            mask_logits:       (B, K, H_mask, W_mask) raw logits from SAM3
            geo_spatial_shape: (H_geo, W_geo) target spatial dims
        Returns:
            attn_bias:         (B, K, H_geo * W_geo)
        """
        masks = mask_logits.detach().sigmoid()      # (B, K, H, W) -> [0, 1]
        B, K, H, W = masks.shape
        H_geo, W_geo = geo_spatial_shape

        # Resize: treat (B, K) as batch dim for interpolation
        masks = masks.reshape(B * K, 1, H, W)
        masks = F.interpolate(masks, size=(H_geo, W_geo), mode="bilinear", align_corners=False)
        masks = masks.reshape(B, K, H_geo * W_geo)  # (B, K, N_spatial)

        # Log-space bias: inside mask -> ~0, outside -> ~-14
        return self.alpha * torch.log(masks + 1e-6)


# ---------------------------------------------------------------------------
# BaseMotionProcessor
# ---------------------------------------------------------------------------

@dataclass
class MotionProcessorOutput:
    pred: torch.Tensor  # (B, N, 3), unit-normalized
    pred_dir_raw: Optional[torch.Tensor] = None  # (B, N, 3)


class BaseMotionProcessor(nn.Module):
    """Shared transformer loop: self-attn + semantic cross-attn + geo cross-attn.

    Subclasses define ``head``, ``forward``, and ``compute_loss``.
    """

    def __init__(
        self,
        model_cfg: Dict[str, Any],
        loss_cfg: Dict[str, Any],
    ) -> None:
        super().__init__()

        model_cfg = model_cfg or {}
        self.model_cfg = model_cfg
        self.loss_cfg = loss_cfg

        # Core hyper-params
        self.q_dim = int(model_cfg.get("q_dim", 256))
        kv_dim = int(model_cfg.get("kv_dim", 4096))
        self.num_layers = int(model_cfg.get("num_layers", 6))
        num_heads = int(model_cfg.get("num_heads", 8))
        mlp_ratio = float(model_cfg.get("mlp_ratio", 4.0))
        dropout = float(model_cfg.get("dropout", 0.0))
        attn_dropout = float(model_cfg.get("attn_dropout", 0.0))
        self.head_hidden = int(model_cfg.get("head_hidden", 256))

        # Geo config
        self.enable_geo = bool(model_cfg.get("enable_geo", False))
        self.use_mask_bias = bool(model_cfg.get("use_mask_bias", True))
        geo_dim = int(model_cfg.get("geo_dim", 256))

        # Self-attention blocks
        self.self_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=self.q_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Semantic cross-attention blocks
        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    q_dim=self.q_dim,
                    kv_dim=kv_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    pre_norm_kv=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Geo cross-attention blocks (new, optional)
        if self.enable_geo:
            self.geo_cross_blocks = nn.ModuleList(
                [
                    GeoAttentionBlock(
                        q_dim=self.q_dim,
                        kv_dim=geo_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                    )
                    for _ in range(self.num_layers)
                ]
            )
            if self.use_mask_bias:
                self.mask_bias_computer = MaskBiasComputer()

    def _run_transformer(
        self,
        query_feats: torch.Tensor,
        context_feats: torch.Tensor,
        geo_tokens: Optional[torch.Tensor] = None,
        mask_logits: Optional[torch.Tensor] = None,
        geo_spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Run the shared self-attn + cross-attn (+ geo cross-attn) loop."""
        x = query_feats
        kv = context_feats

        # Pre-compute mask bias once
        attn_bias = None
        if (
            self.enable_geo
            and self.use_mask_bias
            and hasattr(self, "mask_bias_computer")
            and mask_logits is not None
            and geo_spatial_shape is not None
        ):
            attn_bias = self.mask_bias_computer(mask_logits, geo_spatial_shape)

        for i, (sa, ca) in enumerate(zip(self.self_blocks, self.cross_blocks)):
            x = sa(x)
            x = ca(x, kv)
            if (
                self.enable_geo
                and hasattr(self, "geo_cross_blocks")
                and geo_tokens is not None
            ):
                x = self.geo_cross_blocks[i](x, geo_tokens, attn_bias=attn_bias)

        return x


# ---------------------------------------------------------------------------
# MotionDirectionProcessor  (head -> 3D direction)
# ---------------------------------------------------------------------------

class MotionDirectionProcessor(BaseMotionProcessor):
    """
    Inputs:
      - query_feats:   (B, 200, 256)
      - context_feats: (B, 32, 4096)
    Output:
      - pred: (B, N, 3) direction (unit normalized), per token
    """

    def __init__(
        self,
        model_cfg: Dict[str, Any],
        loss_cfg: Dict[str, Any],
    ) -> None:
        super().__init__(model_cfg, loss_cfg)

        self.head = nn.Sequential(
            nn.LayerNorm(self.q_dim),
            nn.Linear(self.q_dim, self.head_hidden),
            nn.GELU(),
            nn.Linear(self.head_hidden, 3),
        )

    def forward(
        self,
        query_feats: torch.Tensor,
        context_feats: torch.Tensor,
        *,
        geo_tokens: Optional[torch.Tensor] = None,
        mask_logits: Optional[torch.Tensor] = None,
        geo_spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> MotionProcessorOutput:
        x = self._run_transformer(
            query_feats, context_feats,
            geo_tokens=geo_tokens,
            mask_logits=mask_logits,
            geo_spatial_shape=geo_spatial_shape,
        )

        pred_raw = self.head(x)                        # (B, N, 3)
        pred = F.normalize(pred_raw, dim=-1, eps=1e-8) # (B, N, 3)

        return MotionProcessorOutput(pred=pred, pred_dir_raw=pred_raw)

    def compute_loss(
        self,
        pred_dir: torch.Tensor,     # (B,N,3) from model
        target_dir: torch.Tensor,  # (M,3)
        indices: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        reduction = self.loss_cfg.get("reduction", "mean")
        weight = self.loss_cfg.get("weight", 1.0)

        loss = cosine_direction_loss_sparse(
            pred_dir_dense=pred_dir,
            target_dir_sparse=target_dir,
            indices=indices,
            reduction=reduction,
        ) * weight

        # Also return cosine similarity for monitoring
        batch_idx, token_idx, _ = indices
        pred_sparse = pred_dir[batch_idx, token_idx, :]
        pred_n = F.normalize(pred_sparse, dim=-1, eps=1e-8)
        targ_n = F.normalize(target_dir.to(pred_dir.device), dim=-1, eps=1e-8)
        cos_sim_vec = (pred_n * targ_n).sum(dim=-1)  # (M,)

        cos_sim = cos_sim_vec if reduction == "none" else (
            cos_sim_vec.sum() if reduction == "sum" else cos_sim_vec.mean()
        )
        return {"loss": loss, "cos_sim": cos_sim}
