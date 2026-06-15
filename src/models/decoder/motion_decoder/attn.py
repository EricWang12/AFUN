from typing import Optional

import torch
import torch.nn as nn

class SelfAttentionBlock(nn.Module):
    """
    Standard Transformer block (self-attn + FFN) on x: (B, N, d).
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be (B,N,dim). Got {tuple(x.shape)}")

        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(attn_out)

        x = x + self.ff(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention + FFN:
      x (B, Nq, q_dim) attends to kv (B, Nk, kv_dim) and returns updated x.

    Uses torch.nn.MultiheadAttention with kdim/vdim to support kv_dim != q_dim.
    """

    def __init__(
        self,
        q_dim: int = 256,
        kv_dim: int = 4096,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        pre_norm_kv: bool = True,
    ) -> None:
        super().__init__()
        if q_dim % num_heads != 0:
            raise ValueError(f"q_dim ({q_dim}) must be divisible by num_heads ({num_heads}).")

        self.norm_q = nn.LayerNorm(q_dim)
        self.norm_kv = nn.LayerNorm(kv_dim) if pre_norm_kv else nn.Identity()

        self.attn = nn.MultiheadAttention(
            embed_dim=q_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
            kdim=kv_dim,
            vdim=kv_dim,
        )
        self.drop = nn.Dropout(dropout)

        self.norm_ff = nn.LayerNorm(q_dim)
        hidden = int(q_dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(q_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, q_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be (B,Nq,q_dim). Got {tuple(x.shape)}")
        if kv.ndim != 3:
            raise ValueError(f"kv must be (B,Nk,kv_dim). Got {tuple(kv.shape)}")

        q = self.norm_q(x)
        kvn = self.norm_kv(kv)

        attn_out, _ = self.attn(query=q, key=kvn, value=kvn, need_weights=False)
        x = x + self.drop(attn_out)

        x = x + self.ff(self.norm_ff(x))
        return x


class GeoAttentionBlock(nn.Module):
    """
    Cross-attention + FFN with optional additive attention bias for
    mask-guided spatial attention on geo_tokens.

    Output projection is zero-initialized so the block contributes nothing
    at init, preserving compatibility with pre-trained checkpoints.
    """

    def __init__(
        self,
        q_dim: int = 256,
        kv_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if q_dim % num_heads != 0:
            raise ValueError(f"q_dim ({q_dim}) must be divisible by num_heads ({num_heads}).")

        self.num_heads = num_heads

        self.norm_q = nn.LayerNorm(q_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=q_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
            kdim=kv_dim,
            vdim=kv_dim,
        )
        self.drop = nn.Dropout(dropout)

        self.norm_ff = nn.LayerNorm(q_dim)
        hidden = int(q_dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(q_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, q_dim),
            nn.Dropout(dropout),
        )

        # Zero-init output projections so this block is a no-op at init.
        # Both attention output and FFN output must be zeroed for true no-op.
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.ff[3].weight)  # last Linear in FFN
        nn.init.zeros_(self.ff[3].bias)

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, Nq, q_dim)      instance queries
            kv:        (B, Nk, kv_dim)      geo_tokens
            attn_bias: (B, Nq, Nk)          optional additive attention bias
        """
        if x.ndim != 3:
            raise ValueError(f"x must be (B,Nq,q_dim). Got {tuple(x.shape)}")
        if kv.ndim != 3:
            raise ValueError(f"kv must be (B,Nk,kv_dim). Got {tuple(kv.shape)}")

        q = self.norm_q(x)
        kvn = self.norm_kv(kv)

        # Reshape bias for nn.MultiheadAttention: (B*num_heads, Nq, Nk)
        mask = None
        if attn_bias is not None:
            B = x.shape[0]
            mask = (
                attn_bias
                .unsqueeze(1)                                          # (B, 1, Nq, Nk)
                .expand(-1, self.num_heads, -1, -1)                    # (B, H, Nq, Nk)
                .reshape(B * self.num_heads, x.shape[1], kv.shape[1])  # (B*H, Nq, Nk)
            )
            mask = mask.to(dtype=q.dtype)

        attn_out, _ = self.attn(
            query=q, key=kvn, value=kvn,
            attn_mask=mask, need_weights=False,
        )
        x = x + self.drop(attn_out)

        x = x + self.ff(self.norm_ff(x))
        return x
