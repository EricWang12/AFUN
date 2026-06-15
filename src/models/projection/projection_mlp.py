import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionMLP(nn.Module):
    """
    Projects features from ℝ^N → ℝ^M with an optional hidden layer.
    Args
    ----
    in_dim  : int — N,  input feature dimension
    out_dim : int — M,  output feature dimension
    hidden  : int | None — size of hidden layer; if None, uses a single Linear
    dropout : float — dropout rate applied *after* the hidden activation
    """
    def __init__(self, in_dim: int, out_dim: int, hidden: int | None = None,
                 dropout: float = 0.0):
        super().__init__()

        if hidden is None:                    # simplest “linear head”
            self.net = nn.Linear(in_dim, out_dim, bias=True)
        else:                                 # 2-layer MLP: N → hidden → M
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden, bias=False),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim, bias=True)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N) or (B, *, N) tensor of input features.
        Returns (B, M) or (B, *, M) tensor in same batch/sequence shape.
        """
        # Flatten everything except the last dim, run MLP, then reshape back.
        original_shape = x.shape[:-1]
        x = self.net(x.view(-1, x.size(-1)))
        return x.view(*original_shape, -1)
