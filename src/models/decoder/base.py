# src/models/encoder/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional
import torch
from torch import nn
from ..model_base import ModelBase
import numpy as np
from typing import Dict, Any

class DecoderBase(ModelBase, ABC):
    projection: nn.Module = torch.nn.Identity()
    
    def __init__(self, freeze: bool = False):
        super().__init__(freeze)
        self.apply_freeze()
    
    # """Raw input -> continuous features."""
    # @abstractmethod
    # def encode(self, raw: Any, **kwargs) -> torch.Tensor:
    #     raise NotImplementedError("Subclasses must implement encode")

    @abstractmethod
    def load_model(self, ckpt_path: str = None, device: str = "cuda", **kwargs):
        raise NotImplementedError("Subclasses must implement load_model")
    
    @abstractmethod
    def forward(self, point_cloud: torch.Tensor, feature: Optional[torch.Tensor] = None, images: Optional[torch.Tensor] = None, normalize: bool = False, **kwargs) -> torch.Tensor:
        """
        Forward pass for the decoder.

        Args:
            point_cloud (torch.Tensor): Input point cloud tensor of B x N x 3
                B is the batch size, N is the number of points, and 3 corresponds to (x, y, z) coordinates.
            feature (Optional[torch.Tensor], optional): Optional per-point feature tensor of shape B x N x F
                where F is the feature dimension. Default is None.
            images (Optional[torch.Tensor], optional): Optional image tensor of shape B x C x H x W
                where C is the number of channels, H and W are height and width. Default is None.
            **kwargs: Additional keyword arguments for specific decoder implementations.

        Returns:
            torch.Tensor: Decoded feature tensor of shape (B, D), where D is the output feature dimension.
        """
        raise NotImplementedError("Subclasses must implement forward")
    
    # @abstractmethod
    def make_wandb_vis(self, batch: Dict[str, Any], model_outputs: Dict[str, Any], *, step: int):
        """
        Make visualization for WandB.
        """
        raise NotImplementedError("Subclasses must implement make_wandb_vis")
    
    @abstractmethod
    def normalize_pc(self, pc : np.ndarray | torch.Tensor):
        """
        Normalize the point cloud to [-1, 1]
        Args:
            pc (np.ndarray | torch.Tensor): Point cloud tensor of shape (B, N, 3)
                B is the batch size, N is the number of points, and 3 corresponds to (x, y, z) coordinates.

        Returns:
            torch.Tensor: Normalized point cloud tensor of shape (B, N, 3)
        """
        raise NotImplementedError("Subclasses must implement normalize_pc")
    
    @staticmethod
    def to_BxNxC(pc):
        """Ensure point cloud is (B, N, C). Accepts (N, C), (B, N, C), or (B, C, N)."""
        is_torch = isinstance(pc, torch.Tensor)
        if pc.ndim == 2:                  # (N, C) -> (1, N, C)
            pc = pc.unsqueeze(0) if is_torch else pc[None, ...]
        elif pc.ndim != 3:
            raise ValueError(f"pc must be 2D or 3D, got {pc.ndim}D.")

        # Heuristic: if middle dim looks like channels (3/6/… ) and last dim doesn’t, assume (B, C, N)
        if pc.shape[1] in (3, 6) and pc.shape[2] not in (3, 6):
            pc = pc.permute(0, 2, 1) if is_torch else np.transpose(pc, (0, 2, 1))
        return pc
        