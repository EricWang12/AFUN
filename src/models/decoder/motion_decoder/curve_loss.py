from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


class CurveLoss:
    """Point matching loss: L1 between pre-sampled curve points.

    Each processor samples its own curves via ``sample_curve`` and passes
    the resulting (M, T, 3) tensors here.  This class is intentionally
    *not* an ``nn.Module`` — it has no learnable parameters.

    Reference: Curve-GCN (Ling et al., CVPR 2019), Section 3.2.1.
    """

    def __init__(self, T: int = 16) -> None:
        self.T = T

    def __call__(self, pred_points: Tensor, gt_points: Tensor) -> Tensor:
        """
        Args:
            pred_points: (M, T, 3) predicted curve samples.
            gt_points:   (M, T, 3) ground-truth curve samples.

        Returns:
            Scalar mean L1 loss over all points and coordinates.
        """
        return F.l1_loss(pred_points, gt_points)
