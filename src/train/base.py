"""Checkpoint handling shared by all training stages."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.utils.checkpoint import load_weights

logger = logging.getLogger(__name__)


class TrainerBase:
    def _is_main(self) -> bool:
        return not dist.is_initialized() or dist.get_rank() == 0

    def _unwrap_model(self):
        return self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

    def load_model(self, path: str, *, resume: bool = False) -> None:
        checkpoint = load_weights(self._unwrap_model(), path, allow_new_motion=not resume)
        if resume:
            if "optimizer" not in checkpoint or "global_step" not in checkpoint:
                raise ValueError("Resume requires a training checkpoint with optimizer state and global_step.")
            saved_names = checkpoint.get("optimizer_param_names")
            if saved_names is not None and saved_names != self.param_names:
                raise ValueError("Resume checkpoint has different trainable parameters; use train.ckpt_path to initialize.")
            self.optim.load_state_dict(checkpoint["optimizer"])
            self.global_step = int(checkpoint["global_step"])
            self.epoch = int(checkpoint.get("epoch", 0))
            self.batch_in_epoch = int(checkpoint.get("batch_in_epoch", 0))
            self.best_loss = float(checkpoint.get("best_loss", float("inf")))
        logger.info("%s %s (step %d)", "Resumed" if resume else "Initialized from", path, self.global_step)

    def save(self, save_best: bool = False) -> None:
        """Atomically save trainable weights and optimizer state on rank zero."""
        if not self._is_main():
            return
        model = self._unwrap_model()
        names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if name in names}
        payload = {
            "format_version": 1, "model": state,
            "optimizer": self.optim.state_dict(),
            "optimizer_param_names": self.param_names,
            "global_step": self.global_step, "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch, "best_loss": self.best_loss,
        }
        name = "model_best.pt" if save_best else f"model_step_{self.global_step}.pt"
        path = Path(self.save_dir) / name
        temporary = path.with_suffix(".pt.tmp")
        try:
            torch.save(payload, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        logger.info("Saved %s", path)
        if self.training_cfg.get("rolling_ckpt", False) and not save_best:
            checkpoints = sorted(Path(self.save_dir).glob("model_step_*.pt"),
                                 key=lambda p: int(p.stem.rsplit("_", 1)[1]))
            for old in checkpoints[:-3]:
                old.unlink()
