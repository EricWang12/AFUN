"""Training loop for alignment, segmentation, and joint motion prediction."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .base import TrainerBase

logger = logging.getLogger(__name__)


class Trainer(TrainerBase):
    def __init__(self, *, lr, training_cfg, grad_accumulation_steps=1,
                 use_wandb=False, save_dir="outputs", device="cuda",
                 ckpt_path=None, resume_path=None, batch_size=None):
        self.lr = float(lr)
        self.training_cfg = training_cfg
        self.grad_accumulation_steps = int(grad_accumulation_steps)
        self.max_steps = int(training_cfg.max_steps)
        self.log_every = int(training_cfg.get("log_every_step", 25))
        self.save_every = int(training_cfg.get("save_ckpt_every", 1000))
        self.grad_clip = training_cfg.get("grad_clip", 1.0)
        if min(self.grad_accumulation_steps, self.max_steps, self.log_every) < 1 or self.save_every < 0:
            raise ValueError("Steps, accumulation, and logging intervals must be positive; save interval may be zero.")
        self.device = torch.device(device)
        self.save_dir = str(save_dir)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.use_wandb = bool(use_wandb)
        self.ckpt_path, self.resume_path = ckpt_path, resume_path
        self.global_step = self.epoch = self.batch_in_epoch = 0
        self.best_loss = float("inf")

    def build(self, model, model_ckpt=None, report_trainable=False):
        self.model = model.to(self.device)
        if hasattr(model, "llm"):
            # build_model sets the dtypes. Qwen's embedding hook limits updates to the added MetaQuery rows.
            model.llm.llm_model.requires_grad_(False)
            model.llm._get_embed_tokens().weight.requires_grad_(True)
            if self.training_cfg.get("gradient_checkpointing", True):
                model.llm.llm_model.config.use_cache = False
                model.llm.llm_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                model.llm.llm_model.enable_input_require_grads()
        self.param_names = [name for name, p in model.named_parameters() if p.requires_grad]
        self.trainable = [p for p in model.parameters() if p.requires_grad]
        if not self.trainable:
            raise ValueError("The model has no trainable parameters.")
        self.optim = torch.optim.Adam(self.trainable, lr=self.lr)
        # An explicit resume takes precedence over the preceding-stage default.
        if self.resume_path:
            self.load_model(self.resume_path, resume=True)
        elif model_ckpt or self.ckpt_path:
            self.load_model(model_ckpt or self.ckpt_path)
        if dist.is_initialized() and dist.get_world_size() > 1:
            self.model = DDP(model, device_ids=[self.device.index] if self.device.type == "cuda" else None,
                             find_unused_parameters=True)
        if self._is_main():
            count = sum(p.numel() for p in self.trainable)
            logger.info("Training %d tensors (%s parameters, including the gradient-masked embedding table)",
                        len(self.trainable), f"{count:,}")
            if report_trainable:
                (Path(self.save_dir) / "trainable_params.txt").write_text("\n".join(self.param_names) + "\n")

    def step(self, batch, *, accum_steps=1, accum_idx=0):
        if accum_idx == 0:
            self.optim.zero_grad(set_to_none=True)
        self.model.train()
        batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            result = self.model(
                prompt=batch["query"], image=batch["image"],
                image_raw=batch.get("image_unnormalized"), find_target=batch.get("find_target"),
                depth=batch.get("depth"), head_intrinsic=batch.get("head_intrinsic"),
                has_motion=batch.get("has_motion", False),
            )
            loss = result["losses"]
            loss = sum(v.float() for v in loss.values()) if isinstance(loss, dict) else loss.float()
        finite = torch.isfinite(loss.detach()).to(torch.int32)
        if dist.is_initialized():
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            self.optim.zero_grad(set_to_none=True)
            raise FloatingPointError(f"Non-finite loss at step {self.global_step}.")
        (loss / accum_steps).backward()
        if accum_idx + 1 == accum_steps:
            torch.nn.utils.clip_grad_norm_(self.trainable, float(self.grad_clip or float("inf")), error_if_nonfinite=True)
            self.optim.step()
            self.global_step += 1
        metrics = {k: float(v.detach()) if torch.is_tensor(v) else float(v)
                   for k, v in result.get("metrics", {}).items()}
        return {"loss": float(loss.detach()), **metrics}

    def fit(self, loader):
        if len(loader) == 0:
            raise ValueError("The training loader is empty; check dataset paths and splits.")
        if self.batch_in_epoch:
            # A resumed run starts the next epoch's shuffle instead of re-reading the batches it
            # already consumed (Stage 1 epochs are hours long).
            self.epoch += 1
            self.batch_in_epoch = 0
        accum_idx = 0
        window = []
        while self.global_step < self.max_steps:
            sampler = getattr(loader, "sampler", None)
            if hasattr(loader, "set_epoch"):
                loader.set_epoch(self.epoch)
            elif hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.epoch)
            epoch_loss, batches = 0.0, 0
            for index, batch in enumerate(loader):
                metrics = self.step(batch, accum_steps=self.grad_accumulation_steps, accum_idx=accum_idx)
                self.batch_in_epoch = index + 1
                epoch_loss += metrics["loss"]
                batches += 1
                window.append(metrics["loss"])
                accum_idx = (accum_idx + 1) % self.grad_accumulation_steps
                if accum_idx:
                    continue
                if self.global_step % self.log_every == 0:
                    loss = torch.tensor(sum(window) / len(window), device=self.device)
                    if dist.is_initialized():
                        dist.all_reduce(loss)
                        loss /= dist.get_world_size()
                    if self._is_main():
                        logger.info("step %d/%d | loss %.6f", self.global_step, self.max_steps, loss)
                        if self.use_wandb:
                            import wandb
                            wandb.log({"train/loss": float(loss), "train/lr": self.optim.param_groups[0]["lr"]},
                                      step=self.global_step)
                    window.clear()
                if self.save_every and self.global_step % self.save_every == 0:
                    self.save()
                if self.global_step >= self.max_steps:
                    return
            if batches and accum_idx == 0:
                totals = torch.tensor([epoch_loss, batches], dtype=torch.float64, device=self.device)
                if dist.is_initialized():
                    dist.all_reduce(totals)
                average = float(totals[0] / totals[1])
                every = int(self.training_cfg.get("save_best_every_epoch", 0))
                if every > 0 and (self.epoch + 1) % every == 0 and average < self.best_loss:
                    self.best_loss = average
                    self.save(save_best=True)
            self.epoch += 1
            self.batch_in_epoch = 0
