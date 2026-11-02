"""Train AFUN with a Stage 1, 2, or 3 Hydra configuration."""
from __future__ import annotations

import logging
import os
import random
from datetime import timedelta
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_stage3")
def main(cfg: DictConfig):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo",
                                timeout=timedelta(minutes=30))
    seed = int(cfg.get("seed", 354))
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed)
    output_dir = HydraConfig.get().runtime.output_dir
    if dist.is_initialized():
        paths = [output_dir]
        dist.broadcast_object_list(paths, src=0)
        output_dir = paths[0]
    if rank:
        logging.getLogger().setLevel(logging.WARNING)
    tracking = False
    try:
        selected = cfg.train.get("resume_path") or cfg.train.get("ckpt_path")
        if selected and not Path(selected).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {selected}")
        if rank == 0:
            logger.info("Stage %d | output: %s", cfg.stage, output_dir)
        data = instantiate(cfg.data)
        loader = data.build_dataloader()
        from src.models.build import build_model
        model = build_model(cfg, device, alignment=cfg.stage == 1)
        trainer = instantiate(cfg.train, device=str(device), save_dir=output_dir)
        trainer.build(model, report_trainable=True)
        if rank == 0 and cfg.train.use_wandb:
            import wandb
            wandb.init(project="AFUN", name=Path(output_dir).name,
                       config=OmegaConf.to_container(cfg, resolve=True))
            tracking = True
        trainer.fit(loader)
        if not trainer.save_every or trainer.global_step % trainer.save_every:
            trainer.save()  # the last periodic checkpoint otherwise already covers this step
    finally:
        if tracking:
            import wandb
            wandb.finish()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
