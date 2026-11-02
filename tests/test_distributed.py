"""Exercise alternating branches, gradient accumulation, and saving on two ranks."""
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch import nn

from src.train.trainer import Trainer


class BranchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Parameter(torch.tensor(0.5))
        self.motion = nn.Parameter(torch.tensor(0.25))

    def forward(self, image, has_motion=False, **kwargs):
        value = self.shared * image
        if has_motion:
            value = value + self.motion * image
        return {"losses": (value - 1).square().mean()}


def worker(rank, directory):
    dist.init_process_group("gloo", init_method=f"file://{directory}/rendezvous", rank=rank,
                            world_size=2, timeout=timedelta(seconds=45))
    try:
        cfg = OmegaConf.create({"max_steps": 2, "save_ckpt_every": 1, "log_every_step": 1})
        run = Trainer(lr=0.01, training_cfg=cfg, save_dir=directory, device="cpu", grad_accumulation_steps=2)
        run.build(BranchModel())
        batches = [{"image": torch.tensor([float(rank + 1)]), "query": ["test"], "has_motion": motion}
                   for motion in (False, True, False, True)]
        run.fit(batches)
        run.save()
        weights = torch.stack([p.detach() for p in run.model.parameters()])
        gathered = [torch.zeros_like(weights) for _ in range(2)]
        dist.all_gather(gathered, weights)
        torch.testing.assert_close(gathered[0], gathered[1])
        assert run.global_step == 2
        dist.barrier()
        checkpoint = Path(directory) / "model_step_2.pt"
        resumed = Trainer(lr=0.01, training_cfg=cfg, save_dir=directory, device="cpu", resume_path=checkpoint)
        resumed.build(BranchModel())
        assert resumed.global_step == 2
        assert resumed.batch_in_epoch == 4
    finally:
        dist.destroy_process_group()


def test_two_rank_training_and_resume(tmp_path):
    mp.spawn(worker, args=(str(tmp_path),), nprocs=2, join=True)
    assert len(list(tmp_path.glob("model_step_*.pt"))) == 2
    assert not list(tmp_path.glob("*.tmp"))
