"""Mix homogeneous segmentation and robot batches for Stage 3."""
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from .seg_multi_source import _InterleavedLoader, allocate_workers


class CombinedDataModule:
    def __init__(self, segmentation, robot, mixing):
        weights = dict(mixing.get("weights", {"segmentation": 1, "robot": 1}))
        active = {k: v for k, v in weights.items() if v != 0}
        budget = mixing.get("max_total_workers")
        workers = allocate_workers(int(budget), active) if budget is not None else {}
        loaders = {}
        if "segmentation" in active:
            cfg = OmegaConf.create(segmentation)
            if "segmentation" in workers:
                with open_dict(cfg):
                    cfg.mixing.max_total_workers = workers["segmentation"]
            self.seg_module = instantiate(cfg)
            loaders["segmentation"] = self.seg_module.build_dataloader()
        if "robot" in active:
            self.robot_dataset = instantiate(robot)
            options = {"num_workers": workers["robot"]} if "robot" in workers else {}
            loaders["robot"] = self.robot_dataset.build_dataloader(**options)
        self.train_loader = _InterleavedLoader(
            loaders, weights=active, fixed_steps=mixing.get("fixed_steps", 200),
            seed=mixing.get("seed", 0), motion_sources={"robot"},
        )

    def build_dataloader(self):
        return self.train_loader
