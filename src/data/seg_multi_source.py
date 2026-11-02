"""Deterministic batch sampling across segmentation datasets."""
from __future__ import annotations

import logging
import math

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def allocate_workers(budget, weights):
    """Distribute at most ``budget`` workers, including a zero-worker budget."""
    if budget < 0:
        raise ValueError("Worker budget must be nonnegative.")
    if not weights:
        return {}
    total = sum(weights.values())
    if total <= 0 or any(not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("Sampling weights must be finite, nonnegative, and have a positive sum.")
    # Give every loader a worker when possible; a loader without workers loads in the training process.
    floor = 1 if budget >= len(weights) else 0
    quotas = {name: (budget - floor * len(weights)) * weight / total for name, weight in weights.items()}
    result = {name: floor + int(value) for name, value in quotas.items()}
    order = sorted(weights, key=lambda name: quotas[name] - int(quotas[name]), reverse=True)
    for name in order[:budget - sum(result.values())]:
        result[name] += 1
    return result


class _InterleavedLoader:
    def __init__(self, loaders, *, fixed_steps=0, weights=None, seed=0, motion_sources=()):
        if not loaders or any(len(loader) == 0 for loader in loaders.values()):
            raise ValueError("Every source must contain at least one batch.")
        self.loaders = loaders
        self.names = list(loaders)
        weights = weights or {}
        values = [weights.get(name, len(getattr(loader, "dataset", loader))) for name, loader in loaders.items()]
        self.probs = torch.tensor(values, dtype=torch.float64)
        if not torch.isfinite(self.probs).all() or (self.probs < 0).any() or self.probs.sum() <= 0:
            raise ValueError("Invalid dataset weights.")
        self.probs /= self.probs.sum()
        self.iter_all = fixed_steps <= 0
        self.total_steps = sum(map(len, loaders.values())) if self.iter_all else int(fixed_steps)
        self.seed, self._epoch = int(seed), 0
        self.motion_sources = set(motion_sources)
        logger.info("Mixed loader: %d batches, sources=%s", self.total_steps, self.names)

    def __len__(self):
        return self.total_steps

    def set_epoch(self, epoch):
        self._epoch = int(epoch)
        for loader in self.loaders.values():
            target = loader if hasattr(loader, "set_epoch") else getattr(loader, "sampler", None)
            if hasattr(target, "set_epoch"):
                target.set_epoch(epoch)

    def _make_source_plan(self, generator):
        if self.iter_all:
            plan = [name for name, loader in self.loaders.items() for _ in range(len(loader))]
            return [plan[i] for i in torch.randperm(len(plan), generator=generator).tolist()]
        return [self.names[i] for i in torch.multinomial(self.probs, self.total_steps, replacement=True, generator=generator).tolist()]

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self._epoch)
        iterators = {}
        for source in self._make_source_plan(generator):
            if source not in iterators:
                iterators[source] = iter(self.loaders[source])
            try:
                batch = next(iterators[source])
            except StopIteration:
                if self.iter_all:
                    raise RuntimeError(f"{source} stopped before its reported length.") from None
                iterators[source] = iter(self.loaders[source])
                batch = next(iterators[source])
            yield {**batch, "source": source, "has_motion": source in self.motion_sources}


class SegMultiSourceDataModule:
    def __init__(self, datasets, mixing):
        weights = dict(mixing.get("weights") or {})
        self.datasets = {
            name: instantiate(cfg) if isinstance(cfg, (dict, DictConfig)) else cfg
            for name, cfg in datasets.items() if weights.get(name, 1) != 0
        }
        weights = {name: weights.get(name, len(ds)) for name, ds in self.datasets.items()}
        budget = mixing.get("max_total_workers")
        workers = allocate_workers(int(budget), weights) if budget is not None else {}
        loaders = {name: ds.build_dataloader(**({"num_workers": workers[name]} if name in workers else {}))
                   for name, ds in self.datasets.items()}
        self.train_loader = _InterleavedLoader(loaders, fixed_steps=mixing.get("fixed_steps", 0),
                                               weights=weights, seed=mixing.get("seed", 0))

    def build_dataloader(self):
        return self.train_loader
