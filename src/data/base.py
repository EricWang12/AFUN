from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
import pathlib
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class DatasetBase(Dataset, ABC):
    """Dataset paths, collation, and per-rank data-loader construction."""

    def __init__(
        self,
        *,
        data_dir: str | pathlib.Path,
        split: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.data_dir = self.as_path(data_dir) if data_dir is not None else None
        self.split = split
        self.dtype = dtype

        # Training progress hook (default: plain int). Some datasets override this with
        # a shared-memory tensor for multi-worker behavior.
        self._global_step: int = 0

    def set_progress(self, global_step: int) -> None:
        """Optional hook called by training loops to communicate progress to the dataset."""
        self._global_step = int(global_step)

    @staticmethod
    @abstractmethod
    def collate_function(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Subclasses must implement this static method to collate a batch of samples.
        """
        pass


    @staticmethod
    def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def get_collate_defaults(self) -> Dict[str, Any]:
        """
        Hook for datasets that need fixed collate kwargs in addition to loader_cfg.collate_cfg.
        """
        return {}

    def get_collate_cfg(self) -> Dict[str, Any]:
        collate_cfg = self._cfg_get(getattr(self, "loader_cfg", None), "collate_cfg", None)
        if collate_cfg is None:
            return {}
        if isinstance(collate_cfg, dict):
            return dict(collate_cfg)
        return {key: collate_cfg[key] for key in collate_cfg}

    def get_collate_fn(self) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        collate_kwargs = {
            **self.get_collate_defaults(),
            **self.get_collate_cfg(),
        }
        if not collate_kwargs:
            return self.collate_function
        return partial(self.collate_function, **collate_kwargs)

    def get_dataloader_kwargs(self) -> Dict[str, Any]:
        """
        Hook for datasets that need extra DataLoader kwargs without reimplementing the builder.
        """
        return {}

    def build_dataloader(self, **kwargs: Any) -> DataLoader:
        loader_cfg = getattr(self, "loader_cfg", None)
        batch_size, sampler, shuffle = self.resolve_distributed_loader(
            batch_size=int(self._cfg_get(loader_cfg, "batch_size", 1)),
            shuffle=bool(self._cfg_get(loader_cfg, "shuffle", False)),
            seed=int(self._cfg_get(loader_cfg, "seed", 0)),
        )
        num_workers = int(self._cfg_get(loader_cfg, "num_workers", 0))
        dataloader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "sampler": sampler,
            "num_workers": num_workers,
            "pin_memory": bool(self._cfg_get(loader_cfg, "pin_memory", False)),
            "collate_fn": self.get_collate_fn(),
        }
        if num_workers > 0:
            persistent = bool(self._cfg_get(loader_cfg, "persistent_workers", False))
            if persistent:
                dataloader_kwargs["persistent_workers"] = True
            prefetch = int(self._cfg_get(loader_cfg, "prefetch_factor", 2))
            if prefetch != 2:
                dataloader_kwargs["prefetch_factor"] = prefetch
            mp_ctx = self._cfg_get(loader_cfg, "multiprocessing_context", None)
            if mp_ctx is not None:
                dataloader_kwargs["multiprocessing_context"] = str(mp_ctx)
        dataloader_kwargs.update(self.get_dataloader_kwargs())
        dataloader_kwargs.update(kwargs)
        # Worker-only options were chosen from loader_cfg above, but a caller can
        # still override num_workers (e.g. num_workers=0 when debugging). Drop the
        # options DataLoader rejects without workers.
        if int(dataloader_kwargs.get("num_workers", 0)) == 0:
            for k in ("persistent_workers", "prefetch_factor", "multiprocessing_context"):
                dataloader_kwargs.pop(k, None)
        return DataLoader(self, **dataloader_kwargs)

    def maybe_distributed_sampler(self, *, shuffle: bool, seed: int = 0) -> Optional[DistributedSampler]:
        """
        Build a DistributedSampler only when torch.distributed is active with >1 process.
        """
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return None
        if torch.distributed.get_world_size() <= 1:
            return None
        return DistributedSampler(self, shuffle=shuffle, seed=seed)

    def resolve_distributed_loader(self, *, batch_size: int, shuffle: bool, seed: int = 0) -> tuple[int, Optional[DistributedSampler], bool]:
        """
        Resolve loader settings for DDP.

        `batch_size` is PER RANK: each rank loads this many samples, so the
        effective global batch is `batch_size * world_size * grad_accumulation_steps`.
        """
        sampler = self.maybe_distributed_sampler(shuffle=shuffle, seed=seed)
        if sampler is None:
            return int(batch_size), None, shuffle

        per_rank_batch_size = batch_size
        return per_rank_batch_size, sampler, False
    

    @property
    def global_step(self) -> int:
        return int(self._global_step)

    @staticmethod
    def as_path(p: str | pathlib.Path) -> pathlib.Path:
        return p if isinstance(p, pathlib.Path) else pathlib.Path(p)

    @staticmethod
    def require_exists(path: str | pathlib.Path, *, what: str = "path") -> pathlib.Path:
        p = DatasetBase.as_path(path)
        if not p.exists():
            raise FileNotFoundError(f"{what} not found: {p}")
        return p

