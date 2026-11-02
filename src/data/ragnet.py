"""RAGNet images, part masks, and task instructions."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import DatasetBase
from .sam3_utils import collate_to_sam3

logger = logging.getLogger(__name__)


class RAGNetDataset(DatasetBase):
    """Read the public list annotations and class-indexed evaluation annotations."""

    def __init__(self, data_dir, *, source=("graspnet", "egoobjects", "openx", "rlbench", "handal"),
                 split="train", reasoning=("easy", "hard"), ann_file=None, query_template=None,
                 dtype=torch.float32, strict=True, mask_threshold=0.5, loader_cfg=None):
        super().__init__(data_dir=data_dir, split=split, dtype=dtype)
        self.loader_cfg = loader_cfg
        self.mask_threshold = mask_threshold
        sources = [source] if isinstance(source, str) else list(source)
        variants = [reasoning] if isinstance(reasoning, str) else list(reasoning or [])
        if ann_file and len(sources) != 1:
            raise ValueError("An explicit annotation file requires exactly one source.")
        self.samples = []
        unused = 0
        for name in sources:
            if ann_file:
                annotations = [self.data_dir / ann_file]
            else:
                candidates = [f"{name}_{variant}_reasoning_{split}.pkl" for variant in variants]
                candidates.append(f"{name}_{split}.pkl")
                annotations = [self.data_dir / p for p in dict.fromkeys(candidates) if (self.data_dir / p).is_file()]
            if not annotations:
                raise FileNotFoundError(f"No {name} annotations for split={split} in {self.data_dir}")
            for path in annotations:
                for record in self._records(path):
                    question = record.get("question") or record.get("task_instruction")
                    answer = record.get("answer") or ""
                    if question or answer:
                        # A few records have an empty question; their answer is the instruction.
                        query = f"{question or ''} {answer}".strip()
                    else:
                        # Class-indexed evaluation annotations name the class. Training records
                        # with only an object class (e.g. graspnet_train.pkl) were not used by AFUN.
                        query = record.get("query")
                    if not isinstance(query, str) or not query.strip():
                        unused += 1
                        continue
                    self.samples.append({
                        "image": self._resolve(record["frame_path"]),
                        "mask": self._resolve(record["mask_path"]),
                        "query": query_template.format(query) if query_template else query,
                        "class_id": record.get("_class_id", record.get("graspnet_object_id")),
                        "source": name,
                    })
        if not self.samples:
            raise ValueError(f"No RAGNet samples in {self.data_dir} for split={split}")
        logger.info("RAGNet: %d samples (%s); skipped %d records without an instruction",
                    len(self.samples), split, unused)

    def _resolve(self, value):
        path = Path(value)
        if path.is_absolute():
            return path
        # Public PKLs address images relative to the original repository's data/.
        if path.parts and path.parts[0] == "data":
            path = Path(*path.parts[1:])
        return self.data_dir / path

    @staticmethod
    def _records(path):
        with path.open("rb") as handle:
            records = pickle.load(handle)
        if isinstance(records, list):
            return records
        if not isinstance(records, dict) or "images" not in records or "labels" not in records:
            raise ValueError(f"Unsupported RAGNet annotation format: {path}")
        result = []
        for name, images in records["images"].items():
            masks = records["labels"][name]
            ids = records.get("class_ids", {}).get(name, [None] * len(images))
            for image, mask, class_id in zip(images, masks, ids, strict=True):
                result.append({"frame_path": image, "mask_path": mask, "query": name, "_class_id": class_id})
        return result

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample["image"]) as image:
            image = np.array(image.convert("RGB"))
        with Image.open(sample["mask"]) as mask:
            raw = np.array(mask if sample["class_id"] is not None else mask.convert("L"))
        if sample["class_id"] is not None:
            mask = raw == int(sample["class_id"])
        else:
            mask = raw > (255 * self.mask_threshold if raw.max() > 1 else self.mask_threshold)
        height, width = image.shape[:2]
        mask = Image.fromarray(mask).resize((width, height), Image.Resampling.NEAREST)
        return {"image": torch.from_numpy(image).permute(2, 0, 1).to(self.dtype) / 255,
                "mask": torch.from_numpy(np.array(mask)).unsqueeze(0).to(self.dtype),
                "query": [sample["query"]], "source": sample["source"], "split": self.split}

    @staticmethod
    def collate_function(batch, **kwargs):
        return collate_to_sam3(batch, **kwargs)
