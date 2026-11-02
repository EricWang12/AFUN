"""Visual Genome region captions paired with their full RGB images."""
from __future__ import annotations

import json
import logging

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import resize

from .base import DatasetBase

logger = logging.getLogger(__name__)


class VisualGenomeDataset(DatasetBase):
    def __init__(self, data_dir, annotation_file="region_descriptions.json", loader_cfg=None,
                 max_samples=None, strict=True):
        super().__init__(data_dir=data_dir, split="train", dtype=torch.float32)
        self.loader_cfg = loader_cfg
        self.samples = []
        missing = set()
        with (self.data_dir / annotation_file).open() as handle:
            records = json.load(handle)
        for record in records:
            image_id = record.get("id", record.get("image_id"))
            candidates = [self.data_dir / folder / f"{image_id}.jpg" for folder in ("VG_100K", "VG_100K_2", "images")]
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                missing.add(image_id)
                continue
            for region in record.get("regions", []):
                phrase = str(region.get("phrase", "")).strip()
                if phrase:
                    self.samples.append((path, phrase))
                if max_samples and len(self.samples) >= max_samples:
                    break
            if max_samples and len(self.samples) >= max_samples:
                break
        if missing and strict:
            raise FileNotFoundError(f"Missing {len(missing)} Visual Genome images; extract both image archives.")
        if not self.samples:
            raise ValueError(f"No image-caption pairs found in {self.data_dir}.")
        logger.info("Visual Genome: %d image-caption pairs", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, phrase = self.samples[index]
        with Image.open(path) as image:
            image = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).float() / 255
        return {"image": image, "query": phrase}

    @staticmethod
    def collate_function(batch, image_resolution=1008):
        images = [resize(s["image"], [image_resolution, image_resolution], antialias=True) for s in batch]
        return {"image": (torch.stack(images) - 0.5) / 0.5,
                "image_unnormalized": images, "query": [s["query"] for s in batch]}
