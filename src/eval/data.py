"""Benchmark samples, with the queries and ground truth of the paper's evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

SEGMENTATION_SETS = ("handal-mini", "3doi", "handal-easy", "handal-hard", "3doi-easy",
                     "hova500k", "reasonaff", "instruct-part")
MOTION_SETS = ("afun_test", "scenefun3d_test", "robomind2_test")
RAGNET_FILES = {"handal-mini": ("handal", "handal_mini_val.pkl"),
                "3doi": ("3doi", "3doi_val.pkl"),
                "handal-easy": ("handal", "handal_easy_reasoning_val.pkl"),
                "handal-hard": ("handal", "handal_hard_reasoning_val.pkl"),
                "3doi-easy": ("3doi", "3doi_easy_reasoning_val.pkl")}


def segmentation_dataset(name, root):
    """RAGNet's class-name sets use its template; its reasoning sets use "question answer"."""
    root = Path(root)
    if name in RAGNET_FILES:
        from src.data.ragnet import RAGNetDataset
        source, annotation = RAGNET_FILES[name]
        return RAGNetDataset(root / "RAGNet/data/data", source=source, split="val",
                             ann_file=annotation, strict=True,
                             query_template="Please segment the affordance map of {} in this image" if name in ("handal-mini", "3doi") else None)
    if name == "hova500k":
        from src.data.hova500k import HOVA500KDataset
        return HOVA500KDataset(root / "HOVA-500K", split="test")
    if name == "reasonaff":
        from src.data.reasonaff import ReasonAFFDataset
        return ReasonAFFDataset(root / "reasonaff", split="test", strict=True)
    if name == "instruct-part":
        from src.data.instruct_part import InstructPartDataset
        return InstructPartDataset(root / "instruct_part", split="test", strict=True)  # first instruction
    raise ValueError(f"Unknown segmentation dataset: {name}")


def segmentation_samples(name, root, limit=None):
    dataset = segmentation_dataset(name, root)
    for index in range(min(len(dataset), limit or len(dataset))):
        try:
            sample = dataset[index]
        except OSError as error:  # an unreadable file, e.g. RAGNet's empty 3doi/images/EK_frame_0000040462.jpg
            yield {"id": str(index), "error": f"{type(error).__name__}: {error}"}
            continue
        yield {"id": str(index), "query": sample["query"][0], "image": sample["image"], "mask": sample["mask"][0]}


def motion_samples(name, root, limit=None):
    directory = Path(root) / "afun_eval" / name
    with (directory / "manifest.json").open() as handle:
        records = json.load(handle)["samples"]
    for record in records[:limit]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid manifest path: {relative}")
        sample_dir = directory / relative
        with (sample_dir / "trajectory.json").open() as handle:
            trajectory = json.load(handle)
        intrinsic = trajectory["camera_info"]["intrinsics"]
        with Image.open(sample_dir / "obs_frame.png") as image:
            image = np.array(image.convert("RGB"))
        with Image.open(sample_dir / "sam_mask.png") as mask:
            mask = np.array(mask.convert("L")) > 127
        yield {"id": record["path"], "query": record["language"], "image": image, "mask": mask,
               "depth": np.load(sample_dir / "obs_frame_depth.npy", allow_pickle=False),
               "intrinsic": {key: float(intrinsic[key]) for key in ("fx", "fy", "cx", "cy")},
               "trajectory": np.asarray([p["position_3d"] for p in trajectory["trajectory_3d"]])}
