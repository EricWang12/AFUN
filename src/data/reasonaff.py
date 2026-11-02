from __future__ import annotations

import io
import pathlib
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from .base import DatasetBase

try:
    from datasets import Dataset as HFDataset
    from datasets import DatasetDict, load_from_disk
except Exception:  # pragma: no cover
    HFDataset = None  # type: ignore[assignment]
    DatasetDict = None  # type: ignore[assignment]
    load_from_disk = None  # type: ignore[assignment]

from src.data.sam3_utils import collate_to_sam3


QUESTION_TEMPLATE = (
    'Please answer "{Question}" with bboxs and points.'
    "Analyze the functional properties of specific parts of each object in the image and carefully find all "
    "the part(s) that matches the problem."
    "Output the thinking process in <think> </think>, rethinking process in <rethink> </rethink> and final "
    "answer in <answer> </answer> tags."
    "Output the bbox(es) and point(s) and affordance tpye(s) inside the interested object(s) in JSON format."
    "i.e., <think> thinking process here </think>,"
    "<rethink> rethinking process here </rethink>,"
    "<answer>{Answer}</answer>"
)

DEMO_ANSWER = (
    '[{"bbox_2d": [10,100,200,210], "point_2d": [30,110], "affordance": "hold"}, '
    '{"bbox_2d": [225,296,706,786], "point_2d": [302,410], "affordance": "grasp"}]'
)


def _resize_mask_nearest(mask: np.ndarray, *, out_h: int, out_w: int) -> np.ndarray:
    if mask.shape[0] == out_h and mask.shape[1] == out_w:
        return mask
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    return (np.asarray(mask_img.resize((out_w, out_h), resample=Image.NEAREST)) > 0).astype(np.float32)


def _normalize_mask(mask: Any, image_size: tuple[int, int]) -> np.ndarray:
    mask_np = np.asarray(mask).astype(bool)
    target_w, target_h = image_size
    if mask_np.shape != (target_h, target_w):
        mask_np = _resize_mask_nearest(mask_np, out_h=target_h, out_w=target_w) > 0
    return mask_np.astype(np.float32)


def _as_query_list(query: Any) -> List[str]:
    if query is None:
        return []
    if isinstance(query, str):
        q = query.strip()
        return [q] if q else []
    if isinstance(query, (list, tuple)):
        out: List[str] = []
        for item in query:
            out.extend(_as_query_list(item))
        return out
    return [str(query)]


def _build_training_query(problem_text: str) -> str:
    return QUESTION_TEMPLATE.format(
        Question=problem_text.lower().strip('.\"?!'),
        Answer=DEMO_ANSWER,
    )


class ReasonAFFDataset(DatasetBase):
    """
    Loader for the Hugging Face ReasonAFF export saved with `datasets.load_from_disk`.

    Returns the same minimal sample schema used by the other SAM3-compatible datasets:
      - `image`: (3, H, W) float tensor in [0, 1]
      - `mask`:  (1, H, W) float tensor in {0, 1}
      - `query`: non-empty list[str]
    """

    def __init__(
        self,
        data_dir: str | pathlib.Path,
        *,
        split: str = "train",
        split_root: Optional[str] = None,
        query_field: str = "problem",
        query_style: str = "raw",
        dtype: torch.dtype = torch.float32,
        transform=None,
        cache: bool = False,
        strict: bool = False,
        loader_cfg: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if load_from_disk is None:
            raise ModuleNotFoundError(
                "ReasonAFFDataset requires the `datasets` package. "
                "Install it with `pip install datasets` in the active environment."
            )

        if isinstance(data_dir, str):
            data_dir = pathlib.Path(data_dir)

        super().__init__(data_dir=data_dir, split=str(split), dtype=dtype)
        self.data_dir = data_dir
        self.split = str(split)
        self.split_root = None if split_root in (None, "") else str(split_root)
        self.query_field = str(query_field)
        self.query_style = str(query_style)
        self.dtype = dtype
        self.transform = transform
        self.cache = bool(cache)
        self.strict = bool(strict)
        self.loader_cfg = loader_cfg
        self.kwargs = kwargs

        self.dataset_root = self._resolve_dataset_root()
        self.dataset = self._load_split_dataset(self.dataset_root)
        self._cache: Dict[int, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def set_progress(self, global_step: int) -> None:
        self._global_step = global_step

    def _candidate_root_names(self) -> List[str]:
        names: List[str] = []
        if self.split_root:
            names.append(self.split_root)

        split_aliases = [self.split]
        if self.split == "val":
            split_aliases.append("test")

        for split_name in split_aliases:
            names.append(f"{split_name}_new")

        # Keep the common train/test roots available even when the requested split
        # is provided through a direct dataset root or an alias.
        names.extend(["train_new", "test_new"])

        deduped: List[str] = []
        seen = set()
        for name in names:
            if name not in seen:
                seen.add(name)
                deduped.append(name)
        return deduped

    def _resolve_dataset_root(self) -> pathlib.Path:
        base = self.data_dir
        candidates: List[pathlib.Path] = []

        # Direct root first, then the notebook-observed nested structure.
        candidates.append(base)
        for name in self._candidate_root_names():
            candidates.append(base / name)
            candidates.append(base / "reasonaff" / name)

        seen = set()
        filtered: List[pathlib.Path] = []
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "dataset_dict.json").exists():
                filtered.append(candidate)

        if filtered:
            return filtered[0]

        msg = (
            f"[ReasonAFFDataset] Could not locate a saved Hugging Face dataset root for split={self.split}.\n"
            f"  data_dir: {self.data_dir}\n"
            f"  Tried:\n    " + "\n    ".join(str(p) for p in candidates)
        )
        raise FileNotFoundError(msg)

    def _load_split_dataset(self, dataset_root: pathlib.Path):
        obj = load_from_disk(str(dataset_root))

        if HFDataset is not None and isinstance(obj, HFDataset):
            return obj

        if DatasetDict is not None and isinstance(obj, DatasetDict):
            if self.split in obj:
                return obj[self.split]

            if self.split == "val" and "test" in obj:
                return obj["test"]

            available = list(obj.keys())
            raise KeyError(
                f"[ReasonAFFDataset] Requested split={self.split!r} not found under {dataset_root}. "
                f"Available splits: {available}"
            )

        if isinstance(obj, dict):
            if self.split in obj:
                return obj[self.split]
            available = list(obj.keys())
            raise KeyError(
                f"[ReasonAFFDataset] Requested split={self.split!r} not found under {dataset_root}. "
                f"Available splits: {available}"
            )

        raise TypeError(
            f"[ReasonAFFDataset] Unsupported object loaded from {dataset_root}: {type(obj).__name__}"
        )

    def _decode_image(self, image_value: Any) -> np.ndarray:
        if isinstance(image_value, Image.Image):
            image = image_value.convert("RGB")
        elif isinstance(image_value, dict):
            if image_value.get("bytes") is not None:
                image = Image.open(io.BytesIO(image_value["bytes"])).convert("RGB")
            elif image_value.get("path"):
                image = Image.open(image_value["path"]).convert("RGB")
            else:
                raise ValueError("Unsupported Hugging Face image dict without `bytes` or `path`.")
        else:
            image = Image.fromarray(np.asarray(image_value)).convert("RGB")

        arr = np.asarray(image).astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if self.transform is not None:
            arr = self.transform(arr)
        return arr

    def _build_query(self, sample: Dict[str, Any]) -> List[str]:
        raw_query = sample.get(self.query_field)
        queries = _as_query_list(raw_query)
        if not queries:
            if self.strict:
                raise ValueError(
                    f"[ReasonAFFDataset] Sample id={sample.get('id')} is missing query field {self.query_field!r}."
                )
            queries = [""]

        if self.query_style == "raw":
            return queries
        if self.query_style == "affordance_r1":
            return [_build_training_query(q) for q in queries]

        raise ValueError(
            f"[ReasonAFFDataset] Unsupported query_style={self.query_style!r}. "
            "Expected one of: 'raw', 'affordance_r1'."
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.cache and idx in self._cache:
            return self._cache[idx]

        sample = self.dataset[int(idx)]

        img = self._decode_image(sample["image"])
        mask = _normalize_mask(sample["mask"], image_size=(img.shape[1], img.shape[0]))

        img_t = torch.as_tensor(img, dtype=self.dtype).permute(2, 0, 1).contiguous()
        mask_t = torch.as_tensor(mask, dtype=self.dtype).unsqueeze(0).contiguous()

        item = {
            "image": img_t,
            "mask": mask_t,
            "query": self._build_query(sample),
            "id": sample.get("id"),
            "aff_name": sample.get("aff_name"),
            "part_name": sample.get("part_name"),
            "split": self.split,
        }

        if self.cache:
            self._cache[idx] = item
        return item

    @staticmethod
    def collate_function(batch: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        return collate_to_sam3(batch, **kwargs)
