from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image
import numpy as np
import torch
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
from .base import DatasetBase


from src.data.sam3_utils import collate_to_sam3

def _clean_name(s: str) -> str:
    return str(s).strip().replace(" ", "_")


def _try_suffixes(path_no_suffix: pathlib.Path, suffixes: List[str]) -> Optional[pathlib.Path]:
    for suf in suffixes:
        cand = path_no_suffix.with_suffix(suf)
        if cand.exists():
            return cand
    return None


def _imread_rgb(path: pathlib.Path) -> np.ndarray:
    if cv2 is None:
        raise ModuleNotFoundError(
            "InstructPartDataset requires an image reader. Install Pillow (`pip install pillow`) "
            "or ensure OpenCV (`cv2`) is available."
        )
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb



def _imread_gray(path: pathlib.Path) -> np.ndarray:
    try:
        with Image.open(path) as img:
            g = img.convert("L")
            arr = np.array(g)
    except Exception as e:
        raise FileNotFoundError(f"Could not read mask: {path}") from e
    return arr


def _resize_mask_nearest(mask: np.ndarray, *, out_h: int, out_w: int) -> np.ndarray:
    """Nearest resize using OpenCV if available, else torch interpolate."""
    if mask.shape[0] == out_h and mask.shape[1] == out_w:
        return mask
    if cv2 is not None:
        return cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    # torch fallback
    m = torch.from_numpy(mask).float()[None, None]  # (1,1,H,W)
    m2 = torch.nn.functional.interpolate(m, size=(out_h, out_w), mode="nearest")
    return m2[0, 0].cpu().numpy()


@dataclass(frozen=True)
class _InstructPartSample:
    image_file: str               # basename in images/
    mask_file: str                # basename in masks/
    query: List[str]       # list of query variants (len can be 1..6)




class InstructPartDataset(DatasetBase):
    """
    InstructPart loader (minimal):
      - `image`:        (3, H, W) float32 in [0,1]
      - `mask`:         (1, H, W) float32 in [0,1] (a.k.a. heatmap)
      - `query`: list[str] (all query variants provided in JSON)

    Each sample corresponds to one (image, object-part) item from JSON.
    """

    def __init__(
        self,
        data_dir: str | pathlib.Path,
        *,
        split: str = "test",
        subset: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        transform=None,  # optional callable applied to RGB image (H,W,3) float32 in [0,1] -> same shape
        cache: bool = False,
        strict: bool = False,  # if True: error on missing files; else: skip
        loader_cfg: Dict[str, Any] = None,
    ):
        if isinstance(data_dir, str):
            data_dir = pathlib.Path(data_dir)
        super().__init__(
            data_dir=data_dir,
            split=str(split),
            dtype=dtype,
        )
        self.data_dir = data_dir
        self.split = str(split)
        self.subset = subset
        self.dtype = dtype
        self.transform = transform
        self.cache = bool(cache)
        self.strict = bool(strict)
        self.loader_cfg = loader_cfg

        base_dir = self.data_dir
        if not (base_dir / self.split / "masks").is_dir() and (base_dir / "InstructPart").is_dir():
            base_dir = base_dir / "InstructPart"
        if self.split.startswith("train") and (base_dir / "train" / self.split).is_dir():
            base_dir = base_dir / "train"
        if self.subset is not None:
            split_dir = base_dir / self.split / str(self.subset)
        else:
            split_dir = base_dir / self.split

        if not split_dir.exists():
            raise FileNotFoundError(f"InstructPart split dir not found: {split_dir}")

        self.split_dir = split_dir
        self.images_dir = split_dir / "images"
        self.masks_dir = split_dir / "masks"
        if not self.images_dir.exists():
            raise FileNotFoundError(f"images dir not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"masks dir not found: {self.masks_dir}")

        # Infer JSON file name
        # - train/*: data_train.json
        # - test:    data_test.json
        # - all:     data_all.json
        if self.split.startswith("train"):
            json_name = "data_train.json"
        elif self.split == "test":
            json_name = "data_test.json"
        elif self.split == "all":
            json_name = "data_all.json"
        else:
            # fallback: data_<split>.json
            json_name = f"data_{self.split}.json"

        self.json_path = split_dir / json_name
        if not self.json_path.exists():
            raise FileNotFoundError(f"annotation json not found: {self.json_path}")

        # Optional RAM caching (decoded arrays)
        self._img_cache: Dict[str, np.ndarray] = {}
        self._mask_cache: Dict[str, np.ndarray] = {}

        self.meta: List[_InstructPartSample] = []
        self._build_index()

    def __len__(self) -> int:
        return len(self.meta)

    def set_progress(self, global_step: int) -> None:
        self._global_step = global_step

    def _build_index(self) -> None:
        with open(self.json_path, "r") as f:
            data = json.load(f)

        img_suffixes = [".jpg", ".jpeg", ".png", ".webp"]
        mask_suffixes = [".png", ".jpg", ".jpeg", ".webp"]

        skipped = 0
        for entry in data:
            image_path = str(entry.get("image_path", "")).strip()
            if not image_path:
                continue
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            orig_ext = os.path.splitext(os.path.basename(image_path))[1] or ".jpg"

            part_list = entry.get("part_list", []) or []
            for item in part_list:
                obj = _clean_name(item.get("object", ""))
                part = _clean_name(item.get("part", ""))
                inst_list = item.get("instruction", []) or []
                if not isinstance(inst_list, list):
                    inst_list = [str(inst_list)]
                inst_list = [str(x) for x in inst_list]

                # Most releases store per-(object,part) images using:
                #   <base>-<object>-<part>.<ext>
                # but some might store the raw `image_path` as-is.
                direct_image = self.images_dir / os.path.basename(image_path)
                if direct_image.exists():
                    image_file = direct_image.name
                    image_stem = direct_image.stem
                else:
                    derived_stem = f"{base_filename}-{obj}-{part}"
                    cand = _try_suffixes(self.images_dir / derived_stem, [orig_ext] + img_suffixes)
                    if cand is None:
                        if self.strict:
                            raise FileNotFoundError(
                                f"Could not resolve image for {image_path} obj={obj} part={part} under {self.images_dir}"
                            )
                        skipped += 1
                        continue
                    image_file = cand.name
                    image_stem = cand.stem

                # Mask usually matches image stem
                mask_cand = _try_suffixes(self.masks_dir / image_stem, mask_suffixes)
                if mask_cand is None:
                    if self.strict:
                        raise FileNotFoundError(f"Could not resolve mask for image={image_file} under {self.masks_dir}")
                    skipped += 1
                    continue

                self.meta.append(
                    _InstructPartSample(
                        image_file=image_file,
                        mask_file=mask_cand.name,
                        query=inst_list,
                    )
                )

        if skipped > 0:
            print(f"[InstructPartDataset] skipped {skipped} entries due to missing image/mask files")
        print(f"[InstructPartDataset] loaded {len(self.meta)} samples from {self.json_path}")

    def _load_image(self, image_file: str) -> np.ndarray:
        if self.cache and image_file in self._img_cache:
            return self._img_cache[image_file]
        path = self.images_dir / image_file
        arr = _imread_rgb(path).astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if self.transform is not None:
            arr = self.transform(arr)
        if self.cache:
            self._img_cache[image_file] = arr
        return arr

    def _load_mask(self, mask_file: str) -> np.ndarray:
        if self.cache and mask_file in self._mask_cache:
            return self._mask_cache[mask_file]
        path = self.masks_dir / mask_file
        arr = _imread_gray(path).astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if self.cache:
            self._mask_cache[mask_file] = arr
        return arr

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.meta[idx]

        img = self._load_image(s.image_file)  # (H,W,3) float32 in [0,1]
        mask = self._load_mask(s.mask_file)   # (H,W) float32 in [0,1]

        H, W = img.shape[0], img.shape[1]
        if mask.shape[0] != H or mask.shape[1] != W:
            # best-effort: resize mask to image size (nearest)
            mask = _resize_mask_nearest(mask, out_h=H, out_w=W)
            if mask.max() > 1.0:
                mask = mask.astype(np.float32) / 255.0

        # To torch
        img_t = torch.as_tensor(img, dtype=self.dtype).permute(2, 0, 1).contiguous()   # (3,H,W)
        mask_t = torch.as_tensor(mask, dtype=self.dtype).unsqueeze(0).contiguous()    # (1,H,W)

        return {
            "image": img_t,
            "mask": mask_t,                 # heatmap / segmentation target
            "query": [s.query[0]] if self.split == "test" else s.query,  # eval: fixed [0]; train: all variants

        }
        
    @staticmethod
    def collate_function(batch: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        return collate_to_sam3(batch, **kwargs)
