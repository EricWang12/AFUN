"""Run the segmentation or motion benchmarks and write JSON results."""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import MOTION_SETS, SEGMENTATION_SETS, motion_samples, segmentation_samples
from .metrics import anchor_point, mask_iou, motion_metrics, summarize

logger = logging.getLogger(__name__)


def evaluate(score, samples, task, *, strict=False, output=None):
    records = []
    for sample in samples:
        row = {key: sample[key] for key in ("id", "query", "error") if key in sample}
        if "error" not in row:
            try:
                row.update(score(sample))
            except (RuntimeError, ValueError, KeyError, FloatingPointError) as error:
                if strict:
                    raise
                row["error"] = f"{type(error).__name__}: {error}"
        if "error" in row:
            logger.warning("No prediction for %s: %s", row["id"], row["error"])
        records.append(row)
        if output:
            output.write(json.dumps(row, allow_nan=False) + "\n")
            output.flush()
    return summarize(records, task)


@torch.inference_mode()
def predict(model, image, query, device, motion=False):
    """Top-scoring detection for a (3, H, W) image in [0, 1]: its 1008x1008 mask and motion parameters.

    The model runs as in the paper's evaluation: for segmentation SAM3 sees the normalized image, as
    in training; the motion benchmark gave it the unnormalized image and ran the motion head without
    the depth tokens.
    """
    from src.data.sam3_utils import collate_to_sam3
    batch = collate_to_sam3([{"image": image, "mask": torch.zeros_like(image[:1]), "query": [query]}],
                            normalize_image=not motion)
    images = batch["image"].to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        features = model.encode_prompt(batch["query"], [x.to(device) for x in batch["image_unnormalized"]], device)
        embeddings, masks = features["metaquery_prompt_embeds"], features["metaquery_attention_mask"]
        outputs, _ = model.decoder(images, feature=embeddings["semantic"], metaquery_mask=masks["semantic"])
        if motion:
            outputs["motion_directions"] = model.decoder.motion_processor(outputs["queries"], embeddings["motion"]).pred
    post = model.decoder._postprocess(outputs, orig_sizes=[images.shape[-2:]], has_motion=motion)
    scores = post["scores"][0]
    if len(scores) == 0:
        raise RuntimeError("no detection above the confidence threshold")
    best = int(scores.argmax())
    return post["masks"][0][best], post["motion_directions"][0][best].float().cpu() if motion else None


def resize_nearest(mask, size):
    return torch.nn.functional.interpolate(mask[None, None].float(), size=tuple(size), mode="nearest")[0, 0] > 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("segmentation", "motion"), required=True)
    parser.add_argument("--dataset", default="all", help="Benchmark name, or all.")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("AFUN_DATA_ROOT", "data")))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, help="Limit samples per dataset for a smoke test.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Stop on the first prediction failure.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    choices = SEGMENTATION_SETS if args.task == "segmentation" else MOTION_SETS
    names = choices if args.dataset == "all" else (args.dataset,)
    if any(name not in choices for name in names):
        parser.error(f"--dataset must be all or one of {', '.join(choices)}")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output_dir = args.output or Path("outputs/eval") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {output_dir}")
    from src.inference import load_model
    from src.models.decoder.motion_decoder.curves import compute_spline_curve
    config = args.config or ("inference" if args.task == "motion" else "eval_segmentation")
    model = load_model(str(args.checkpoint), config_name=config, device=args.device)
    device = torch.device(args.device)

    def score_segmentation(sample):
        mask, _ = predict(model, sample["image"], sample["query"], device)
        # Scored at image resolution, against the GT after the same round trip through 1008x1008.
        size = sample["mask"].shape
        target = resize_nearest(resize_nearest(sample["mask"] > 0.5, mask.shape), size)
        return mask_iou(resize_nearest(mask, size).cpu().numpy(), target.numpy())

    def score_motion(sample):
        image = torch.from_numpy(sample["image"]).permute(2, 0, 1).float() / 255
        mask, ctrl = predict(model, image, sample["query"], device, motion=True)
        height, width = sample["depth"].shape
        mask = cv2.resize(mask.cpu().numpy().astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        anchor = anchor_point(mask > 0, sample["depth"], sample["intrinsic"])
        curve = compute_spline_curve({"ctrl": ctrl, "r0": anchor}, frames=15)[0]  # 16 points, as in training
        return motion_metrics(curve, sample["trajectory"], sample["intrinsic"], sample["mask"])

    results = {}
    for name in names:
        if args.task == "motion":
            score, samples = score_motion, motion_samples(name, args.data_root, args.limit)
        else:
            score, samples = score_segmentation, segmentation_samples(name, args.data_root, args.limit)
        with (output_dir / f"{name}.jsonl").open("w") as handle:
            results[name] = evaluate(score, samples, args.task, strict=args.strict, output=handle)
        logger.info("%s: %s", name, json.dumps(results[name]))
    summary = {"task": args.task, "checkpoint": str(args.checkpoint.resolve()), "config": config,
               "sample_limit": args.limit, "results": results}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
