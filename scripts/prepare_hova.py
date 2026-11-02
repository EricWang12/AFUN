"""Convert HOVA contact annotations into SAM3 masks and task instructions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SOURCE_DIRS = {"3doi": "3doi", "ego4d": "Ego4D", "handal": "HANDAL", "epic100": "epic-100"}


def contacts(root, sources, splits, skip=()):
    for split in splits:
        for source in sources:
            annotation = root / "annotations" / split / f"{source}.json"
            if not annotation.exists():
                continue
            records = json.loads(annotation.read_text())
            for index, record in enumerate(records):
                folder = root / SOURCE_DIRS[source]
                if source == "3doi":
                    image = folder / "images" / record["img_name"]
                elif source == "ego4d":
                    image = folder / "frames" / f"{Path(record['img_path']).stem}.jpg"
                elif source == "handal":
                    image = folder / "images" / record["img_path"]
                else:
                    image = folder / "EPIC-KITCHENS_frames" / record["img_path"]
                    if not image.exists():
                        image = folder / record["img_path"]
                uid = hashlib.sha256(f"{source}/{split}/{index}/{image.name}".encode()).hexdigest()[:20]
                if uid in skip:
                    continue
                with Image.open(image) as opened:
                    width, height = opened.size
                # HANDAL keeps its contact heatmaps under annotations/GT_gaussian_<split>.
                heatmaps = folder / "annotations" / f"GT_gaussian_{split}" if source == "handal" else folder / "GT_gaussian"
                heatmap = heatmaps / record.get("gt_path", "")
                if heatmap.is_file() and source != "epic100":
                    with Image.open(heatmap) as opened:
                        heat = np.array(opened.convert("L"))
                    v, u = np.unravel_index(heat.argmax(), heat.shape)
                    point = (u * width / heat.shape[1], v * height / heat.shape[0])
                elif record.get("points"):
                    point = np.asarray(record["points"], dtype=float).mean(0)
                elif record.get("affordance"):
                    point = np.asarray(record["affordance"], dtype=float) * [width, height]
                else:
                    raise ValueError(f"No contact annotation: {annotation}:{index}")
                yield {"uid": uid, "image": image, "source": source, "split": split,
                       "point": point, "noun": record.get("noun", record.get("object", "object")),
                       "verb": record.get("verb", record.get("action", "interact"))}


class Annotator:
    def __init__(self, model_name, device):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=torch.bfloat16,
                                                                device_map=device).eval()
        self.sam = Sam3Processor(build_sam3_image_model(device=device))

    @torch.inference_mode()
    def describe(self, image, text):
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     return_dict=True, return_tensors="pt").to(self.device)
        generated = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        return self.processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    @torch.inference_mode()
    def __call__(self, sample):
        with Image.open(sample["image"]) as opened:
            image = opened.convert("RGB")
        u, v = sample["point"]
        part = self.describe(image, f"The contact point is ({u:.0f}, {v:.0f}) pixels in this {image.width} by {image.height} image. "
                                   f"Object: {sample['noun']}. Action: {sample['verb']}. "
                                   "Name the functional object part at this point. Return only a short noun phrase, such as 'drawer handle'.")
        with torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda")):
            state = self.sam.set_image(image)
            result = self.sam.set_text_prompt(state=state, prompt=part)
        masks = result["masks"].squeeze(1).cpu().numpy() > 0
        scores = result["scores"].float().cpu().numpy()
        selected = None
        for index in np.argsort(scores)[::-1]:
            ys, xs = np.where(masks[index])
            if scores[index] > 0.5 and len(xs) and np.hypot(xs.mean() - u, ys.mean() - v) < 0.07 * image.width:
                selected = masks[index]
                break
        if selected is None:
            return None
        overlay = np.array(image).copy()
        overlay[selected] = (0.5 * overlay[selected] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        query = self.describe(Image.fromarray(overlay),
                              f"Object: {sample['noun']}. Action: {sample['verb']}. "
                              "Write one short task instruction that requires interacting with the red-highlighted region. "
                              "Do not mention the highlight, coordinates, or the part name. Return only the instruction.")
        if not query:
            raise ValueError("The instruction model returned an empty query.")
        return selected, query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    # AFUN's HOVA masks cover 3doi, Ego4D, and HANDAL; EPIC-100 was not used for training.
    parser.add_argument("--sources", nargs="+", choices=SOURCE_DIRS, default=["3doi", "ego4d", "handal"])
    parser.add_argument("--splits", nargs="+", choices=("train", "test"), default=["train", "test"])
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    output = args.output or args.root / "mask_annotation"
    (output / "mask").mkdir(parents=True, exist_ok=True)
    index_path = output / "dataset_index.json"
    rejected_path = output / "rejected.txt"  # contacts that failed the filters; restarts skip them
    records = json.loads(index_path.read_text()) if index_path.exists() else []
    skip = {r["uid"] for r in records}
    if rejected_path.exists():
        skip |= set(rejected_path.read_text().split())

    def relative(path):
        # Paths are relative to the dataset root when output is inside it.
        return str(path.resolve().relative_to(args.root.resolve())) if path.resolve().is_relative_to(args.root.resolve()) else str(path.resolve())

    def save_index():
        temporary = index_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(records, indent=2) + "\n")
        temporary.replace(index_path)

    annotator = Annotator(args.model, args.device)
    accepted, attempted = 0, 0
    try:
        with rejected_path.open("a") as rejected:
            for sample in contacts(args.root, args.sources, args.splits, skip=skip):
                if args.limit is not None and attempted >= args.limit:
                    break
                attempted += 1
                result = annotator(sample)
                if result is None:
                    rejected.write(sample["uid"] + "\n")
                    rejected.flush()
                    continue
                mask, query = result
                mask_path = output / "mask" / f"{sample['uid']}.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
                records.append({"uid": sample["uid"], "image_path": relative(sample["image"]),
                                "mask_path": relative(mask_path), "query": query,
                                "source": sample["source"], "split": sample["split"]})
                accepted += 1
                if accepted % 100 == 0:  # rewriting the whole index per sample would be quadratic
                    save_index()
                print(f"{attempted}: accepted {accepted}", flush=True)
    finally:
        save_index()
    (output / "preparation.json").write_text(json.dumps({"model": args.model, "sources": args.sources,
                                                         "attempted": attempted, "accepted": accepted}, indent=2))
    print(f"This run: {accepted} of {attempted} contacts accepted. Saved {len(records)} records to {index_path}")


if __name__ == "__main__":
    main()
