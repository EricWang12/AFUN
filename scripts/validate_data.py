"""Validate dataset loading and collation without loading model weights."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("visual_genome", "instruct_part", "ragnet", "hova500k", "reasonaff", "robot"), required=True)
    parser.add_argument("--root", type=Path, required=True, help="This dataset's directory.")
    parser.add_argument("--samples", type=int, default=8, help="Evenly spaced samples; 0 validates every sample.")
    args = parser.parse_args()
    if args.samples < 0:
        parser.error("--samples must be nonnegative")
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="train_stage3", overrides=[f"data={args.dataset}", f"data.data_dir={args.root}",
                                                            "train.batch_size=1", "data.loader_cfg.num_workers=0"])
    dataset = instantiate(cfg.data)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")
    import numpy as np
    count = len(dataset) if args.samples == 0 else min(args.samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    for index in indices:
        batch = dataset.get_collate_fn()([dataset[int(index)]])
        if not np.isfinite(batch["image"].numpy()).all():
            raise ValueError(f"Non-finite image at {index}")
    print(json.dumps({"dataset": args.dataset, "samples": len(dataset), "validated": count}))


if __name__ == "__main__":
    main()
