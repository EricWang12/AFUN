"""Download AFUN weights and cache the pretrained backbones."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = "1J4TNj_w3b7BTaeriBPuDJCn7q_SlV6PR"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbones-only", action="store_true", help="Skip the trained AFUN checkpoint.")
    parser.add_argument("--checkpoint-only", action="store_true", help="Skip pretrained backbones.")
    parser.add_argument("--include-depth", action="store_true", help="Also cache optional RGB/sensor depth models.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/afun.pt")
    parser.add_argument("--checkpoint-source", default=os.environ.get("AFUN_CKPT_GDRIVE", DEFAULT_CHECKPOINT),
                        help="Google Drive file ID or share URL.")
    args = parser.parse_args()
    if args.backbones_only and args.checkpoint_only:
        parser.error("Choose --backbones-only or --checkpoint-only.")
    if not args.backbones_only and not args.checkpoint.is_file():
        import gdown
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.checkpoint.with_suffix(".pt.download")
        source = args.checkpoint_source
        options = {"url": source, "fuzzy": True} if source.startswith("http") else {"id": source}
        result = gdown.download(output=str(temporary), **options)
        if not result or not temporary.is_file():
            raise RuntimeError(f"Checkpoint download failed: {source}")
        temporary.replace(args.checkpoint)
    if not args.checkpoint_only:
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
            cfg = compose(config_name="inference")
        for repository in (cfg.llm.model_name, cfg.decoder.depth_encoder.ckpt_path):
            snapshot_download(repository)
        # SAM3 is gated; accept its model terms and run `hf auth login` first.
        snapshot_download("facebook/sam3", allow_patterns=["sam3.pt", "config.json"])
        if args.include_depth:
            for repository in ("depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                               "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"):
                snapshot_download(repository)
    print("Weights are ready.")


if __name__ == "__main__":
    main()
