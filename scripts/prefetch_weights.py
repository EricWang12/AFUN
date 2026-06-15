"""Pre-download the AFUN checkpoint + every backbone weight so the first demo run is instant.

    python scripts/prefetch_weights.py

- The AFUN checkpoint ``afun.pt`` is fetched from Google Drive into ``checkpoints/``.
  Configure the source via ``AFUN_CKPT_GDRIVE_DEFAULT`` below, or at runtime without
  editing this file:
      AFUN_CKPT_GDRIVE=<file-id-or-share-url> python scripts/prefetch_weights.py
- The backbone weights load from Hugging Face into ~/.cache/huggingface/hub/ (or
  $HF_HOME if set), NOT this repo (~32 GB total):
    * Qwen3-VL (~17.5 GB), SAM3 (~3.5 GB), Sonata (~3 GB) — the inference model
    * Depth-Anything-3 (~6.8 GB) + lingbot-depth (~1.3 GB) — depth preprocessing
  The Qwen3-VL / Sonata repo ids are read from the ``inference`` config and SAM3 from its
  own downloader, so they track the configs rather than a copy hardcoded here.
"""
from __future__ import annotations

import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Hydra config interpolations expect AFUN_ROOT (matches demo.py).
os.environ.setdefault("AFUN_ROOT", str(REPO_ROOT))

# AFUN checkpoint (afun.pt) source on Google Drive.
# Paste EITHER the file ID (e.g. "1AbCdEf0123...") OR a full share URL
# (e.g. "https://drive.google.com/file/d/1AbCdEf0123.../view?usp=sharing").
# Leave as "" to skip the auto-download (you can still place afun.pt manually).
AFUN_CKPT_GDRIVE_DEFAULT = "1J4TNj_w3b7BTaeriBPuDJCn7q_SlV6PR"


def _download_checkpoint() -> None:
    """Download afun.pt from Google Drive into checkpoints/ (skips if already present)."""
    dest = REPO_ROOT / "checkpoints" / "afun.pt"
    if dest.exists():
        print(f"✓ AFUN checkpoint already present: {dest}")
        return

    source = os.environ.get("AFUN_CKPT_GDRIVE", AFUN_CKPT_GDRIVE_DEFAULT).strip()
    if not source:
        print("!! No Google Drive source set for the AFUN checkpoint — skipping.")
        print("   Set AFUN_CKPT_GDRIVE_DEFAULT in this file, or run:")
        print("     AFUN_CKPT_GDRIVE=<file-id-or-share-url> python scripts/prefetch_weights.py")
        return

    import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"==> downloading AFUN checkpoint from Google Drive → {dest} ...")
    if source.startswith("http"):
        gdown.download(url=source, output=str(dest), fuzzy=True, quiet=False)
    else:
        gdown.download(id=source, output=str(dest), quiet=False)

    if not dest.exists():
        print("!! Download did not produce checkpoints/afun.pt. Check that the Drive file is "
              "shared ('Anyone with the link') and the ID/URL is correct.")
        sys.exit(1)
    print(f"✓ AFUN checkpoint saved to {dest}")


def _prefetch_backbones() -> None:
    """Pre-download the Hugging Face backbones the inference model loads:
    Qwen3-VL (LLM), SAM3 (segmentation decoder), and Sonata (depth geo-encoder).

    Qwen3-VL / Sonata repo ids come from the composed ``inference`` config and SAM3
    from its own downloader, so they track the configs rather than a copy hardcoded
    here. Files land in ~/.cache/huggingface/hub/ (or $HF_HOME).
    """
    from huggingface_hub import snapshot_download
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(config_name="inference")

    qwen_id = cfg.llm.model_name
    print(f"==> prefetching Qwen3-VL weights ({qwen_id}) ...")
    snapshot_download(repo_id=qwen_id)

    # SAM3: reuse the builder's own downloader so we fetch exactly the files it loads
    # (config.json + sam3.pt) from whatever repo it targets — never a duplicated id.
    print("==> prefetching SAM3 weights ...")
    sam3_root = str(REPO_ROOT / "src" / "models" / "decoder" / "sam3")
    if sam3_root not in sys.path:
        sys.path.append(sam3_root)
    from sam3.model_builder import download_ckpt_from_hf  # pyright: ignore[reportMissingImports]
    download_ckpt_from_hf()

    # Sonata: only when the decoder has a geo depth-encoder pointing at an HF repo
    # (a local path means the weights are already on disk — nothing to fetch).
    depth_enc = cfg.decoder.get("depth_encoder", None)
    sonata_id = depth_enc.get("ckpt_path", None) if depth_enc is not None else None
    if sonata_id and not os.path.exists(sonata_id):
        print(f"==> prefetching Sonata weights ({sonata_id}) ...")
        snapshot_download(repo_id=sonata_id)


def main() -> int:
    print("==> prefetching AFUN checkpoint ...")
    _download_checkpoint()

    _prefetch_backbones()

    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("==> prefetching Depth-Anything-3 weights ...")
    from depth.da3 import _load_model as _load_da3  # noqa: PLC2701
    _load_da3(torch.device(device))

    print("==> prefetching lingbot-depth weights ...")
    from depth.lingbot import _load_model as _load_lingbot  # noqa: PLC2701
    _load_lingbot(torch.device(device))

    print("✓ checkpoint + backbones (Qwen3-VL, SAM3, Sonata) + depth models (DA3, lingbot) cached.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
