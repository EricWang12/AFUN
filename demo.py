"""AFUN single-image inference demo.

Examples
--------
RGB only — DA3 generates depth + intrinsics:

    python demo.py --rgb path/to/image.png --query "open the cabinet"

RGB + RealSense raw depth — lingbot refines the depth:

    python demo.py --rgb path/to/color.png --query "open the drawer" \\
                   --depth-dir path/to/folder/   # contains depth.npy + cam_K.txt

Outputs under ``--out`` (default ``outputs/<rgb-stem>``):

    pred.npz       — mask, motion params, intrinsics, confidence
    pred_seg.png   — 2D mask + curve overlay on RGB
    pred_3d.html   — Plotly 3D viewer (omit with --no-3d)
    meta.json      — run metadata: rgb, query, ckpt, config, depth_source, device, ts
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import re
import sys

import numpy as np
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Make AFUN_ROOT available to the Hydra config interpolations.
os.environ.setdefault("AFUN_ROOT", str(REPO_ROOT))


def _slugify(text: str, maxlen: int = 48) -> str:
    """Make a filesystem-safe slug from a free-form query."""
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:maxlen] or "query"


def _k_to_cam_dict(K: np.ndarray) -> dict:
    """3x3 intrinsic matrix → {fx, fy, cx, cy} dict (the contract infer_image expects)."""
    K = np.asarray(K)
    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="AFUN single-image inference demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--rgb", required=True, type=pathlib.Path,
                   help="Path to RGB image (PNG/JPG/...).")
    p.add_argument("--query", required=True, type=str,
                   help='Natural-language task description, e.g. "open the cabinet".')
    p.add_argument("--depth-dir", type=pathlib.Path, default=None,
                   help="Optional folder containing depth.npy (mm float32) + cam_K.txt (3x3). "
                        "When given, the depth is refined via lingbot. When omitted, DA3 generates depth from RGB.")
    p.add_argument("--no-refine", action="store_true",
                   help="With --depth-dir, skip lingbot refinement and feed the raw sensor depth "
                        "straight to the model. No effect without --depth-dir.")
    p.add_argument("--ckpt", type=pathlib.Path,
                   default=REPO_ROOT / "checkpoints" / "afun.pt",
                   help="AFUN checkpoint to load.")
    p.add_argument("--config", type=str, default="inference",
                   help="Hydra config name under configs/.")
    p.add_argument("--out", type=pathlib.Path, default=None,
                   help="Output directory. Default: outputs/<rgb-stem>_<query-slug>.")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Torch device.")
    p.add_argument("--no-3d", action="store_true",
                   help="Skip the Plotly 3D HTML render. Keeps NPZ + PNG + meta.json.")
    args = p.parse_args(argv)

    # Resolve inputs
    if not args.rgb.exists():
        print(f"ERROR: --rgb file not found: {args.rgb}", file=sys.stderr)
        return 1
    if not args.ckpt.exists():
        print(f"ERROR: --ckpt not found: {args.ckpt}", file=sys.stderr)
        return 1
    out_dir = args.out or (REPO_ROOT / "outputs" / f"{args.rgb.stem}_{_slugify(args.query)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load RGB
    rgb_pil = Image.open(args.rgb).convert("RGB")
    rgb_np = np.asarray(rgb_pil)
    H, W = rgb_np.shape[:2]
    print(f"==> RGB: {args.rgb}  ({W}x{H})")
    print(f"==> query: {args.query!r}")
    print(f"==> out: {out_dir}")

    # Depth: DA3 generates it from RGB, or lingbot refines a provided --depth-dir
    # (use --no-refine to feed that sensor depth in raw, skipping lingbot).
    from depth import prepare_depth
    if args.depth_dir is None:
        if args.no_refine:
            print("==> --no-refine has no effect without --depth-dir (nothing to refine)")
        print("==> Generating depth with Depth Anything 3 ...")
    elif args.no_refine:
        print(f"==> Using raw sensor depth from {args.depth_dir} (--no-refine: skipping lingbot) ...")
    else:
        print(f"==> Refining depth with lingbot ({args.depth_dir}) ...")
    depth_mm, K, depth_source = prepare_depth(
        rgb_np, args.depth_dir, device=args.device, refine=not args.no_refine
    )
    print(f"    final depth: shape={depth_mm.shape}  "
          f"range=[{depth_mm.min():.1f},{depth_mm.max():.1f}] mm  (source={depth_source})")

    # Load model
    from src.inference import load_model, infer_image, save_results
    print(f"==> loading model from {args.ckpt} (config={args.config}, device={args.device})...")
    model = load_model(str(args.ckpt), args.config, device=args.device)

    # Run inference
    cam = _k_to_cam_dict(K)
    print(f"==> running inference (cam fx={cam['fx']:.1f}, fy={cam['fy']:.1f}, "
          f"cx={cam['cx']:.1f}, cy={cam['cy']:.1f})...")
    result = infer_image(
        model,
        image=rgb_pil,
        query=args.query,
        device=args.device,
        depth=depth_mm,
        cam=cam,
    )

    # Persist outputs (NPZ + seg PNG + optional 3D HTML)
    print(f"==> saving outputs to {out_dir}/ ...")
    save_results(result, str(out_dir), tag="pred")

    if args.no_3d:
        plotly_html = out_dir / "pred_3d.html"
        if plotly_html.exists():
            plotly_html.unlink()
            print("    (--no-3d: removed pred_3d.html)")

    # Forensic metadata
    meta = {
        "rgb": str(args.rgb.resolve()),
        "query": args.query,
        "ckpt": str(args.ckpt.resolve()),
        "config": args.config,
        "depth_source": depth_source,
        "depth_dir": str(args.depth_dir.resolve()) if args.depth_dir else None,
        "device": args.device,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "image_size": [H, W],
        "out_dir": str(out_dir.resolve()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Summary
    best_conf = float(result.get("score", 0.0)) if result.get("score") is not None else float("nan")
    n_det = int(result.get("num_detections", 0))
    print(f"\n✓ pred saved to {out_dir}/   depth={depth_source}   "
          f"detections={n_det}   best_conf={best_conf:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
