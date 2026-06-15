"""Batch-run the AFUN demo over a folder of example scenes.

Each scene is a subfolder of the examples directory:

    examples/
    ├── <scene_a>/
    │   ├── rgb.png        # required
    │   ├── query.txt      # required — the task description
    │   ├── depth.npy      # optional — RealSense raw depth (mm); if present → lingbot refine
    │   └── cam_K.txt      # optional — 3x3 intrinsic; required iff depth.npy present
    └── <scene_b>/ ...

Depth handling per scene:
  - no depth.npy            → Depth-Anything-3 generates depth from the RGB
  - depth.npy + cam_K.txt   → lingbot refines the provided depth

The model is loaded ONCE and reused across all scenes.

Outputs go to <out>/<scene>/{pred.npz, pred_seg.png, pred_3d.html} plus a
summary.csv and montage.png.

Run:
  conda activate afun
  python run_examples.py                          # examples/ -> outputs/examples/
  python run_examples.py --examples-dir path --out path [--no-3d] [--device cuda:0]
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("AFUN_ROOT", str(REPO_ROOT))


def _discover(examples_dir: pathlib.Path):
    """Yield (scene_name, query, rgb_path, depth_dir_or_None).

    depth_dir is the scene folder itself when it carries depth.npy + cam_K.txt
    (→ lingbot refine), otherwise None (→ DA3 generate).
    """
    scenes = []
    for d in sorted(p for p in examples_dir.iterdir() if p.is_dir()):
        rgb = d / "rgb.png"
        qf = d / "query.txt"
        if not rgb.exists():
            print(f"[skip] {d.name}: no rgb.png")
            continue
        if not qf.exists():
            print(f"[skip] {d.name}: no query.txt")
            continue
        query = qf.read_text().strip()
        depth_dir = d if (d / "depth.npy").exists() and (d / "cam_K.txt").exists() else None
        scenes.append((d.name, query, rgb, depth_dir))
    return scenes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples-dir", default=str(REPO_ROOT / "examples"))
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "examples"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "checkpoints" / "afun.pt"))
    ap.add_argument("--config", default="inference")
    ap.add_argument("--no-3d", action="store_true", help="Skip the Plotly 3D HTML render.")
    args = ap.parse_args()

    examples_dir = pathlib.Path(args.examples_dir)
    if not examples_dir.is_dir():
        print(f"ERROR: examples dir not found: {examples_dir}", file=sys.stderr)
        return 1
    out_root = pathlib.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    scenes = _discover(examples_dir)
    if not scenes:
        print(f"ERROR: no valid scenes in {examples_dir}", file=sys.stderr)
        return 1
    n_da3 = sum(1 for s in scenes if s[3] is None)
    print(f"==> {len(scenes)} scenes ({n_da3} DA3-generate, {len(scenes)-n_da3} lingbot-refine)")

    from src.inference import load_model, infer_image, save_results
    from depth import prepare_depth

    print(f"==> loading model {args.ckpt} (config={args.config})...")
    model = load_model(args.ckpt, args.config, device=args.device)

    rows = []
    for i, (name, query, rgb_path, depth_dir) in enumerate(scenes, 1):
        sample_out = out_root / name
        kind = "lingbot" if depth_dir is not None else "da3"
        print(f"\n[{i}/{len(scenes)}] {name}  [{kind}]  query={query!r}")
        try:
            rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
            depth_mm, K, kind = prepare_depth(rgb, depth_dir, device=args.device)
            cam = {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                   "cx": float(K[0, 2]), "cy": float(K[1, 2])}
            result = infer_image(model, image=Image.fromarray(rgb), query=query,
                                 device=args.device, depth=depth_mm, cam=cam)

            sample_out.mkdir(parents=True, exist_ok=True)
            if args.no_3d:
                stash = result.get("depth")
                result["depth"] = None
                save_results(result, str(sample_out), tag="pred")
                result["depth"] = stash
            else:
                save_results(result, str(sample_out), tag="pred")

            score = float(result.get("score") or 0.0)
            ndet = int(result.get("num_detections", 0))
            rows.append((name, kind, query, ndet, round(score, 3), "ok"))
            print(f"    ✓ detections={ndet} best_conf={score:.3f}")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            rows.append((name, kind, query, 0, 0.0, f"ERROR: {type(e).__name__}"))
            print(f"    ✗ {type(e).__name__}: {e}")

    csv_path = out_root / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "depth_path", "query", "detections", "best_conf", "status"])
        w.writerows(rows)
    ok = sum(1 for r in rows if r[5] == "ok")
    print(f"\n==> {ok}/{len(rows)} succeeded.  summary: {csv_path}")

    try:
        _build_montage(out_root, rows)
        print(f"==> montage: {out_root / 'montage.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"[montage] skipped: {e}")
    return 0 if ok == len(rows) else 1


def _build_montage(out_root: pathlib.Path, rows, cols: int = 3, thumb_w: int = 460):
    items = [(name, q, conf, out_root / name / "pred_seg.png")
             for name, kind, q, ndet, conf, status in rows
             if status == "ok" and (out_root / name / "pred_seg.png").exists()]
    if not items:
        return
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
    except Exception:
        font = ImageFont.load_default()
    label_h = 30
    thumbs = []
    for name, query, conf, png in items:
        im = Image.open(png).convert("RGB")
        w, h = im.size
        th = int(h * thumb_w / w)
        im = im.resize((thumb_w, th), Image.LANCZOS)
        canvas = Image.new("RGB", (thumb_w, th + label_h), (18, 18, 18))
        canvas.paste(im, (0, label_h))
        d = ImageDraw.Draw(canvas)
        label = f'"{query}"  ({conf})'
        if len(label) > 58:
            label = label[:55] + "…"
        d.text((6, 6), label, fill=(255, 210, 90), font=font)
        thumbs.append(canvas)
    cell_w, cell_h = thumb_w, max(t.height for t in thumbs)
    rows_n = (len(thumbs) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell_w, rows_n * cell_h), (0, 0, 0))
    for idx, t in enumerate(thumbs):
        r, c = divmod(idx, cols)
        grid.paste(t, (c * cell_w, r * cell_h))
    grid.save(out_root / "montage.png")


if __name__ == "__main__":
    sys.exit(main())
