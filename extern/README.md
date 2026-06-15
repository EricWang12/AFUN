# `extern/` — Vendored Depth Stack

This directory contains two **patched, vendored** depth models. They are committed
as regular files (not git submodules) because the patched in-tree state must
persist exactly as is.

## DO NOT re-clone these from upstream.

Both repos carry in-tree fixes that disappear if you re-clone from GitHub /
HuggingFace. Re-cloning will silently reintroduce import failures and runtime
errors that have already been patched out.

| Dir | Upstream | Why vendored |
|---|---|---|
| `Depth-Anything-3/` | https://github.com/DepthAnything/Depth-Anything-3 | DA3 streaming optional-deps patched to be no-ops when absent (`faiss`, `pypose`, `triton`, `numba`, `trimesh`, salad helpers). Plus the AFUN-inference `evo` patch (below). |
| `lingbot-depth/` | https://github.com/robbyant/lingbot-depth | Patched to load cleanly with the shared `xformers==0.0.35` build. |

### AFUN-inference patch: lazy `evo` import

`src/depth_anything_3/utils/pose_align.py` originally did
`from evo.core.trajectory import PosePath3D` at module top. `evo` is only used
for **multi-view** pose alignment (`align_poses_umeyama`), which single-image
depth generation never reaches, and it is not in the AFUN dependency set. The
import was moved inside `_umeyama_sim3_from_paths` (lazy) so `depth_anything_3.api`
imports cleanly for single-image depth. If you ever do multi-view alignment,
`pip install evo` first.

### AFUN-inference patch: lazy export backends

`src/depth_anything_3/utils/export/__init__.py` originally imported every export
backend at module top (`gs` → `moviepy`, `colmap` → `pycolmap`, `glb` → `trimesh`,
…). Importing `depth_anything_3.api` therefore pulled in heavy/optional export
deps even for plain depth inference (which never exports). The per-format imports
were moved **inside** `export()`, so each backend is only imported when that
format is actually requested. Depth-only inference imports cleanly without
`moviepy` / `pycolmap`.

The `gsplat` / `e3nn` warnings DA3 prints at load are harmless — those deps are
only needed for 3D Gaussian-splat rendering, which the demo never invokes.

These are vendored, patched copies of the upstream repos (links above). The
patches are what make them work for AFUN's single-image depth inference, so do
**not** replace these trees by re-cloning upstream.

## How they get imported

The wrappers under `depth/` (in the repo root) add the appropriate paths to
`sys.path` at first use, then import:

- `depth_anything_3.api.DepthAnything3` from `Depth-Anything-3/src/`
- `mdm.model.v2.MDMModel` from `lingbot-depth/`

No `pip install` of these two is needed — they run straight from `extern/` via
`sys.path`.

Model weights for both download from HuggingFace on first call (~6.8 GB for
DA3 nested-giant-large, ~1.3 GB for lingbot vitl-14). They land in
`~/.cache/huggingface/hub/`, **not** in this directory.
