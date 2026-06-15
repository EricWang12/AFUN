# examples — demo inputs

Each subfolder is **one demo task**. `run_examples.py` runs every subfolder
here and writes results to `outputs/examples/<task>/`.

## Naming convention

Folder name = a short, lowercase `scene_task` label (snake_case). The folder
name is just a label for organizing/sorting — the actual instruction lives in
`query.txt`, so keep names short.

```
<scene>_<task>/        e.g. kitchen_microwave, cabinet_open_left, robot_toaster
```

Current scenes: `kitchen_*`, `cabinet_*`, `robot_*`.

## Folder contents

```
<task>/
├── rgb.png       # required — the RGB image
├── query.txt     # required — the task instruction, one line (e.g. "Open the drawer")
├── depth.npy     # optional — raw depth (mm, float32). If present → lingbot refines it.
└── cam_K.txt     # optional — 3x3 intrinsic. Required iff depth.npy is present.
```

- **No `depth.npy`** → Depth-Anything-3 generates depth from the RGB (default).
- **`depth.npy` + `cam_K.txt`** → lingbot refines your provided (e.g. RealSense) depth.

## Add a new demo task

```bash
mkdir examples/kitchen_faucet
cp /path/to/photo.png        examples/kitchen_faucet/rgb.png
echo "Turn on the faucet"  > examples/kitchen_faucet/query.txt

# then run everything (or just see it picked up):
python run_examples.py
```

That's it — drop in `rgb.png` + `query.txt` and re-run. Results appear in
`outputs/examples/kitchen_faucet/`.
