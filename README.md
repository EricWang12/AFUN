

<h1 align="center">AFUN: Towards an Affordance Foundation Model for Functionality Understanding</h1>
<h3 align="center">Where + How for Functionality Understanding</h3>

<p align="center">
  Zhaoning Wang<sup>1,*</sup> &nbsp;·&nbsp; Yi Zhong<sup>1,*</sup> &nbsp;·&nbsp; Jiawei Fu<sup>2</sup> &nbsp;·&nbsp; Henrik I. Christensen<sup>2</sup> &nbsp;·&nbsp; Jun Gao<sup>1,3</sup>
</p>
<p align="center">
  <sup>1</sup>University of Michigan &nbsp;&nbsp; <sup>2</sup>UC San Diego &nbsp;&nbsp; <sup>3</sup>NVIDIA &nbsp;&nbsp;&nbsp; <sup>*</sup>Equal contribution
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.02551"><img src="https://img.shields.io/badge/arXiv-2606.02551-b31b1b.svg?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://www.zhaoningwang.com/AFUN/"><img src="https://img.shields.io/badge/Project-Page-1f8acb.svg?logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://huggingface.co/AFUN-dataset"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-ffd21e.svg" alt="Hugging Face Datasets"></a>
</p>
<p align="center"><img src="assets/afun_teaser.png" width="95%"></p>

**TL;DR**
```
RGB + text query + depth ──► AFUN model ──► mask + 3D motion curve
```


> From a single RGB observation and a natural-language task description, **AFUN** predicts a
> **task-conditional functional mask** (*where* to interact) and a **3D post-contact motion
> curve** (*how* to interact). It pairs frozen pretrained vision-language features with
> lightweight trainable modules for joint mask + motion prediction.

This repository provides **inference code** (load a trained AFUN checkpoint and run it on your own
images) and the **training code** for all three stages. Depth is handled automatically — [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)
**generates** metric depth from a single RGB image, or [lingbot-depth](https://github.com/robbyant/lingbot-depth)
**refines** a noisy sensor (e.g. RealSense) depth map if you provide one.


<details>
<summary><b>Contents</b></summary>

- [📋 TODO](#-todo)
- [🔧 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [📦 Dataset & Benchmark](#-dataset--benchmark)
- [📊 Evaluation](#-evaluation)
- [🏋️ Training](#-training)
- [🙏 Acknowledgements](#-acknowledgements)
- [📚 Citation](#-citation)

</details>

---

## 📋 TODO

- [x] Inference code &amp; demo 
- [x] Training code (Stages 1&ndash;3)
- [x] **Dataset, Evaluation, and Benchmark** — see [📦 Dataset & Benchmark](#-dataset--benchmark) and [📊 Evaluation](#-evaluation)
- [ ] **Robot Demonstration Code**
- [x] Claude Agent md Files ([`CLAUDE.md`](CLAUDE.md))

---

## 🔧 Installation

### 1. Clone (with submodules)

```bash
git clone --recursive https://github.com/EricWang12/AFUN.git
cd AFUN
# (already cloned non-recursively? run: git submodule update --init --recursive)
```

### 2. Create the environment

```bash
conda create -n afun python=3.11 -y
conda activate afun
bash install.sh
```
> NOTE: `install.sh` installs into the active Python 3.11 environment and targets CUDA 13
> (`cu130`). For a different CUDA build, edit the file directly.
>
> It downloads ~10 GB of wheels through `$TMPDIR` and the finished env is ~9 GB, so point
> `TMPDIR` at a volume with room if `/tmp` is small (`OSError: [Errno 28] No space left on
> device`). On a node without outbound network, set `HF_HUB_OFFLINE=1` — model construction
> contacts huggingface.co to revalidate even when every weight is already cached.


### 3. Get the checkpoint
SAM3 is gated on Hugging Face: request access at [facebook/sam3](https://huggingface.co/facebook/sam3)
and log in once with `hf auth login`.

The AFUN checkpoint `afun.pt` is hosted on **Google Drive**. Fetch it automatically —
this also pre-caches the Hugging Face backbone weights (Qwen3-VL-8B, SAM3, Sonata, and with
`--include-depth` Depth-Anything-3 and lingbot):
```bash
python scripts/prefetch_weights.py --include-depth
```
Or download it by hand from the [Google Drive link](https://drive.google.com/file/d/1J4TNj_w3b7BTaeriBPuDJCn7q_SlV6PR/view?usp=sharing) and place it at:
```
checkpoints/afun.pt
```
Backbone weights that are not cached download from Hugging Face on first use.

---

## 🚀 Quick Start


Run the demo with an RGB image, a text prompt, and (optionally) a sensor-depth folder:
```bash
python demo.py --rgb <path_to_rgb_image> --query <text_prompt> --depth-dir <path_to_depth_map>
```

`--depth-dir` would be a folder containing two files:
```
<depth-dir>/
├── depth.npy    # raw sensor depth — float32, millimeters, shape (H, W) matching the RGB
└── cam_K.txt    # 3×3 camera-intrinsics matrix
```
By default lingbot refines this depth; pass `--no-refine` to skip refinement and use the raw sensor depth as-is. Omit `--depth-dir`
entirely and Depth-Anything-3 estimates depth from the RGB instead.

> NOTE: Raw sensor depth without refinement generally does not work well with our model, as it is not trained on such data.

**Example 1 — sensor depth** (lingbot refines the bundled depth via `--depth-dir`):
```bash
python demo.py --rgb       examples/robot_toaster_oven/rgb.png \
               --query     "Open the toaster oven door" \
               --depth-dir examples/robot_toaster_oven
```

**Example 2 — Depth-Anything-3** (no `--depth-dir`; DA3 estimates depth from the RGB):
```bash
python demo.py --rgb examples/kitchen_fridge/rgb.png --query "Open the refrigerator door"
```


**Output** (per run, under `outputs/`):

| File | Contents |
|---|---|
| `pred_seg.png` | RGB + predicted affordance mask (red) + motion trajectory (yellow→blue, yellow = contact) |
| `pred.npz` | mask, motion parameters, intrinsics, confidence |
| `pred_3d.html` | interactive 3D point cloud + motion curve (omit with `--no-3d`) |
| `meta.json` / `summary.csv` / `montage.png` | run metadata (single / batch) |

**Run all bundled examples** (model loads once, then loops over `examples/`):
```bash
python run_examples.py            # add --no-3d to skip the interactive 3D HTML
```

---

## 📦 Dataset & Benchmark

All data is released on Hugging Face under the [**AFUN-dataset**](https://huggingface.co/AFUN-dataset)
organization (CC BY-NC-SA 4.0). Each data point is one folder: an RGB frame, its metric depth map,
the ground-truth affordance mask, the ground-truth 3D motion curve, and a language instruction.

| Dataset | What it is | Size |
|---|---|---:|
| [**AFUN**](https://huggingface.co/datasets/AFUN-dataset/AFUN) | Training set: **44,749** data points from SceneFun3D, RoboMIND, DROID, AgiBot World, RH20T, CALVIN, RLBench and VITRA human videos | 231 GiB |
| [**AFUN_eval**](https://huggingface.co/datasets/AFUN-dataset/AFUN_eval) | The three evaluation sets of Table 3: `afun_test` (121), `scenefun3d_test` (721), `robomind2_test` (156). The ADE / FDE / CIM protocol is described in its README | 9.9 GiB |
| [**AFUN_pool**](https://huggingface.co/datasets/AFUN-dataset/AFUN_pool) | The full **183,657**-point pool that [**AFUN**](https://huggingface.co/datasets/AFUN-dataset/AFUN) was curated from. Its 106,225 Ego4D-derived points ship as annotations only, with a script that rebuilds the frames from your own Ego4D download | 363 GiB |

```bash
pip install -U huggingface_hub
hf download AFUN-dataset/AFUN_eval --repo-type dataset --local-dir afun_eval    # 9.9 GiB — start here
hf download AFUN-dataset/AFUN      --repo-type dataset --local-dir afun_train   # 231 GiB
```

Each dataset's README documents the folder layout, the `trajectory.json` schema, and a loading snippet.
AFUN ships as `.tar.zst` archives; for training, `scripts/download_datasets.py afun-train` downloads
and extracts them into the layout the code expects (see [Training](#-training)).

---

## 📊 Evaluation

`python -m src.eval` runs the paper's benchmarks and writes one JSONL row per sample plus `summary.json`
to `outputs/eval/<date>_<time>/` (or `--output`). `--limit N` evaluates the first N samples of each set.

### 3D motion (Table 3)

```bash
python scripts/download_datasets.py afun-eval --root $AFUN_DATA_ROOT   # afun_test (121), scenefun3d_test (721), robomind2_test (156)
python -m src.eval --task motion --checkpoint checkpoints/afun.pt       # all three; or --dataset afun_test
```

Sets are read from `<--data-root>/afun_eval/<set>/` (default `$AFUN_DATA_ROOT`). The predicted curve starts at
the median 3D point of the predicted mask (back-projected with the depth map); prediction and ground truth
are linearly resampled to 50 points.

| Metric | Definition |
|---|---|
| ADE / FDE (m) | mean / final-point L2 distance; *abs* in the camera frame, *rel* with both curves shifted to start at the origin |
| CIM (%) | the curve start, projected into the image, lies inside the GT mask; failures count as misses |
| #fail (`num_failures`) | samples without a prediction; left out of ADE / FDE |

The released `afun.pt` is the Table 3 checkpoint; on `afun_test` it reproduces Table 3 to within 2 mm,
with the same CIM. For self-trained Stage-3 checkpoints add `--config inference_stage3`.

### Affordance segmentation (Table 1)

| `--dataset` | Data under `$AFUN_DATA_ROOT` | Samples | Query |
|---|---|---:|---|
| `handal-mini`, `3doi` | RAGNet `handal_mini_val.pkl`, `3doi_val.pkl` | 1003, 1013 | "Please segment the affordance map of {class} in this image" |
| `handal-easy`, `handal-hard`, `3doi-easy` | RAGNet `*_reasoning_val.pkl` | 1003, 1003, 1013 | question + answer |
| `hova500k` | `HOVA-500K/mask_annotation/dataset_index.json`, `test` split | 419 | task instruction |
| `reasonaff` | ReasonAFF `test_new` | 600 | `problem` |
| `instruct-part` | `instruct_part/InstructPart/test` | 600 | first instruction |

```bash
python -m src.eval --task segmentation --checkpoint checkpoints/afun.pt   # all eight sets
```

The prediction is the top-scoring SAM3 detection above confidence 0.5, compared with the GT mask at the
image's resolution. gIoU averages per-sample IoU; cIoU divides summed intersections by summed unions.
As in the evaluation behind the paper's numbers, samples without a detection above the threshold are
left out of both and counted in `num_failures`. Table 1 was computed with an earlier checkpoint; the
released `afun.pt` is close on average but differs per set, by up to 1.5 points gIoU and 7 points cIoU
(HOVA-500K). HOVA-500K masks regenerated with `prepare_hova.py` also differ from ours, so
`hova500k` numbers will differ from Table 1.

---

## 🏋️ Training

AFUN is trained in three stages (Sec. 4.2 of the paper). Each stage starts from the weights of the
previous one; Qwen3-VL, SAM3, and Sonata stay frozen throughout.

| Stage | Config | What it learns | Data | Batch | Steps |
|---|---|---|---|---|---|
| 1 | `train_stage1` | MetaQuery &rarr; SAM3 alignment: SAM3 driven by the projected MetaQueries is regressed (MSE) onto SAM3 driven by its own text encoder | Visual Genome | 196 | 10k |
| 2 | `train_stage2` | Task-conditional segmentation (SAM3 grounding loss) | HOVA-500K, RAGNet, InstructPart, ReasonAFF | 128 | 40k |
| 3 | `train_stage3` | Functional mask + 3D motion curve, jointly | AFUN robot dataset + the Stage-2 data | 96 | 20k |

Batch sizes are totals over 4 GPUs, i.e. 49 / 32 / 24 per GPU; Stage 1 reaches its 49 as 7 samples per
pass with 7 gradient-accumulation steps. Every stage uses Adam with a constant learning rate of 2e-4,
bf16 autocast, and gradient checkpointing on Qwen3-VL.
The paper's runs took about eight days on 4&times; GH200.

### 1. Get the data

All datasets live under one root, `$AFUN_DATA_ROOT` (default `./data`); `configs/paths.yaml` maps
each dataset to its folder.

```bash
export AFUN_DATA_ROOT=/path/to/afun_datasets
export AFUN_OUTPUT_DIR=/path/to/run_outputs     # optional, default: ./outputs
```

```
$AFUN_DATA_ROOT/
├── visual_genome/        # Stage 1
├── HOVA-500K/            # Stages 2–3
├── RAGNet/data/data/     # Stages 2–3
├── instruct_part/        # Stages 2–3
├── reasonaff/            # Stages 2–3
└── afun_train/           # Stage 3
```

| Dataset | Stages | Training samples | Source |
|---|---|---|---|
| Visual Genome | 1 | 5.4M region captions on 108k images (~15 GB) | [jn12/VisualGenome](https://huggingface.co/datasets/jn12/VisualGenome) |
| HOVA-500K | 2, 3 | 87,227 | [JiaaZ/HOVA-500K](https://huggingface.co/datasets/JiaaZ/HOVA-500K) + masks from `prepare_hova.py` |
| RAGNet | 2, 3 | 37,538 | [wudongming/RAGNet](https://huggingface.co/datasets/wudongming/RAGNet) + HANDAL |
| ReasonAFF | 2, 3 | 2,400 | [hqking/affordance-R1](https://huggingface.co/datasets/hqking/affordance-R1) (gated) |
| InstructPart | 2, 3 | 1,800 | [Authors' Google Drive](https://drive.google.com/drive/folders/1b876tX1wdy-jbyLvGSmZkAS4U5DuXlST) |
| AFUN robot dataset | 3 | 44,749 (~231 GB) | [AFUN-dataset/AFUN](https://huggingface.co/datasets/AFUN-dataset/AFUN) |

`scripts/download_datasets.py` fetches everything except InstructPart into the expected folders:

```bash
python scripts/download_datasets.py visual-genome --root $AFUN_DATA_ROOT
python scripts/download_datasets.py hova500k      --root $AFUN_DATA_ROOT   # 3doi, Ego4D, HANDAL (~233 GB)
python scripts/download_datasets.py ragnet        --root $AFUN_DATA_ROOT
python scripts/download_datasets.py reasonaff     --root $AFUN_DATA_ROOT
python scripts/download_datasets.py afun-train    --root $AFUN_DATA_ROOT
```

The downloaded archives (each dataset's `.downloads/` folder and `RAGNet/data.zip`) can be deleted
after extraction.

- **ReasonAFF** is gated: accept its terms on the dataset page and run `hf auth login` first.
- **RAGNet** does not ship the HANDAL images and part masks. The HOVA-500K download includes them;
  link them in (or point `--handal-root` at an official [HANDAL](https://github.com/NVlabs/HANDAL) release):
  ```bash
  python scripts/prepare_ragnet.py $AFUN_DATA_ROOT/RAGNet/data/data \
      --handal-root $AFUN_DATA_ROOT/HOVA-500K/HANDAL/images
  ```
  Only records with a task instruction are used (EgoObjects and HANDAL reasoning, OpenX); the
  annotations that only name the object class are skipped.
- **HOVA-500K** provides contact points, not masks. We converted them into masks and task
  instructions with Qwen3-VL + SAM3 (Appendix B.4). Our converted set is not public yet;
  `prepare_hova.py` reruns the conversion on a GPU, so its masks and instructions will differ from ours:
  ```bash
  python scripts/prepare_hova.py $AFUN_DATA_ROOT/HOVA-500K     # writes HOVA-500K/mask_annotation/
  ```
  It runs Qwen3-VL twice and SAM3 once on each of the ~470k contact annotations, which takes days
  on one GPU; rerunning it resumes where it stopped.
  To train without HOVA-500K, add `'~data.datasets.hova500k'` (Stage 2) or
  `'~data.segmentation.datasets.hova500k'` (Stage 3) to the launch command.
- **InstructPart**: the Hugging Face mirror lacks the annotation files, and Google Drive throttles
  scripted downloads, so download it by hand from the authors' Google Drive folder. Place
  `InstructPart/train/train1800` (and `InstructPart/test` for evaluation) so that
  `$AFUN_DATA_ROOT/instruct_part/InstructPart/train/train1800/` contains `images/`, `masks/`, and
  `data_train.json`.
- **AFUN robot dataset**: `--source <name>` downloads a single source (SceneFun3D is 89% of the
  samples). Each sample is a folder with `obs_frame.png`, `obs_frame_depth.npy` (float32, mm),
  `sam_mask.png`, and `trajectory.json`; the instructions are in `afun_train/manifest.json`.
  `python scripts/prepare_robot_data.py $AFUN_DATA_ROOT/afun_train` checks the extraction.

Check that a dataset loads, without loading any model:

```bash
python scripts/validate_data.py --dataset ragnet --root $AFUN_DATA_ROOT/RAGNet/data/data
```

**Mixtures.** Stage 2 draws one dataset per step with probability proportional to its size, except
HOVA-500K, which counts as 50,000 samples (&asymp;55% HOVA-500K, 41% RAGNet, 3% ReasonAFF, 2%
InstructPart). Stage 3 picks robot or segmentation batches at random (1:1), and segmentation batches
follow dataset size (&asymp;68% HOVA-500K). Without HOVA-500K, RAGNet supplies &asymp;90% of the
segmentation batches.

### 2. Train

```bash
torchrun --standalone --nproc_per_node=4 -m src.main --config-name train_stage1
torchrun --standalone --nproc_per_node=4 -m src.main --config-name train_stage2 \
    train.ckpt_path=$AFUN_OUTPUT_DIR/afun-stage1/<run>/model_step_10000.pt
torchrun --standalone --nproc_per_node=4 -m src.main --config-name train_stage3 \
    train.ckpt_path=$AFUN_OUTPUT_DIR/afun-stage2/<run>/model_step_40000.pt
```

`train.ckpt_path` loads the previous stage's weights and starts at step 0 with a fresh optimizer
(without the override, Stages 2 and 3 look for `checkpoints/afun_stage1.pt` and
`checkpoints/afun_stage2.pt`). Each run writes `model_step_<N>.pt` every 1,000 steps to
`$AFUN_OUTPUT_DIR/afun-stage<k>/<date>_<time>/`.

Useful overrides:

```bash
# smaller GPUs: effective batch = batch_size × GPUs × grad_accumulation_steps
torchrun ... train.batch_size=12 train.grad_accumulation_steps=2
# resume an interrupted run: restores weights, optimizer state, and step
torchrun ... train.resume_path=$AFUN_OUTPUT_DIR/afun-stage3/<run>/model_step_12000.pt
# log to Weights & Biases
torchrun ... train.use_wandb=true
# keep only the latest three checkpoints (each is ~4 GB)
torchrun ... train.training_cfg.rolling_ckpt=true
```

> **GPU memory.** The defaults fit on 96 GB GPUs: on one RTX PRO 6000 (96 GB) Stages 1–3 peaked at
> about 39 / 78 / 81 GB. On smaller GPUs, divide `train.batch_size` and multiply
> `train.grad_accumulation_steps` by the same factor.

> **Tested configuration.** Every stage was run on a single GPU, including resuming Stage 3 from a
> checkpoint. The multi-GPU path is exercised only by a two-process CPU test (`tests/test_distributed.py`).

### 3. Run your model

```bash
python demo.py --rgb examples/kitchen_fridge/rgb.png --query "Open the refrigerator door" \
    --ckpt $AFUN_OUTPUT_DIR/afun-stage3/<run>/model_step_20000.pt --config inference_stage3
```

`inference_stage3` builds the 6-layer motion head that `train_stage3` trains. The released `afun.pt`
has a 4-layer motion head and uses the default `inference` config.

### What is trained

- the 64 MetaQuery token embeddings (32 semantic, 32 motion), two boundary tokens, and their RMSNorm
- the Qwen&rarr;SAM3 projector
- Stage 3 only: the spline motion head and the projection on top of the frozen Sonata encoder

Checkpoints store only these tensors plus the Adam state (~4 GB), not the frozen backbones.

---

## 🙏 Acknowledgements

This project builds on [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL),
[SAM3](https://github.com/EricWang12/sam3),
[Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3),
[lingbot-depth](https://github.com/robbyant/lingbot-depth), and
[Sonata](https://github.com/facebookresearch/sonata). We thank the authors for releasing their
code and models.

---

## 📚 Citation

```bibtex
@article{wang2026afun,
  title   = {{AFUN}: Towards an Affordance Foundation Model for Functionality Understanding},
  author  = {Wang, Zhaoning and Zhong, Yi and Fu, Jiawei and Christensen, Henrik I. and Gao, Jun},
  journal = {arXiv preprint arXiv:2606.02551},
  year    = {2026},
}
```
