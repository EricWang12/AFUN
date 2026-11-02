# CLAUDE.md

Guidance for Claude Code and other coding agents working in this repository. The README is the
user-facing guide (installation, data download, training); this file covers what an agent needs to
change the code safely.

## Project

AFUN takes one RGB image and a task instruction (plus depth for motion) and predicts a **functional
mask** (where to interact) and a **3D post-contact motion curve** (how). Paper: arXiv 2606.02551.

## Environment

- Python 3.11, CUDA 13 (`cu130`): `conda create -n afun python=3.11 -y && conda activate afun && bash install.sh`.
  `install.sh` installs pinned versions and ends with `pip check`.
- Keep NumPy below 2: the SAM3 submodule requires it. This is also why `pyarrow` is pinned to 25.0.1
  (pyarrow 26 refuses to import with NumPy < 2, and `pip check` does not catch it). Check both before
  changing any pin in `requirements.txt`.
- SAM3 (`facebook/sam3`) and ReasonAFF (`hqking/affordance-R1`) are gated on Hugging Face: the user must
  request access and run `hf auth login`. Set `HF_HUB_OFFLINE=1` on nodes without internet access.
- Dev tools: `pip install -r requirements-dev.txt` (pytest, ruff).

## Commands

```bash
# Inference (released checkpoint: checkpoints/afun.pt, fetched by scripts/prefetch_weights.py)
python demo.py --rgb examples/kitchen_fridge/rgb.png --query "Open the refrigerator door"
python demo.py --rgb examples/robot_toaster_oven/rgb.png --query "Open the toaster oven door" \
    --depth-dir examples/robot_toaster_oven          # sensor depth, refined by lingbot
python run_examples.py

# Tests and lint: run both before finishing a change
pytest
ruff check .

# Training (Hydra; override any config value on the command line)
torchrun --standalone --nproc_per_node=4 -m src.main --config-name train_stage1
torchrun --standalone --nproc_per_node=4 -m src.main --config-name train_stage2 train.ckpt_path=<stage-1 checkpoint>
torchrun --standalone --nproc_per_node=4 -m src.main --config-name train_stage3 train.ckpt_path=<stage-2 checkpoint>
# Quick functional check of a stage on one GPU
torchrun --standalone --nproc_per_node=1 -m src.main --config-name train_stage3 train.ckpt_path=null \
    train.batch_size=4 train.training_cfg.max_steps=20 train.training_cfg.save_ckpt_every=0

# Data
python scripts/download_datasets.py <visual-genome|hova500k|ragnet|reasonaff|afun-train|afun-eval> --root $AFUN_DATA_ROOT
python scripts/validate_data.py --dataset <visual_genome|instruct_part|ragnet|hova500k|reasonaff|robot> --root <dataset dir>

# Evaluation (README: Evaluation); writes JSONL + summary.json under outputs/eval/
python -m src.eval --task segmentation --checkpoint checkpoints/afun.pt [--dataset instruct-part] [--limit 2]
python -m src.eval --task motion --checkpoint checkpoints/afun.pt [--dataset afun_test] [--limit 2]
```

Runs write to `$AFUN_OUTPUT_DIR/afun-stage<k>/<date>_<time>/` (default `./outputs`). Useful overrides:
`train.batch_size`, `train.grad_accumulation_steps`, `train.resume_path`, `train.use_wandb=true`,
`train.training_cfg.rolling_ckpt=true`. Use `HYDRA_FULL_ERROR=1` when debugging config errors.

## Architecture

```
image + instruction --> Qwen3-VL-8B (frozen) + 64 MetaQuery tokens (32 semantic, 32 motion)
                            | hidden states at the MetaQuery positions, RMSNorm
              +-------------+--------------------+
      semantic queries                     motion queries
      projector (MLP 4096 -> 1024 -> 256)        |
      SAM3 (frozen) --> masks, boxes --> spline motion head <-- Sonata (frozen) depth features
                                         (cubic Bezier, 3 control points)
```

Trainable: the MetaQuery embedding rows plus two boundary tokens (a gradient hook masks the rest of the
embedding table), their RMSNorm, the projector, and in Stage 3 the motion head and the projection on top
of Sonata. Qwen3-VL, SAM3, and Sonata stay frozen.

| Path | Role |
|---|---|
| `src/main.py` | Hydra entry point for all stages; Stage 1 builds `AlignmentModel` |
| `src/train/trainer.py`, `src/train/base.py` | Training loop (bf16 autocast, Adam, gradient accumulation, DDP) and checkpoint save/load |
| `src/models/build.py` | Builds the same model for training, evaluation, and inference |
| `src/models/end_to_end.py` | Qwen3-VL -> SAM3 decoder (+ motion head) |
| `src/models/alignment.py` | Stage 1: MSE between SAM3 driven by projected MetaQueries and by its own text encoder |
| `src/models/llm/qwen_wrapper.py` | Qwen3-VL wrapper: MetaQuery tokens, embedding gradient mask, RMSNorm |
| `src/models/decoder/sam3_decoder.py` | SAM3 grounding and loss, motion head call |
| `src/models/decoder/motion_decoder/` | `spline_processor.py` (motion head, curve loss), `sonata_geo_encoder.py` |
| `src/data/` | One loader per dataset; `sam3_utils.collate_to_sam3` builds SAM3 targets; `seg_multi_source.py` and `combined.py` mix sources per step |
| `src/utils/checkpoint.py` | `load_weights`: loads non-strictly but rejects checkpoints missing trainable tensors |
| `src/inference.py`, `demo.py`, `run_examples.py` | Inference and visualization |
| `src/eval/` | Benchmarks: gIoU/cIoU for masks, ADE/FDE/CIM for motion |
| `scripts/` | Weight and dataset download, preparation, validation |
| `src/models/decoder/sam3/` | SAM3 fork (git submodule, installed editable) |

## Configs

Hydra, rooted at `configs/`. Top level: `train_stage1.yaml`, `train_stage2.yaml`, `train_stage3.yaml`;
`inference.yaml` (the released `afun.pt`); `inference_stage3.yaml` (checkpoints from `train_stage3`);
`eval_segmentation.yaml`. Groups: `model/llm`, `model/decoder` (+ `motion/`, `loss/`), `data/` (one file
per dataset; `multi_source.yaml` is the Stage-2 mix, `combined.yaml` the Stage-3 mix,
`default/loader.yaml` the DataLoader settings), `train/trainer.yaml`. Dataset and output roots come from
`configs/paths.yaml` (`AFUN_DATA_ROOT`, `AFUN_OUTPUT_DIR`).

| Stage | Data | Per-GPU batch | Steps |
|---|---|---|---|
| 1 | Visual Genome region captions | 49 (7 per pass x 7 accumulation; 49 at once exceeds 96 GB) | 10k |
| 2 | HOVA-500K, RAGNet, InstructPart, ReasonAFF | 32 | 40k |
| 3 | AFUN robot dataset + Stage-2 data (1:1 per step) | 24 | 20k |

All stages use Adam at a constant 2e-4 and gradient checkpointing on Qwen3-VL.

## Checkpoints

- Checkpoints hold only the trainable tensors plus the Adam state (~4 GB each).
- `train.ckpt_path` initializes weights and starts at step 0; `train.resume_path` restores weights,
  optimizer state, and step, and continues on the next epoch's shuffle rather than re-reading the
  batches already consumed. Both fail fast if the file is missing.
- `checkpoints/`, `*.pt`, `outputs/`, and the repo-root `/data/` are gitignored. `src/data/` and
  `configs/data/` are code.

## Intentional behavior: matches the original training code

These look like bugs but reproduce how AFUN was trained. Changing them changes model behavior and
requires retraining; keep them unless asked otherwise.

- Qwen3-VL receives float images in [0, 1], which the Hugging Face processor rescales by 1/255 again,
  so Qwen effectively sees a black image. SAM3 and Sonata receive the real image and depth.
- The MetaQuery RMSNorm trains in fp32 in Stage 1 (`build_model(..., alignment=True)`); Stages 2-3 keep
  it in bf16, where Adam updates near 1.0 round away, so it stays at its Stage-1 values.
- `sam_loss_weight` (0.5 in Stage 3) scales the SAM3 loss only on robot batches; segmentation batches
  keep the full loss.
- Stage 3 draws segmentation batches in proportion to dataset size (`segmentation.mixing.fixed_steps: 0`),
  not with Stage 2's HOVA-500K weight.
- RAGNet uses only records with a task instruction (37,538); class-name-only annotations are skipped.
- `src/eval` reproduces the code behind the paper's tables: the motion benchmark gives SAM3 the
  unnormalized image and runs the motion head without depth tokens, and segmentation samples without a
  detection are left out of gIoU/cIoU.
- The released `afun.pt` has a 4-layer motion head; `train_stage3` trains the paper's 6 layers. Use
  `--config inference_stage3` for self-trained checkpoints, in `demo.py` and `src.eval`.

## Data notes

- AFUN robot data (`afun_train/`): one folder per sample with `obs_frame.png`, `obs_frame_depth.npy`
  (float32, mm), `sam_mask.png`, and `trajectory.json`; instructions live in `manifest.json`. Curve
  targets come from `spline_params` (16 points per curve). Mask components are paired with trajectories;
  the ~0.2% of samples that cannot be paired are skipped even with `strict: true`.
- HOVA-500K needs masks generated by `scripts/prepare_hova.py` (Qwen3-VL + SAM3, days on one GPU;
  rerunning resumes, skipping contacts already accepted or listed in `rejected.txt`).
  RAGNet needs the HANDAL images and masks linked by `scripts/prepare_ragnet.py`. InstructPart is
  downloaded by hand. The README has the exact steps.
- DataLoader workers use fork: spawn re-imports SAM3 in every worker and is very slow on network
  filesystems.
