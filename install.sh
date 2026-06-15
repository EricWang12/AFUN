#!/usr/bin/env bash
# AFUN setup. Run from the repo root: bash install.sh
set -e

# Conda environment
conda create -y -n afun python=3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate afun

# PyTorch (CUDA 13)
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130

# Python dependencies
pip install -r requirements.txt

# xformers needs the wheel built against the SAME torch+CUDA as above. PyPI only
# carries the cu128 build, so we pull it from the cu130 index instead. (Hence it
# is NOT in requirements.txt.) Change the index URL if you change torch/CUDA.
pip install --no-deps --index-url https://download.pytorch.org/whl/cu130 xformers==0.0.35

# Sonata depth-encoder compiled deps (both are torch/CUDA-version-specific).
pip install spconv-cu124            # CUDA-12.4 build; works with the cu13 runtime above

# torch-scatter must match your installed torch + CUDA build. Install the matching
# PRE-BUILT wheel (no CUDA toolkit / compiler needed): pick the index URL for YOUR
# torch+CUDA from https://data.pyg.org/whl/ . The one below matches the torch 2.10.0 +
# cu130 pinned above — change it if you change the torch / CUDA versions.
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.10.0+cu130.html
# Fallback — ONLY if no pre-built wheel exists for your combo, build from source. That
# needs a CUDA toolkit (nvcc) matching your torch CUDA build; point CUDA_HOME at it:
#   CUDA_HOME=/path/to/cuda pip install --no-build-isolation torch_scatter

# SAM3 (git submodule, editable)
git submodule update --init --recursive
pip install -e src/models/decoder/sam3 --no-deps
