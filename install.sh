#!/usr/bin/env bash
# Install into an activated Python 3.11 environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11."'
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install xformers==0.0.34 --index-url https://download.pytorch.org/whl/cu130
python -m pip install spconv-cu124==2.3.8
python -m pip install --only-binary=:all: torch_scatter==2.1.2+pt210cu130 \
    --find-links https://data.pyg.org/whl/torch-2.10.0+cu130.html
if [[ ! -f src/models/decoder/sam3/sam3/model_builder.py ]]; then
    git submodule update --init --recursive
fi
python -m pip install --no-deps -e src/models/decoder/sam3
python -m pip check
python -c 'import torch, sam3, spconv, torch_scatter; print(f"AFUN ready: PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")'
