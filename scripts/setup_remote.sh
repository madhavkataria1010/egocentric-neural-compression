#!/usr/bin/env bash
# One-shot setup on the A40 box. Idempotent.
#
#   ssh kartik@<host>
#   cd /home/kartik/madhav/Industry
#   git clone <repo-url> ego-codec && cd ego-codec
#   bash scripts/setup_remote.sh

set -euo pipefail

PYTHON="${PYTHON:-python3.10}"
VENV="${VENV:-.venv}"

if [[ ! -d "$VENV" ]]; then
  echo "[+] Creating venv at $VENV"
  "$PYTHON" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "[+] Installing torch + deps"
pip install --upgrade pip wheel
# Adjust to your CUDA version. A40 supports CUDA 11.8 / 12.1+.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[aria,dev]"

echo "[+] Sanity check (synthetic data, no Aria download needed)"
ARIA_SYNTHETIC=1 pytest -q tests/test_shapes.py

echo
echo "Next:"
echo "  1. Get an Aria URL list from https://www.projectaria.com/datasets/ (sign EULA)."
echo "  2. export ARIA_URLS_FILE=/path/to/urls.txt"
echo "  3. ./scripts/download_aria.sh   # downloads + preprocesses ~5 sequences"
echo "  4. bash scripts/train_all.sh    # full training sweep"
