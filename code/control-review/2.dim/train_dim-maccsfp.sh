#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=python
TRAIN_PY="/data/yinghuazhang/MolF-DAEs/code/control-review/2.dim/train_dim-maccsfp.py"

for dim in 2 3 4 8 16 32; do
  CUDA_VISIBLE_DEVICES=0 \
  "${PYTHON_BIN}" "${TRAIN_PY}" \
  "${dim}"
done