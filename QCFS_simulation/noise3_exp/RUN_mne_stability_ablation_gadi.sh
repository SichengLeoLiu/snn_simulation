#!/bin/bash
set -euo pipefail

# Run this after entering an interactive Gadi GPU job and loading NCI-ai-ml/24.11.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

export PYTHONUNBUFFERED=1
export CIFAR_ROOT="${CIFAR_ROOT:-/scratch/gs14/sl9144/datasets}"
export MNIST_ROOT="${MNIST_ROOT:-/scratch/gs14/sl9144/datasets}"
export CIFAR_NUM_WORKERS="${CIFAR_NUM_WORKERS:-8}"
export CIFAR_BATCH="${CIFAR_BATCH:-128}"
export CIFAR_EPOCHS="${CIFAR_EPOCHS:-300}"

DATASETS=${DATASETS:-"cifar10"}
VARIANTS=${VARIANTS:-"old_detach fanin_mean full_bn"}
RCS=${RCS:-"1e-4 3e-4 1e-3 3e-3 1e-2"}
SEEDS=${SEEDS:-"40 41 42 43 44"}
NOISE_POSITION=${NOISE_POSITION:-"post_input_if"}
OUT_ROOT=${OUT_ROOT:-"../important_results/mne_stability_ablation_${NOISE_POSITION}"}
REG_WARMUP_EPOCHS=${REG_WARMUP_EPOCHS:-0}

python -u noise3_exp/run_mne_stability_ablation.py \
  --datasets ${DATASETS} \
  --variants ${VARIANTS} \
  --rcs ${RCS} \
  --seeds ${SEEDS} \
  --epochs "${CIFAR_EPOCHS}" \
  --batch-size "${CIFAR_BATCH}" \
  --workers "${CIFAR_NUM_WORKERS}" \
  --first-layer-noise-position "${NOISE_POSITION}" \
  --reg-warmup-epochs "${REG_WARMUP_EPOCHS}" \
  --out-root "${OUT_ROOT}"
