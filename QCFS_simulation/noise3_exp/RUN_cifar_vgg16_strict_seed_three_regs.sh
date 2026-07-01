#!/usr/bin/env bash
# CIFAR-10 / CIFAR-100 VGG16：strict-seed 三路正则 (L2 / MNE L2 / No reg) + 噪声 mean±std
#
# 用法：
#   bash noise3_exp/RUN_cifar_vgg16_strict_seed_three_regs.sh cifar10
#   bash noise3_exp/RUN_cifar_vgg16_strict_seed_three_regs.sh cifar100
#   bash noise3_exp/RUN_cifar_vgg16_strict_seed_three_regs.sh cifar10 --force-test
#   bash noise3_exp/RUN_cifar_vgg16_strict_seed_three_regs.sh cifar10 --method no_regularization --retrain --force-test
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

DATASET="${1:-cifar10}"
shift || true

export CIFAR_ROOT="${CIFAR_ROOT:-${HOME}/datasets}"

echo "[INFO] QCFS_simulation root: ${ROOT}"
echo "[INFO] CIFAR_ROOT: ${CIFAR_ROOT}"
echo "[INFO] dataset: ${DATASET}"
python -c 'import torch; print("[INFO] torch:", torch.__version__, "cuda:", torch.cuda.is_available())'

python "${SCRIPT_DIR}/run_cifar_vgg16_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16.py" \
  --dataset "${DATASET}" "$@"

echo "[DONE] results: ${ROOT}/noise3_exp/${DATASET}_vgg16_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16/"
