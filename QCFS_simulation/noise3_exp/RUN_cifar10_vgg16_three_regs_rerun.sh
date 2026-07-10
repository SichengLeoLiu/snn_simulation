#!/usr/bin/env bash
# CIFAR-10 VGG16：三路正则重训 + rate_uniform 噪声扫描（5 seeds，不覆盖旧实验）
#
# 新 checkpoint 后缀含 drs_rerun；结果写入 important_results/cifar10_vgg16_three_regs_drs_rerun
#
# Gadi:
#   cd ~/codes/snn_simulation/QCFS_simulation
#   module load NCI-ai-ml/24.11
#   export CIFAR_ROOT=/scratch/gs14/sl9144/datasets
#   export CIFAR_BATCH=128 CIFAR_NUM_WORKERS=4
#   LOG=~/cifar10_vgg16_rerun_$(date +%Y%m%d_%H%M%S).log
#   nohup bash noise3_exp/RUN_cifar10_vgg16_three_regs_rerun.sh > "$LOG" 2>&1 &
#   echo "PID=$! LOG=$LOG"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

export CIFAR_ROOT="${CIFAR_ROOT:-${HOME}/datasets}"
export CIFAR_BATCH="${CIFAR_BATCH:-128}"
export CIFAR_NUM_WORKERS="${CIFAR_NUM_WORKERS:-4}"
export CIFAR_EPOCHS="${CIFAR_EPOCHS:-300}"

RUN_TAG="${RUN_TAG:-drs_rerun}"
OUT_DIR="${OUT_DIR:-../important_results/cifar10_vgg16_three_regs_drs_rerun}"

echo "[INFO] QCFS_simulation root: ${ROOT}"
echo "[INFO] CIFAR_ROOT: ${CIFAR_ROOT}"
echo "[INFO] RUN_TAG: ${RUN_TAG}"
echo "[INFO] OUT_DIR: ${OUT_DIR}"
python -c 'import torch; print("[INFO] torch:", torch.__version__, "cuda:", torch.cuda.is_available())'

python -u "${SCRIPT_DIR}/run_cifar_vgg16_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16.py" \
  --dataset cifar10 \
  --seeds 40 41 42 43 44 \
  --retrain \
  --force-test \
  --run-tag "${RUN_TAG}" \
  --out-dir "${OUT_DIR}"

echo "[DONE] checkpoints: ${ROOT}/cifar10-checkpoints/ (suffix contains ${RUN_TAG})"
echo "[DONE] results: ${OUT_DIR}"
