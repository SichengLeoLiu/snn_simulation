#!/usr/bin/env bash
# MNIST CNN2 多架构 strict-seed 三路正则 + rate_uniform 噪声 mean±std 实验
#
# 模型：c2c4 / c4c8 / c8c16 / c16c32
# 方法：mne_l2 / weight_decay / no_regularization
# 默认 5 seeds (40–44)，L=16, T=16, sigma=0~1 step=0.1
#
# 用法：
#   bash noise3_exp/RUN_cnn_strict_seed_three_regs_rate_uniform.sh
#   bash noise3_exp/RUN_cnn_strict_seed_three_regs_rate_uniform.sh --arch-list c2c4 c8c16
#   bash noise3_exp/RUN_cnn_strict_seed_three_regs_rate_uniform.sh --reg mne_l2 --seed 42
#   bash noise3_exp/RUN_cnn_strict_seed_three_regs_rate_uniform.sh --plot-only --copy-important
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

export MNIST_ROOT="${MNIST_ROOT:-${HOME}/datasets/mnist}"
export CNN_NUM_WORKERS="${CNN_NUM_WORKERS:-8}"
export CNN_BATCH="${CNN_BATCH:-128}"
export CNN_EPOCHS="${CNN_EPOCHS:-100}"

echo "[INFO] QCFS_simulation root: ${ROOT}"
echo "[INFO] MNIST_ROOT: ${MNIST_ROOT}"
echo "[INFO] CNN_EPOCHS=${CNN_EPOCHS} CNN_BATCH=${CNN_BATCH} CNN_NUM_WORKERS=${CNN_NUM_WORKERS}"
python -c 'import torch; print("[INFO] torch:", torch.__version__, "cuda:", torch.cuda.is_available())'

python "${SCRIPT_DIR}/run_cnn_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16.py" \
  --replot \
  --copy-important \
  --font-size 18 \
  --legend-font-size 16 \
  "$@"

echo "[DONE] results: ${ROOT}/noise3_exp/cnn_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16/"
echo "[DONE] important: ${ROOT}/../important results/"
