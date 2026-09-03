#!/usr/bin/env bash
# 重跑 CNN2 c8c16（可选 c4c8 对照）三路正则，验证高噪声 acc 是否复现。
#
# 背景：c8→c16 在 L2 / MNE L2 上 A_m(σ=1) 均值低于 c4→c8，主要因部分 seed 高噪声崩溃，
# 而非 clean acc 不足（σ=0 仍 ~98%）。
#
# 用法（在 QCFS_simulation 目录下）：
#   bash noise3_exp/RUN_cnn_c8c16_high_noise_verify_rerun.sh
#   bash noise3_exp/RUN_cnn_c8c16_high_noise_verify_rerun.sh --with-c4c8
#   bash noise3_exp/RUN_cnn_c8c16_high_noise_verify_rerun.sh --reg weight_decay --seed 42
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

export MNIST_ROOT="${MNIST_ROOT:-${HOME}/datasets/mnist}"
export CNN_NUM_WORKERS="${CNN_NUM_WORKERS:-8}"
export CNN_BATCH="${CNN_BATCH:-128}"
export CNN_EPOCHS="${CNN_EPOCHS:-100}"

ARCH_LIST=(c8c16)
if [[ "${1:-}" == "--with-c4c8" ]]; then
  ARCH_LIST=(c4c8 c8c16)
  shift
fi

LOG="${LOG:-${ROOT}/noise3_exp/cnn_c8c16_verify_rerun_$(date +%Y%m%d_%H%M%S).log}"

echo "[INFO] QCFS_simulation root: ${ROOT}" | tee -a "${LOG}"
echo "[INFO] MNIST_ROOT: ${MNIST_ROOT}" | tee -a "${LOG}"
echo "[INFO] arch-list: ${ARCH_LIST[*]}" | tee -a "${LOG}"
python -c 'import torch; print("[INFO] torch:", torch.__version__, "cuda:", torch.cuda.is_available())' | tee -a "${LOG}"

python -u "${SCRIPT_DIR}/run_cnn_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16.py" \
  --arch-list "${ARCH_LIST[@]}" \
  --retrain \
  --force-test \
  --replot \
  --font-size 18 \
  --legend-font-size 16 \
  "$@" 2>&1 | tee -a "${LOG}"

echo "[DONE] log: ${LOG}" | tee -a "${LOG}"
echo "[DONE] results: ${ROOT}/noise3_exp/cnn_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16/" | tee -a "${LOG}"
echo "[DONE] re-plot high-noise line chart from repo root:" | tee -a "${LOG}"
echo "  cd .. && python plot_high_noise_acc_vs_scale.py" | tee -a "${LOG}"
