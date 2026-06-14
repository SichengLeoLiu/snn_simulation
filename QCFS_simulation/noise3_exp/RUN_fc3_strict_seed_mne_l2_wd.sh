#!/usr/bin/env bash
# MNIST fc3 strict-seed：mne_l2+wd 多 seed + 合并到已有 mean±std 折线图
#
# 用法：
#   bash noise3_exp/RUN_fc3_strict_seed_mne_l2_wd.sh
#   bash noise3_exp/RUN_fc3_strict_seed_mne_l2_wd.sh --h-list 8 16 32 128 --seed 42
#   bash noise3_exp/RUN_fc3_strict_seed_mne_l2_wd.sh --plot-only --copy-important
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

python "${SCRIPT_DIR}/run_fc3_strict_seed_mne_l2_wd_noise_sweep_normal_L16_T16.py" "$@"

echo "[DONE] plots: ${ROOT}/noise3_exp/ablation_mne_l2_vs_weight_decay_l16_fc3_h4_h8_h16_h32_h64_h128/strict_seed_train_normal_L16_T16/"
