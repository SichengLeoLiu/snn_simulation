#!/usr/bin/env bash
# Fashion-MNIST: FC3rev(h8~h256) 三路正则 + rate_uniform 噪声扫描
#
# 默认：
# - h: 8 16 32 64 128 256
# - seeds: 40 41 42 43 44
# - regs: mne_l2 / weight_decay / no_regularization
# - 训练 T=0, 测试 T=16, sigma=0~1 step=0.05
#
# 通过环境变量覆盖：
#   FASHION_FC3REV_H_LIST="8 16 32"
#   FASHION_FC3REV_SEEDS="40 41"
#   FASHION_FC3REV_REGS="mne_l2 weight_decay"
#   FASHION_FC3REV_EPOCHS=50
#   FC3REV_BATCH=128
#   FC3REV_NUM_WORKERS=0
#   CKPT_SAVE_MODE=best|last
#   RETRAIN=1
#   FORCE_TEST=1
#   OUT_ROOT=../important_results/fashion_mnist_fc3rev_three_regs/noise_sweep_rate_uniform_L16_T16
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

export MNIST_ROOT="${MNIST_ROOT:-${HOME}/datasets}"
export FC3REV_BATCH="${FC3REV_BATCH:-128}"
export FC3REV_NUM_WORKERS="${FC3REV_NUM_WORKERS:-0}"
export FASHION_FC3REV_EPOCHS="${FASHION_FC3REV_EPOCHS:-50}"
export CKPT_SAVE_MODE="${CKPT_SAVE_MODE:-best}"
export RETRAIN="${RETRAIN:-0}"
export FORCE_TEST="${FORCE_TEST:-0}"
export OUT_ROOT="${OUT_ROOT:-../important_results/fashion_mnist_fc3rev_three_regs/noise_sweep_rate_uniform_L16_T16}"

H_LIST=(${FASHION_FC3REV_H_LIST:-8 16 32 64 128 256})
SEEDS=(${FASHION_FC3REV_SEEDS:-40 41 42 43 44})
REGS=(${FASHION_FC3REV_REGS:-mne_l2 weight_decay no_regularization})

echo "[INFO] ROOT=${ROOT}"
echo "[INFO] MNIST_ROOT=${MNIST_ROOT}"
echo "[INFO] H_LIST=${H_LIST[*]} SEEDS=${SEEDS[*]} REGS=${REGS[*]}"
echo "[INFO] EPOCHS=${FASHION_FC3REV_EPOCHS} BATCH=${FC3REV_BATCH} WORKERS=${FC3REV_NUM_WORKERS}"
echo "[INFO] CKPT_SAVE_MODE=${CKPT_SAVE_MODE} RETRAIN=${RETRAIN} FORCE_TEST=${FORCE_TEST}"
python -c 'import torch; print("[INFO] torch:", torch.__version__, "cuda:", torch.cuda.is_available())'

for h in "${H_LIST[@]}"; do
  arch="fc3rev_h${h}"
  for reg in "${REGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      if [[ "${reg}" == "mne_l2" ]]; then
        regularizer="mne_l2"; wd="0.0"; rc="5e-2"
        suffix="fashion_strict_seed${seed}_ablation_mne_l2_l16_${arch}_rc5em02"
      elif [[ "${reg}" == "weight_decay" ]]; then
        regularizer="weight_decay"; wd="5e-4"; rc="1.0"
        suffix="fashion_strict_seed${seed}_ablation_wd_l16_${arch}"
      else
        regularizer="weight_decay"; wd="0.0"; rc="1.0"
        suffix="fashion_strict_seed${seed}_ablation_none_l16_${arch}"
      fi

      ckpt="${ROOT}/fashion_mnist-checkpoints/${arch}_L[16]_${suffix}.pth"
      out_dir="${OUT_ROOT}/${arch}/${reg}/seed_${seed}"

      if [[ "${RETRAIN}" == "1" && -f "${ckpt}" ]]; then
        rm -f "${ckpt}"
      fi
      if [[ "${FORCE_TEST}" == "1" && -d "${out_dir}" ]]; then
        rm -f "${out_dir}"/noise_sweep_matrix_*.csv "${out_dir}"/noise_sweep_combined_L_T.csv || true
      fi

      echo "[TRAIN] ${arch} ${reg} seed=${seed}"
      python -u "${ROOT}/main_train.py" \
        -data fashion_mnist -arch "${arch}" -L 16 --epochs "${FASHION_FC3REV_EPOCHS}" \
        -j "${FC3REV_NUM_WORKERS}" -b "${FC3REV_BATCH}" --seed "${seed}" \
        --device auto --time 0 --spike_schedule normal \
        --regularizer "${regularizer}" --weight_decay "${wd}" --reg_coeff "${rc}" \
        --ckpt-save-mode "${CKPT_SAVE_MODE}" \
        --suffix "${suffix}"

      echo "[TEST] ${arch} ${reg} seed=${seed}"
      python -u "${ROOT}/main_test.py" \
        -data fashion_mnist -arch "${arch}" -L 16 -T 16 \
        -j "${FC3REV_NUM_WORKERS}" -b "${FC3REV_BATCH}" --seed "${seed}" \
        --device auto --mode rate_uniform --spike_schedule normal \
        --weights "${ckpt}" \
        --noise_sweep --noise_sigma_start 0.0 --noise_sigma_end 1.0 --noise_sigma_step 0.05 \
        --noise_output_dir "${out_dir}"
    done
  done
done

echo "[DONE] OUT_ROOT=${OUT_ROOT}"
