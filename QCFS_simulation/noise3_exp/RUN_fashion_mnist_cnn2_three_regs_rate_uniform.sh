#!/usr/bin/env bash
# Fashion-MNIST: CNN2(c2c4/c4c8/c8c16/c16c32) 三路正则 + rate_uniform 噪声扫描
#
# 默认：
# - arch: cnn2_c2_c4 cnn2_c4_c8 cnn2_c8_c16 cnn2_c16_c32
# - seeds: 40 41 42 43 44
# - regs: mne_l2 / weight_decay / no_regularization
# - 训练 T=0, 测试 T=16, sigma=0~1 step=0.05
#
# 通过环境变量覆盖：
#   FASHION_CNN2_ARCHS="cnn2_c2_c4 cnn2_c8_c16"
#   FASHION_CNN2_SEEDS="40 41"
#   FASHION_CNN2_REGS="mne_l2 weight_decay"
#   FASHION_CNN2_EPOCHS=100
#   CNN_BATCH=128
#   CNN_NUM_WORKERS=8
#   CKPT_SAVE_MODE=best|last
#   RETRAIN=1
#   FORCE_TEST=1
#   OUT_ROOT=../important_results/fashion_mnist_cnn2_three_regs/noise_sweep_rate_uniform_L16_T16
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

export MNIST_ROOT="${MNIST_ROOT:-${HOME}/datasets}"
export CNN_BATCH="${CNN_BATCH:-128}"
export CNN_NUM_WORKERS="${CNN_NUM_WORKERS:-8}"
export FASHION_CNN2_EPOCHS="${FASHION_CNN2_EPOCHS:-100}"
export CKPT_SAVE_MODE="${CKPT_SAVE_MODE:-best}"
export RETRAIN="${RETRAIN:-0}"
export FORCE_TEST="${FORCE_TEST:-0}"
export OUT_ROOT="${OUT_ROOT:-../important_results/fashion_mnist_cnn2_three_regs/noise_sweep_rate_uniform_L16_T16}"

ARCHS=(${FASHION_CNN2_ARCHS:-cnn2_c2_c4 cnn2_c4_c8 cnn2_c8_c16 cnn2_c16_c32})
SEEDS=(${FASHION_CNN2_SEEDS:-40 41 42 43 44})
REGS=(${FASHION_CNN2_REGS:-mne_l2 weight_decay no_regularization})

echo "[INFO] ROOT=${ROOT}"
echo "[INFO] MNIST_ROOT=${MNIST_ROOT}"
echo "[INFO] ARCHS=${ARCHS[*]} SEEDS=${SEEDS[*]} REGS=${REGS[*]}"
echo "[INFO] EPOCHS=${FASHION_CNN2_EPOCHS} BATCH=${CNN_BATCH} WORKERS=${CNN_NUM_WORKERS}"
echo "[INFO] CKPT_SAVE_MODE=${CKPT_SAVE_MODE} RETRAIN=${RETRAIN} FORCE_TEST=${FORCE_TEST}"
python -c 'import torch; print("[INFO] torch:", torch.__version__, "cuda:", torch.cuda.is_available())'

for arch in "${ARCHS[@]}"; do
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
        -data fashion_mnist -arch "${arch}" -L 16 --epochs "${FASHION_CNN2_EPOCHS}" \
        -j "${CNN_NUM_WORKERS}" -b "${CNN_BATCH}" --seed "${seed}" \
        --device auto --time 0 --spike_schedule normal \
        --regularizer "${regularizer}" --weight_decay "${wd}" --reg_coeff "${rc}" \
        --ckpt-save-mode "${CKPT_SAVE_MODE}" \
        --suffix "${suffix}"

      echo "[TEST] ${arch} ${reg} seed=${seed}"
      python -u "${ROOT}/main_test.py" \
        -data fashion_mnist -arch "${arch}" -L 16 -T 16 \
        -j "${CNN_NUM_WORKERS}" -b "${CNN_BATCH}" --seed "${seed}" \
        --device auto --mode rate_uniform --spike_schedule normal \
        --weights "${ckpt}" \
        --noise_sweep --noise_sigma_start 0.0 --noise_sigma_end 1.0 --noise_sigma_step 0.05 \
        --noise_output_dir "${out_dir}"
    done
  done
done

echo "[DONE] OUT_ROOT=${OUT_ROOT}"
