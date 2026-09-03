#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULT_ROOT="$(cd "${SIM_ROOT}/.." && pwd)/important_results/fashion_l2_scope_followup"

cd "${SIM_ROOT}"

"${PYTHON_BIN}" noise3_exp/run_fashion_spectral_mne_ablation.py \
  --dataset fashion_mnist \
  --models cnn6_vgg \
  --methods wd_weight_only manual_l2_all \
  --seeds 40 41 42 43 44 \
  --epochs 30 --L 16 --train-T 0 --test-T 16 \
  --lr 0.01 --batch-size 128 --workers 2 --device mps \
  --weight-decay 5e-4 --matched-l2-rc 2.5e-4 \
  --ckpt-save-mode best --train-only \
  --out-root "${RESULT_ROOT}/training_registry"

"${PYTHON_BIN}" noise3_exp/run_fashion_spectral_mne_ablation.py \
  --dataset fashion_mnist \
  --models cnn6_vgg \
  --methods manual_l2_w_bn manual_l2_w_if manual_l2_w_bn_if \
  --seeds 42 \
  --epochs 30 --L 16 --train-T 0 --test-T 16 \
  --lr 0.01 --batch-size 128 --workers 2 --device mps \
  --weight-decay 5e-4 --matched-l2-rc 2.5e-4 \
  --ckpt-save-mode best --train-only \
  --out-root "${RESULT_ROOT}/training_registry"

"${PYTHON_BIN}" noise3_exp/run_fashion_spectral_mne_ablation.py \
  --dataset fashion_mnist \
  --models cnn6_narrow_staged cnn6_wide_early \
  --methods wd_weight_only manual_l2_all \
  --seeds 42 \
  --epochs 30 --L 16 --train-T 0 --test-T 16 \
  --lr 0.01 --batch-size 128 --workers 2 --device mps \
  --weight-decay 5e-4 --matched-l2-rc 2.5e-4 \
  --ckpt-save-mode best --train-only \
  --out-root "${RESULT_ROOT}/training_registry"

"${PYTHON_BIN}" noise3_exp/run_fashion_internal_if_noise_screen.py \
  --dataset fashion_mnist --model cnn6_vgg \
  --methods wd_weight_only manual_l2_all \
  --sites post_input_if --sigmas 0 0.25 0.5 0.75 1 \
  --seeds 40 41 42 43 44 --noise-repeats 3 \
  --epochs 30 --L 16 --T 16 --if-mode rate_uniform \
  --spike-schedule normal --batch-size 128 --workers 2 --device mps \
  --out-dir "${RESULT_ROOT}/multiseed_absolute"

"${PYTHON_BIN}" noise3_exp/run_fashion_internal_if_noise_screen.py \
  --dataset fashion_mnist --model cnn6_vgg \
  --methods wd_weight_only manual_l2_w_bn manual_l2_w_if manual_l2_w_bn_if manual_l2_all \
  --sites post_input_if --sigmas 0 0.25 0.5 0.75 1 \
  --seeds 42 --noise-repeats 5 \
  --epochs 30 --L 16 --T 16 --if-mode rate_uniform \
  --spike-schedule normal --batch-size 128 --workers 2 --device mps \
  --out-dir "${RESULT_ROOT}/parameter_scope_absolute"

for MODEL in cnn6_narrow_staged cnn6_wide_early; do
  "${PYTHON_BIN}" noise3_exp/run_fashion_internal_if_noise_screen.py \
    --dataset fashion_mnist --model "${MODEL}" \
    --methods wd_weight_only manual_l2_all \
    --sites post_input_if --sigmas 0 0.25 0.5 0.75 1 \
    --seeds 42 --noise-repeats 3 \
    --epochs 30 --L 16 --T 16 --if-mode rate_uniform \
    --spike-schedule normal --batch-size 128 --workers 2 --device mps \
    --out-dir "${RESULT_ROOT}/architecture_2x2/${MODEL}_post_input_if"
done

"${PYTHON_BIN}" noise3_exp/run_fashion_internal_if_noise_screen.py \
  --dataset fashion_mnist --model cnn6_vgg \
  --methods wd_weight_only manual_l2_all \
  --sites post_input_if --sigmas 0 0.05 0.10 0.15 0.20 \
  --sigma-scale input_if_threshold \
  --seeds 42 --noise-repeats 5 \
  --epochs 30 --L 16 --T 16 --if-mode rate_uniform \
  --spike-schedule normal --batch-size 128 --workers 2 --device mps \
  --out-dir "${RESULT_ROOT}/relative_threshold"

"${PYTHON_BIN}" noise3_exp/run_fashion_internal_if_noise_screen.py \
  --dataset fashion_mnist --model cnn6_vgg \
  --methods wd_weight_only manual_l2_all \
  --sites post_input_if --sigmas 0 0.25 0.5 0.75 1 \
  --sigma-scale post_input_if_rms --rms-calibration-batches 16 \
  --seeds 42 --noise-repeats 5 \
  --epochs 30 --L 16 --T 16 --if-mode rate_uniform \
  --spike-schedule normal --batch-size 128 --workers 2 --device mps \
  --out-dir "${RESULT_ROOT}/relative_rms"

"${PYTHON_BIN}" noise3_exp/analyze_fashion_l2_scope_mechanism.py \
  --model cnn6_vgg \
  --methods wd_weight_only manual_l2_w_bn manual_l2_w_if manual_l2_w_bn_if manual_l2_all \
  --seed 42 --sigmas 0.5 1 --noise-repeats 5 \
  --max-samples 2048 --epochs 30 --L 16 --T 16 \
  --batch-size 64 --workers 2 --device mps \
  --out-dir "${RESULT_ROOT}"

"${PYTHON_BIN}" noise3_exp/summarize_fashion_l2_scope_followup.py \
  --root "${RESULT_ROOT}"

echo "[DONE] Fashion L2-scope follow-up: ${RESULT_ROOT}"
