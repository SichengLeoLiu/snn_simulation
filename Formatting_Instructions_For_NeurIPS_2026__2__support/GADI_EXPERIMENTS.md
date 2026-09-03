# Gadi experiments that most improve the ODI paper

Do not treat noise repeats as training seeds. The target hierarchy is:

1. Extend the existing seed-42 SNR sweep to negative SNRs.
2. Repeat the six-configuration SNR sweep for independent training seeds.
3. Only after those are complete, consider another dataset or hardware timing.

The commands below intentionally do not guess the Gadi project, queue, storage path,
module stack, or Python environment. Set these after authentication and environment
setup:

```bash
export REPO_ROOT=/path/to/QCFS_simulation_FI
export PYTHON_BIN=/path/to/the/project/python
```

## A. Negative-SNR sweep with existing seed-42 checkpoints

This is the highest-value evaluation-only run. It tests whether the six curves remain
close at the severity actually experienced by the low-scale checkpoints.

```bash
cd "$REPO_ROOT/QCFS_simulation"

"$PYTHON_BIN" noise3_exp/run_cifar_vgg16_relative_snr_noise_sweep.py \
  --dataset cifar10 \
  --arch vgg16 \
  --L 16 \
  --T 16 \
  --mode rate_uniform \
  --seed 42 \
  --noise-seed 20260809 \
  --noise-position post_input_if \
  --scale-modes snr \
  --snr-db 30 20 15 10 5 0 -5 -10 -15 -20 -25 \
  --noise-repeats 5 \
  --out-dir ../important_results/odi_snr_extended_seed42
```

Required output:

```text
important_results/odi_snr_extended_seed42/relative_snr_noise_raw.csv
important_results/odi_snr_extended_seed42/relative_snr_noise_mean_std.csv
important_results/odi_snr_extended_seed42/relative_snr_noise_summary.csv
```

## B. Train the missing seeds for all six configurations

Run this as a PBS array with indices 0--4. Each array element trains one seed and
therefore keeps failure recovery simple. Existing matching checkpoints are skipped.
The zero-only absolute evaluation is used here because the full relative sweep is the
next stage.

```bash
set -euo pipefail
: "${PBS_ARRAY_INDEX:?Submit this job as a PBS array with indices 0-4}"
: "${REPO_ROOT:?Set REPO_ROOT first}"
: "${PYTHON_BIN:?Set PYTHON_BIN first}"

SEEDS=(40 41 42 43 44)
SEED="${SEEDS[$PBS_ARRAY_INDEX]}"
cd "$REPO_ROOT/QCFS_simulation"

"$PYTHON_BIN" noise3_exp/run_mne_stability_ablation.py \
  --datasets cifar10 \
  --arch vgg16 \
  --L 16 \
  --train-T 0 \
  --test-T 16 \
  --epochs 300 \
  --seeds "$SEED" \
  --variants old_detach mne_l2_all l1_all \
  --rcs 1e-4 \
  --l1-all-rcs 1e-5 \
  --baselines weight_decay_weights_only manual_l2_all manual_l2_w_bn_gamma \
  --noise-sigma-start 0 \
  --noise-sigma-end 0 \
  --noise-sigma-step 1 \
  --out-root ../important_results/odi_six_config_training
```

Before submitting the full array, run the same command once with `--dry-run` and one
seed to verify checkpoint names and the active Python environment.

## C. Relative-SNR evaluation for every training seed

Run this as a second PBS array with indices 0--4 after stage B succeeds. Three noise
repeats are the minimum useful setting across five training seeds; set
`NOISE_REPEATS=5` if the allocation permits.

```bash
set -euo pipefail
: "${PBS_ARRAY_INDEX:?Submit this job as a PBS array with indices 0-4}"
: "${REPO_ROOT:?Set REPO_ROOT first}"
: "${PYTHON_BIN:?Set PYTHON_BIN first}"

SEEDS=(40 41 42 43 44)
SEED="${SEEDS[$PBS_ARRAY_INDEX]}"
NOISE_REPEATS="${NOISE_REPEATS:-3}"
cd "$REPO_ROOT/QCFS_simulation"

"$PYTHON_BIN" noise3_exp/run_cifar_vgg16_relative_snr_noise_sweep.py \
  --dataset cifar10 \
  --arch vgg16 \
  --L 16 \
  --T 16 \
  --mode rate_uniform \
  --seed "$SEED" \
  --noise-seed 20260809 \
  --noise-position post_input_if \
  --scale-modes snr \
  --snr-db 30 20 15 10 5 0 -5 -10 -15 -20 -25 \
  --noise-repeats "$NOISE_REPEATS" \
  --methods MNE-standard Weights-only L1-all MNE-all L2-all Weights+BN-gamma \
  --ckpt MNE-standard "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_old_detach_rc0p0001_seed${SEED}_L16_trainT0.pth" \
  --ckpt Weights-only "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_weight_decay_weights_only_rcnone_seed${SEED}_L16_trainT0.pth" \
  --ckpt L1-all "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_l1_all_rc1em05_seed${SEED}_L16_trainT0.pth" \
  --ckpt MNE-all "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_mne_l2_all_rc0p0001_seed${SEED}_L16_trainT0.pth" \
  --ckpt L2-all "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_manual_l2_all_rc0p00025_seed${SEED}_L16_trainT0.pth" \
  --ckpt Weights+BN-gamma "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_manual_l2_w_bn_gamma_rc0p00025_seed${SEED}_L16_trainT0.pth" \
  --out-dir "../important_results/odi_snr_multiseed/seed_${SEED}" \
  --no-plot
```

After syncing the five seed directories back to this workspace, aggregate them from
the repository root:

```bash
python3 Formatting_Instructions_For_NeurIPS_2026__2__support/experiments/aggregate_gadi_snr_sweeps.py \
  --root important_results/odi_snr_multiseed
```

Acceptance checks:

- `n_training_seeds` is 5 for every method and SNR level.
- Clean accuracy matches the corresponding checkpoint evaluation.
- Error bars are computed across per-training-seed means, not across raw noise draws.
- Missing or failed array elements are rerun before updating the paper.

## D. MNE-L2 component ablation

### D.1 Scientific question and configurations

Use CIFAR-10 VGG16 first. Keep the architecture, data split, augmentation, optimizer,
training epochs, `L=T=16`, noise site, seeds, and checkpoint rule fixed. Compare:

| Key | Objective component retained | Question |
|---|---|---|
| `weight_decay_weights_only` | raw weights only | Standard weights-only L2 reference |
| `effective_l2` | BN folding only | Is folded amplification useful without margin normalization? |
| `threshold_l2` | threshold normalization only | Is the margin term useful without BN folding? |
| `old_detach` | BN folding + margin + stop-gradients | Full MNE-L2 |
| `folded_w_lambda` | BN folding + margin; trainable threshold | Is threshold stop-gradient necessary? |
| `mne_bn_trainable` | BN folding + margin; trainable BN gain | Is BN-affine stop-gradient necessary? |

Do not compare these objectives using a common numerical coefficient: their reductions
and scales differ. Select one coefficient per configuration using clean validation only.
The preferred protocol is a fixed 45k/5k split of the CIFAR training set with split seed
`20260820`; keep the standard test set untouched until the final sweep. All configurations,
including weights-only L2, must be retrained with this same split.

Implementation prerequisite for Cursor: add optional
`--val-fraction 0.1 --split-seed 20260820` arguments to the CIFAR training loader and
use validation accuracy, not test accuracy, for `--ckpt-save-mode best`. Record train,
validation, and test indices. The current runner can smoke-test the objectives, but its
standard-test output must not be used for coefficient selection before this split exists.

### D.2 Smoke test

```bash
cd "$REPO_ROOT/QCFS_simulation"

"$PYTHON_BIN" noise3_exp/run_mne_stability_ablation.py \
  --datasets cifar10 \
  --arch vgg16 \
  --L 16 \
  --train-T 0 \
  --test-T 16 \
  --epochs 300 \
  --seeds 42 \
  --variants effective_l2 threshold_l2 old_detach folded_w_lambda mne_bn_trainable \
  --rcs 1e-4 \
  --baselines weight_decay_weights_only \
  --noise-sigma-start 0 \
  --noise-sigma-end 0 \
  --noise-sigma-step 1 \
  --out-root ../important_results/odi_mne_components_smoke \
  --dry-run
```

Run once without `--dry-run` only after confirming checkpoint paths and the validation
split. Require finite losses, nonzero explicit-regularizer gradients, and clean accuracy
above chance for every configuration.

### D.3 Coefficient pilot and selection

Use seed 42 only for the pilot. First train full MNE-L2 at its prespecified coefficient
`1e-4` and freeze its clean validation accuracy as the matching target. Run each remaining
variant separately so it can have its own grid. A practical initial grid is
`1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3`; extend one decade only when every point is under-
or over-regularized. For each variant, select the coefficient with clean validation accuracy
closest to the frozen MNE-L2 target, requiring a difference of at most 0.5 percentage points;
break ties by choosing the smaller coefficient. Do not use noisy validation or test accuracy
for selection. Also retain the validation-best coefficient as a secondary, non-matched
operating point if compute permits.

Example for one component:

```bash
"$PYTHON_BIN" noise3_exp/run_mne_stability_ablation.py \
  --datasets cifar10 --arch vgg16 --L 16 --train-T 0 --test-T 16 \
  --epochs 300 --seeds 42 \
  --variants effective_l2 \
  --rcs 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 \
  --baselines \
  --noise-sigma-start 0 --noise-sigma-end 0 --noise-sigma-step 1 \
  --out-root ../important_results/odi_mne_components_pilot
```

Repeat for `threshold_l2`, `folded_w_lambda`, and `mne_bn_trainable`; `old_detach` at
`1e-4` defines the target. Weights-only L2 uses its own weight-decay grid. Save the selection
table with coefficient, best epoch, clean validation accuracy, distance from the frozen target,
and clean test accuracy (the last field is populated only after selection is frozen).

### D.4 Five-seed final experiment

Freeze the selected coefficient for every configuration, retrain seeds 40--44, and evaluate
`sigma=0:0.25:5` after the input IF. Run one variant per PBS array to avoid mixing its
coefficient with another variant's value:

```bash
"$PYTHON_BIN" noise3_exp/run_mne_stability_ablation.py \
  --datasets cifar10 --arch vgg16 --L 16 --train-T 0 --test-T 16 \
  --epochs 300 --seeds 40 41 42 43 44 \
  --variants old_detach \
  --rcs SELECTED_MNE_COEFFICIENT \
  --baselines \
  --noise-sigma-start 0 --noise-sigma-end 5 --noise-sigma-step 0.25 \
  --out-root ../important_results/odi_mne_components_final
```

Report clean top-1, accuracy at `sigma=1,3,5`, relative retention, and normalized AUC
`integral(A(sigma)/A(0)) / 5`, all as mean plus sample standard deviation over training
seeds. Add layerwise `rho` and crossing probability at `sigma=1` for seed 42 as mechanism
diagnostics, not as multi-seed evidence. The clean-matched comparison and normalized AUC
are the primary acceptance criteria for the ablation.

## E. ODI-oriented efficiency evidence

### E.1 Event activity and arithmetic energy (already supported)

This evaluation reuses checkpoints and does not retrain. Run the full test set and preserve
both the raw per-seed CSV and five-seed summary:

```bash
cd "$REPO_ROOT/QCFS_simulation"

"$PYTHON_BIN" noise3_exp/measure_vgg16_horowitz_energy.py \
  --datasets cifar10 cifar100 \
  --methods l2 l2_wo l1 mne_l2 \
  --seeds 40 41 42 43 44 \
  --time-steps 4 8 16 \
  --batch-size 128 \
  --workers 8 \
  --max-samples 0 \
  --out-dir ../important_results/cifar_vgg16_three_regs_horowitz_energy
```

Check `accuracy`, `if_firing_density`, `event_synops_per_sample`, estimated SNN energy,
and ANN/SNN ratio for every row. Arithmetic energy must continue to be labelled an estimate;
it excludes memory, neuron updates, control, interconnect, and leakage.

### E.2 Software latency and peak memory on Gadi

This is useful to ODI only when described as PyTorch simulation performance, not sparse
neuromorphic-hardware performance. Implement a test-only benchmark with batch size 1,
50 warm-up iterations, 200 timed iterations, `torch.cuda.synchronize()` around timing,
and `torch.cuda.reset_peak_memory_stats()` before measurement. Fix GPU model, CUDA/PyTorch
versions, power mode, precision, and dataloader workers. Measure L2-all and MNE-L2 at
`T=4,8,16` on seed 42 first; expand to five seeds only if between-seed variance is material.

Report median and p95 latency, throughput, peak allocated GPU memory, and clean accuracy.
Run the ANN source model with the same input pipeline and precision. Do not interpret a dense
PyTorch SNN slowdown as evidence against event-driven hardware; it measures this simulator.

### E.3 Device measurement when hardware is available

The strongest ODI evidence is batch-1 latency and energy on an actual target such as Jetson
Orin or a neuromorphic platform. Use at least 1000 inferences after warm-up, subtract measured
idle power, integrate power over the inference window, and report device, clock/power mode,
software stack, precision, and whether event sparsity is exploited by the runtime. Compare
accuracy, latency, energy per sample, and peak memory jointly at `T=4` and `T=16`.
