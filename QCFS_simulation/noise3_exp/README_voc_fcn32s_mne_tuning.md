# FCN-32s: tune actual regularization strength first

This is a new experiment, not a change to the original five-regularizer run.
It does not assume that MNE will beat L2. No full VOC training has been run
locally. CPU tests use tiny synthetic fixtures, not publishable results.

## What changes

Keep the original no-BN FCN-32s, ImageNet initialization, QCFS L=16,
ANN training, T=16 rate-uniform inference, and post-input-IF Gaussian noise.
Keep the existing percentile initialization, max pooling and linear output
head unchanged. Train without noise in this first screen.

The current no-BN MNE implementation is:

```text
R = sum_l L^2 ||W_l||_F^2 / (C_out,l (lambda_l^2 + eps))
grad_W (beta R)_l = beta k_l W_l
k_l = 2 L^2 / (C_out,l (lambda_l^2 + eps))
beta_match = wd_ref sqrt(sum_l ||W_l||^2) /
                     sqrt(sum_l k_l^2 ||W_l||^2)
wd_ref = 5e-4
```

The old `rc=1e-4` and `weight_decay=5e-4` do not imply comparable shrinkage.
Wide layers and large IF thresholds can make the MNE weight gradients much
smaller. This is a plausible tuning issue, not an established explanation
of the task-transfer result.

Compute `beta_match` once, after IF initialization, from the 15 matched
Conv-IF weights. Freeze beta throughout training. The formula and layer
allocation of MNE are unchanged; no per-step budget normalization is added.
The no-detach variant also receives lambda gradients, which are deliberately
excluded from the weight-gradient norm used for matching.

| Method | Three strength settings | Linear head weight decay |
| --- | --- | --- |
| L2-wo | body WD = 5e-5, 5e-4, 5e-3 | 5e-4 |
| MNE detach | beta = beta_match x (0.1, 1, 10) | 5e-4 |
| MNE no-detach | beta = beta_match x (0.1, 1, 10) | 5e-4 |

Biases and thresholds receive no optimizer weight decay. Ordinary task-loss
gradients into thresholds remain enabled in every method. Only the MNE
regularization gradient into thresholds is detached for `mne`.

## Data and selection

Prepare a fixed, disjoint 200-image tuning subset from `train` (default) or
an explicitly requested `trainaug`. The manifest rejects train/official-val
overlap. Threshold calibration and training use only the remaining fit set.
No automatic switch to trainaug occurs. Reuse the same choice as the current
experiment when isolating regularization; changing the dataset is a separate
experiment and needs a new output root.

All nine runs use seed 42 and the same 50-epoch schedule. Select one strength
per method using trapezoid-averaged SNN mIoU over sigma=[0,0.5,1,2,3,5],
subject to clean SNN mIoU being no more than 1 percentage point below the best
clean L2 candidate. This criterion is fixed before inspecting the new results.
A method with no eligible candidate is marked unselected, not called a win.

Confirmation retrains from the same initialization recipe on the full chosen
training split, then evaluates the official VOC2012 validation masks. Freeze
the strength multiplier and matching rule; beta itself is initialized per run.
The official validation set has already been inspected in earlier experiments,
so it is not a never-seen test set. Treat these results as development evidence;
reserve a further independent evaluation for strong generalization claims.

## Gadi commands

Use the same project, modules, dataset and cached VGG16 weights as the earlier
jobs. Check the PBS resource requests against your allocation. These commands
submit GPU jobs; none have been submitted by Codex.

```bash
cd /home/595/sl9144/codes/snn_simulation
export VOC_ROOT=/home/595/sl9144/datasets
export OUT_DIR=/scratch/gs14/sl9144/snn_results/voc_fcn32s_mne_tuning_v1
module use /g/data/dk92/apps/Modules/modulefiles
module load NCI-ai-ml/24.11
SCRIPT=QCFS_simulation/noise3_exp/run_voc_fcn32s_mne_tuning.py
PBS=QCFS_simulation/noise3_exp/pbs_voc_fcn32s_mne_tuning.pbs

# Lightweight split preparation only; no training on the login node.
python "$SCRIPT" prepare --voc-root "$VOC_ROOT" --out-root "$OUT_DIR" --train-split train

# One small GPU smoke job first. Inspect its log before the full screen.
qsub -v VOC_ROOT,OUT_DIR,METHOD=mne,STRENGTH=1,SMOKE=1 "$PBS"

# Nine full GPU jobs after the smoke check succeeds.
for METHOD in l2wo mne nodetach; do
  for STRENGTH in 0.1 1 10; do
    qsub -N "tune_${METHOD}_${STRENGTH}" \
      -v VOC_ROOT,OUT_DIR,METHOD="$METHOD",STRENGTH="$STRENGTH",SEED=42 "$PBS"
  done
done

# After all nine runs finish: light aggregation, no GPU required.
python "$SCRIPT" select --out-root "$OUT_DIR" --seed 42 --clean-tolerance 1

# Inspect selection.json first; omit methods marked null.
# Start with seed 42; add 43 and 44 only if the paired comparison is promising.
for METHOD in l2wo mne nodetach; do
  qsub -N "confirm_${METHOD}" \
    -v VOC_ROOT,OUT_DIR,METHOD="$METHOD",STAGE=confirm,SEED=42 "$PBS"
done
```

Rerunning a completed command skips it. Interrupted training resumes from the
last atomic epoch checkpoint, including optimizer and scheduler. Do not submit
duplicate jobs for the same output directory at the same time. Config/source
changes require a new output root; existing configurations are not overwritten.
With `--smoke`, outputs go to `smoke/` and cannot enter parameter selection.

## Outputs and interpretation

- `split.json`: fixed fit/tuning/official-validation IDs.
- `initial_strength.json`: matched beta, per-layer thresholds and old-MNE/L2
  weight-gradient ratios. These are initialization measurements, not full
  training averages.
- `epoch_log.csv`, `final_strength.json`: task loss, weighted penalty, changing
  gradient norms and thresholds. Equal initial gradient norms need not remain
  equal during training.
- `result.json`: ANN clean mIoU, SNN clean/noisy mIoU, per-class IoU and counts.
- `tuning_summary.csv`, `selection.json`: all candidates and locked choices.
- `logs/`: live PBS logs; `last.pth`: resumable checkpoint per candidate.

Compare absolute noisy mIoU/AUC, clean mIoU and the ANN-to-SNN clean gap together.
The reused IF-density diagnostic is an unweighted average over layer/image
observations, not device energy. Do not infer energy savings from this field.

If stronger, tuned MNE wins in confirmation, extend to three paired seeds before
claiming a benefit. If it does not, the initial coefficient alone is not enough.
Only then change architecture or training: conversion-consistent pooling and
noise-aware fine-tuning are separate candidates, and must also be offered to L2.
Adding BN or an IF after signed output logits is not part of this experiment.

## Local checks

```bash
/opt/anaconda3/envs/snn/bin/python -m unittest discover \
  -s QCFS_simulation/noise3_exp -p test_voc_fcn32s_mne_tuning.py -v
bash -n QCFS_simulation/noise3_exp/pbs_voc_fcn32s_mne_tuning.pbs
```
