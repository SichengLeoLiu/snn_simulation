# MNE-L2 current-code, three-seed internal-noise check

## Protocol

- Dataset/model: Fashion-MNIST, CNN2 c2c4
- Training: ANN (`T=0`), 30 epochs, best checkpoint, seeds 40/41/42
- Conversion/evaluation: SNN `T=16`, `L=16`, `rate_uniform`
- Noise: fixed absolute Gaussian activation noise, sigma in {0, 0.25, 0.5, 1}
- Noise sites are reported separately: `post_if1`, `post_input_if`, and `both`
- Each nonzero sigma uses three noise draws per trained seed. Noise draws are averaged
  within seed before the mean and sample standard deviation are computed across seeds.
- Regularizers: MNE-L2 rc=1e-4 with detached IF threshold, L1-all rc=1e-5,
  weight-only weight decay=5e-4, and No Reg.

## Primary result: post_if1 accuracy (%)

| Method | sigma=0 | sigma=0.25 | sigma=0.5 | sigma=1 | Curve AUC |
|---|---:|---:|---:|---:|---:|
| MNE-L2 | 82.770 +/- 0.726 | 80.244 +/- 1.767 | 72.063 +/- 4.813 | 47.423 +/- 17.233 | 69.287 +/- 6.557 |
| L1-all | 82.497 +/- 1.134 | 79.503 +/- 1.325 | 70.829 +/- 5.266 | 46.740 +/- 18.865 | 68.434 +/- 7.040 |
| WD, weights only | 82.890 +/- 0.648 | 79.719 +/- 1.904 | 70.279 +/- 6.963 | 43.356 +/- 17.933 | 67.484 +/- 7.085 |
| No Reg | 82.767 +/- 0.625 | 79.357 +/- 1.377 | 70.127 +/- 4.707 | 44.828 +/- 17.948 | 67.689 +/- 6.637 |

MNE-L2 has the best three-seed mean at every nonzero sigma while preserving clean
accuracy. Its paired AUC differences against No Reg are +1.139, +1.480, and +2.174
points for seeds 40, 41, and 42 (mean +1.597 +/- 0.527). The corresponding paired
differences against L1-all are -0.434, +0.355, and +2.639 (mean +0.853 +/- 1.596),
so the evidence against L1 is weaker and MNE does not win in all three seeds.

## Location check: curve AUC (%)

| Method | post_if1 | post_input_if | both |
|---|---:|---:|---:|
| MNE-L2 | 69.287 +/- 6.557 | 69.897 +/- 6.825 | 53.638 +/- 12.258 |
| L1-all | 68.434 +/- 7.040 | 72.817 +/- 2.660 | 56.187 +/- 9.117 |
| WD, weights only | 67.484 +/- 7.085 | 70.501 +/- 4.207 | 52.326 +/- 9.759 |
| No Reg | 67.689 +/- 6.637 | 71.346 +/- 4.976 | 54.074 +/- 9.093 |

The MNE advantage is localized to noise immediately after IF1. It does not extend
to the first input IF or to the complete two-site curve. At `both`, MNE has a higher
sigma=1 mean than L1 and No Reg, but a lower curve AUC because it is substantially
worse at sigma=0.25.

## Mechanism diagnostics

Across the 12 checkpoints, the following are descriptive correlations with the
`post_if1` curve AUC:

| Diagnostic | Pearson r |
|---|---:|
| Conv2 BN-folded RMS filter norm / IF2 threshold | -0.523 |
| IF1 clean output RMS | +0.701 |
| Conv2 gain / IF1 output RMS | -0.719 |
| IF2 threshold alone | +0.144 |

The average Conv2 BN-folded RMS gain is 0.2711 +/- 0.0470 for MNE-L2, 0.2777 +/-
0.0432 for L1-all, 0.3288 +/- 0.0669 for weight-only WD, and 0.2852 +/- 0.0353 for
No Reg. Clean SNN logit-margin correlations are weak (absolute r <= 0.34).

These diagnostics support the proposed effective-gain mechanism more strongly than
a threshold-only explanation. They are not causal estimates: the 12 checkpoints
are clustered within four methods and only three training seeds are available.

## Supported claim and limits

Supported: in this current-code Fashion-MNIST c2c4 experiment, MNE-L2 provides a
small, seed-consistent AUC gain over No Reg for fixed absolute Gaussian noise after
IF1, without sacrificing clean SNN accuracy. The direction is consistent with MNE's
control of BN-folded gain relative to the next IF threshold.

Not supported: MNE-L2 is globally better than L1, is robust to every internal-noise
location, or has a statistically established architecture-independent advantage.
The large between-seed standard deviation at sigma=1 is a major unresolved issue.
