# Fashion-MNIST deep Conv-BN-IF L2-scope experiment

## Protocol

- Training: ANN only (`T=0`), 30 epochs, seed 42, Fashion-MNIST.
- Conversion/evaluation: SNN with `T=16`, `L=16`, IF mode `rate_uniform`.
- Perturbation: fixed absolute Gaussian noise after the input IF and before `conv1` (`post_input_if`). This is not raw-image noise.
- Noise levels: sigma = 0, 0.25, 0.5, 0.75, 1.0.
- For every nonzero sigma, the reported standard deviation is over three paired noise draws. It is not uncertainty over independently trained models.
- `weights_only`: optimizer weight decay `5e-4` only on Conv/Linear weights.
- `all_params`: explicit L2 coefficient `2.5e-4` on every trainable parameter. Since the derivative is `2 * rc * p`, its matched shrinkage is `5e-4 * p`.

The two implementations use matched L2 strength; the intended factor is parameter scope. One-step SGD checks gave zero state difference between each explicit-L2 and optimizer-weight-decay counterpart when their scopes matched.

## Architectures

- CNN2: channels 2-4.
- CNN4: channels 2-4-4-4.
- CNN6: channels 2-4-4-4-4-4.
- CNN8: channels 2-4-4-4-4-4-4-4.
- CNN10: channels 2-4-4-4-4-4-4-4-4-4.
- Every convolution is followed by BN and IF. Pooling remains after conv1 and conv2, so added blocks operate on the 7x7 feature map.

## SNN accuracy (%)

| Model | L2 scope | sigma=0 | sigma=0.25 | sigma=0.5 | sigma=0.75 | sigma=1 | Drop to sigma=1 | Curve AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CNN2 | weights only | 82.19 | 79.97 +/- 0.30 | 66.45 +/- 0.77 | 53.62 +/- 0.23 | 42.02 +/- 0.19 | 40.17 | 65.54 |
| CNN2 | all params | 81.71 | 75.75 +/- 0.31 | 64.86 +/- 0.26 | 52.08 +/- 0.33 | 40.76 +/- 0.26 | 40.95 | 63.48 |
| CNN4 | weights only | 80.02 | 74.81 +/- 0.31 | 60.60 +/- 0.08 | 53.85 +/- 0.10 | 45.98 +/- 0.30 | 34.04 | 63.07 |
| CNN4 | all params | 78.54 | 63.84 +/- 0.40 | 55.70 +/- 0.14 | 47.40 +/- 0.17 | 41.49 +/- 0.39 | 37.05 | 56.74 |
| CNN6 | weights only | 79.23 | 69.68 +/- 0.34 | 54.04 +/- 0.30 | 44.93 +/- 0.07 | 37.16 +/- 0.30 | 42.07 | 56.71 |
| CNN6 | all params | 80.05 | 62.01 +/- 0.21 | 54.35 +/- 0.08 | 42.66 +/- 0.48 | 36.58 +/- 0.24 | 43.47 | 54.33 |
| CNN8 | weights only | 79.57 | 76.12 +/- 0.18 | 67.27 +/- 0.18 | 59.74 +/- 0.45 | 51.42 +/- 0.40 | 28.15 | 67.16 |
| CNN8 | all params | 78.54 | 73.69 +/- 0.34 | 66.01 +/- 0.42 | 57.87 +/- 0.21 | 50.61 +/- 0.02 | 27.93 | 65.54 |
| CNN10 | weights only | 79.72 | 72.54 +/- 0.17 | 57.73 +/- 0.27 | 48.42 +/- 0.09 | 40.26 +/- 0.13 | 39.46 | 59.67 |
| CNN10 | all params | 80.31 | 73.40 +/- 0.46 | 57.63 +/- 0.08 | 46.80 +/- 0.13 | 38.99 +/- 0.32 | 41.32 | 59.37 |

The best ANN validation accuracies were 84.16/84.30 for CNN4, 84.33/84.58 for CNN6, 83.68/83.68 for CNN8, and 83.24/83.73 for CNN10 (weights-only/all-params respectively), so the deep comparisons are not explained by failed ANN training.

## Findings

1. Weights-only L2 has higher noise-curve AUC at every tested depth, but its advantage is non-monotonic: 2.06, 6.33, 2.38, 1.62, and 0.30 points for CNN2/4/6/8/10.
2. At sigma=0.25, the absolute accuracy deltas (weights-only minus all-params) are 4.22, 10.97, 7.67, 2.43, and -0.86 points. CNN10 is therefore a counterexample to a universal weights-only advantage at moderate noise.
3. At sigma=1, the deltas remain positive at all depths: 1.26, 4.50, 0.59, 0.81, and 1.27 points. CNN4 still shows the largest separation; depth alone does not reproduce the very large CIFAR/VGG weight-decay collapse.
4. All-parameter L2 consistently lowers IF thresholds (hidden-layer mean about 5.0 versus 7.3) and BN gamma. Its BN-folded, threshold-normalized geometric-mean layer gain is about 1.38x, 1.41x, 1.42x, 1.47x, and 1.47x larger for CNN2/4/6/8/10. The product proxy ratio grows from 1.90x to 3.91x, 8.17x, 21.60x, and 46.88x. This supports cumulative internal rescaling, but the accuracy delta is governed by nonlinear IF/pooling dynamics and is not monotonic.
5. The delta plot defines `delta = accuracy(weights-only) - accuracy(all-params)`. The accompanying CSV also reports a clean-adjusted delta that subtracts each model's sigma=0 gap.

## Interpretation boundary

This single-training-seed result supports the narrower claim that decaying BN/IF parameters changes internal scale and tends to reduce post-input-IF noise-curve AUC. It does not establish a universal pointwise advantage, a monotonic depth law, or a full explanation of the CIFAR/VGG failure. A confirmatory run should use at least three training seeds and report both absolute and clean-adjusted deltas.
