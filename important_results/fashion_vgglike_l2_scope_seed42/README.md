# Fashion-MNIST VGG-like CNN6 L2-scope experiment

## Question

Does a staged, widening architecture expose a larger robustness difference between matched weights-only and all-parameter L2 than the narrow CNN6?

## Protocol

- VGG-like CNN6: `c8-c8-pool-c16-c16-pool-c32-c32`.
- Every convolution is followed by BN and IF; an input IF precedes conv1.
- ANN training: `T=0`, 30 epochs, seed 42, learning rate 0.01.
- SNN evaluation: `T=16`, `L=16`, IF mode `rate_uniform`.
- Perturbation: fixed absolute Gaussian noise after the input IF and before conv1 (`post_input_if`). This is not raw-image noise.
- Noise levels: 0, 0.25, 0.5, 0.75, and 1.0.
- Every nonzero sigma uses three paired noise draws. Their standard deviation is evaluation randomness, not training-seed uncertainty.
- Weights-only: optimizer weight decay `5e-4` on Conv/Linear weights only.
- All-parameters: explicit L2 coefficient `2.5e-4` on every trainable parameter, giving the matched gradient coefficient `5e-4`.

The narrow CNN6 and VGG-like CNN6 contain the same number of Conv-BN-IF layers. Their main differences are channels, pooling positions, and parameter count: about 2,709 versus 33,961 parameters.

## VGG-like CNN6 results (%)

| Scope | sigma=0 | sigma=0.25 | sigma=0.5 | sigma=0.75 | sigma=1 | Drop | Curve AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| weights only | 87.49 | 83.19 +/- 0.31 | 74.04 +/- 0.12 | 60.81 +/- 0.44 | 48.52 +/- 0.06 | 38.97 | 71.51 |
| all params | 88.29 | 79.38 +/- 0.24 | 62.94 +/- 0.16 | 48.12 +/- 0.24 | 37.03 +/- 0.24 | 51.26 | 63.28 |

The best ANN validation accuracies were 89.62% for weights-only and 90.13% for all-parameters. All-parameters therefore started higher in both ANN and clean SNN accuracy, but became substantially worse under noise.

## Same-depth architecture comparison

Define `delta = accuracy(weights-only) - accuracy(all-parameters)`.

| Architecture | delta sigma=0 | delta sigma=0.25 | delta sigma=0.5 | delta sigma=0.75 | delta sigma=1 | Delta AUC |
|---|---:|---:|---:|---:|---:|---:|
| narrow CNN6 | -0.82 | 7.67 | -0.31 | 2.27 | 0.59 | 2.38 |
| VGG-like CNN6 | -0.80 | 3.81 | 11.10 | 12.69 | 11.49 | 8.24 |

After subtracting the clean gap, the VGG-like clean-adjusted deltas are 4.61, 11.90, 13.49, and 12.29 points for sigma 0.25/0.5/0.75/1. The result is therefore not caused by weights-only starting with higher clean accuracy.

## Scale diagnostics

All-parameter L2 lowers the VGG-like input IF threshold from 6.59 to 4.21 and the mean hidden threshold from 7.50 to 5.11. Its BN-folded, threshold-normalized layer-gain geometric mean is 0.286 versus 0.196, a 1.46x ratio; the six-layer product proxy ratio is about 9.55x.

The corresponding narrow-CNN6 geometric-mean ratio is already about 1.42x, but its accuracy gap is much smaller. Effective-gain rescaling alone is therefore insufficient to predict classification robustness. Width, stage transitions, activation distributions, and classification margins determine whether the internal rescaling reaches the output decision.

## Conclusion boundary

This experiment supports an architecture-interaction hypothesis: decaying BN/IF parameters can cause a large robustness loss in a staged widening network even when clean accuracy improves. It does not support the simpler claim that the number of BN layers alone determines the loss, because the narrow and VGG-like models both have six BN layers. This is still a single-training-seed result and requires multi-seed confirmation.
