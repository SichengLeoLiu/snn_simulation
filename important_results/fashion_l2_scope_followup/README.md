# Fashion-MNIST L2 Parameter-Scope Follow-up

## Question

Why does L2 applied to all trainable parameters lose much more ANN-to-SNN robustness than the mathematically matched L2 applied only to Conv/Linear weights?

The experiment is deliberately scoped to **post-input-IF Gaussian noise**. It does not test raw-image noise.

## Protocol

- Dataset: Fashion-MNIST.
- Training: ANN (`T=0`), 30 epochs, best-clean checkpoint.
- Conversion/evaluation: QCFS SNN, `T=L=16`, `rate_uniform`.
- Noise site: immediately after the input IF module.
- Main model: `cnn6_vgg`, channels `c8-c8-pool-c16-c16-pool-c32-c32`.
- Matched L2 strength: optimizer weight decay `5e-4` versus explicit `0.5 * 5e-4 = 2.5e-4 * sum(p^2)`.
- Main replication: training seeds 40-44; three noise draws per nonzero absolute sigma.
- Mechanism ablations: seed 42; five noise draws per nonzero sigma.

The matched-checkpoint pilot showed zero maximum parameter difference between optimizer weight decay and explicit L2 when both use the same parameter set. The implementation form is therefore not the source of the observed gap; the parameter set is.

## Main Five-Seed Result

| L2 scope | Clean accuracy | Accuracy at sigma=1 | Drop | Curve AUC |
|---|---:|---:|---:|---:|
| Conv/Linear weights only | 87.40 +/- 0.19 | 44.77 +/- 5.53 | 42.63 +/- 5.37 | 69.51 +/- 3.58 |
| All trainable parameters | 88.42 +/- 0.33 | 36.91 +/- 6.76 | 51.51 +/- 6.79 | 64.70 +/- 3.88 |

Paired across the five training seeds:

- At `sigma=1`, weights-only minus all-parameters is `+7.85 +/- 5.79 pp`.
- After subtracting each model's clean-accuracy difference, the paired advantage is `+8.88 +/- 5.91 pp`.
- The paired absolute-noise curve-AUC advantage is `+4.80 +/- 2.60 pp`.

This confirms that the VGG-like result is not a seed-42 accident, although the remaining between-seed variation is still substantial.

## Which Parameters Cause the Loss?

Seed-42 selective explicit-L2 ablation:

| Penalized parameters | Clean | Accuracy at sigma=1 | Drop |
|---|---:|---:|---:|
| Weights only | 87.49 | 48.25 | 39.24 |
| Weights + BN | 88.37 | 42.24 | 46.13 |
| Weights + IF thresholds | 88.06 | 33.08 | 54.98 |
| Weights + BN + IF thresholds | 87.55 | 35.80 | 51.75 |
| All parameters | 88.29 | 37.07 | 51.22 |

Directly regularizing IF thresholds produces the largest loss in this ablation. BN regularization also hurts, but less strongly. The non-monotonic ordering of `W+IF`, `W+BN+IF`, and all-parameters indicates interaction and compensation, so the result should not be described as a simple additive decomposition.

## Scale Evidence

The scale change is highly reproducible over five training seeds:

| Quantity | Weights only | All parameters | Change |
|---|---:|---:|---:|
| Input IF threshold | 6.585 +/- 0.025 | 4.222 +/- 0.067 | -35.9% |
| Mean hidden IF threshold | 7.507 +/- 0.008 | 5.107 +/- 0.010 | -32.0% |
| Mean absolute BN gamma | 1.224 +/- 0.004 | 0.941 +/- 0.003 | -23.1% |
| Geometric mean folded gain | 0.195 +/- 0.003 | 0.286 +/- 0.005 | +46.2% |
| Product of folded gains | 5.59e-5 | 5.47e-4 | 9.79x |

The ANN can preserve clean accuracy through compensating changes in weights, BN scales, biases, and IF thresholds. After conversion, however, the fixed absolute perturbation is large relative to the lower threshold, while the effective folded gain increases because threshold shrinkage dominates BN-gamma shrinkage.

## Absolute Versus Relative Noise

Seed-42 scale controls provide the strongest evidence for the mechanism:

| Noise definition | Weights only endpoint | All-parameters endpoint | W-only minus all |
|---|---:|---:|---:|
| Absolute `sigma=1` | 48.25 | 37.07 | +11.18 |
| `sigma=0.2 * input threshold` | 35.67 | 43.43 | -7.76 |
| `sigma=1.0 * clean post-input-IF RMS` | 25.19 | 27.01 | -1.82 |

The large absolute-noise advantage disappears under scale-normalized noise and can reverse. This means the defensible claim is **fixed-absolute-noise scale sensitivity**, not an unconditional intrinsic instability of all-parameter L2. The exact relative-noise ordering depends on whether threshold or activation RMS defines the scale.

## Layerwise and Output Diagnostics

At absolute `sigma=1` on 2,048 samples (seed 42):

| Metric | Weights only | All parameters |
|---|---:|---:|
| IF6 relative representation L2 | 0.887 | 0.918 |
| IF6 spike mismatch rate | 10.75% | 13.82% |
| Prediction flip rate | 50.22% | 63.26% |
| Clean-correct to noisy-wrong rate | 48.69% | 62.06% |
| Mean margin change | -7.41 | -8.77 |

The representation-norm difference at IF6 is modest, but it is accompanied by more spike changes and a much larger output decision failure rate. Thus a norm-only diagnostic does not capture the whole failure; quantization decisions and class margins are also involved.

## Architecture Control

All models have six Conv-BN-IF layers. Within each width, early-pool and staged-pool variants have the same parameter count.

| Width | Pooling | Parameters | W-only minus all at sigma=1 |
|---|---|---:|---:|
| Narrow | Early | 2,709 | +0.59 pp |
| Narrow | Staged | 2,709 | +5.88 pp |
| Wide | Early | 33,961 | +7.00 pp |
| Wide | Staged | 33,961 | +11.49 pp |

Both width and staged pooling expose a larger gap, with the largest result in the wide-staged VGG-like model. This refutes a simple "more BN layers means a larger gap" explanation: depth and BN count are fixed here. These four cells are currently seed 42 only and should be treated as structural evidence, not a population estimate.

## Defensible Research Conclusion

1. MNE-L2 is robust relative to the original all-parameter optimizer-L2 baseline, but earlier five-seed results show that Orthogonal regularization is nearly tied and No Reg is also strong. MNE-L2 is not yet uniquely superior.
2. The optimizer API is not the cause: matched optimizer and explicit L2 are identical when their parameter scope matches.
3. In the VGG-like Fashion-MNIST model, all-parameter L2 reproducibly compresses BN and especially IF thresholds, increasing effective gain and making fixed absolute post-IF noise relatively larger.
4. Direct IF-threshold regularization is the strongest negative factor in the selective ablation; BN contributes and interacts with it.
5. The failure is conditional on architecture and noise scaling. It should not be generalized to raw-image noise or to every CNN depth.

## Short Oral Summary

"Our earlier experiments seemed to show that ordinary L2 was uniquely fragile while MNE-L2 was robust. The new controls show that this was too broad. When optimizer weight decay and explicit L2 act on the same parameters, they produce the same checkpoint. The real difference is parameter scope. In a VGG-like six-layer network, applying L2 to all trainable parameters lowers the input IF threshold from about 6.58 to 4.22 and increases the effective folded gain, while clean accuracy is preserved through parameter compensation. Under a fixed absolute post-input-IF perturbation, weights-only L2 is 7.85 percentage points better at sigma one over five training seeds. Directly regularizing IF thresholds creates the largest loss. Most importantly, the gap disappears or reverses when noise is normalized by each model's threshold or clean activation RMS. Therefore, the current result supports a scale-sensitivity explanation under fixed absolute internal noise, not a universal claim that all-parameter L2 or deep BN networks are intrinsically non-robust. MNE-L2 still avoids this failure mode, but it must be compared against weights-only L2, Orthogonal, and No Reg before claiming a unique advantage."

## Next Experiments

1. Expand the two most informative selective scopes, `weights-only` and `weights+IF`, to five training seeds.
2. Repeat the same parameter-scope and scale-normalized protocol on CIFAR-10/100 VGG checkpoints.
3. Add threshold-clamped or threshold-reparameterized L2 to test whether preventing threshold shrinkage recovers absolute-noise robustness without sacrificing clean accuracy.
4. Keep raw-image, normalized-input, pre-input-IF, and post-input-IF perturbations as separate experimental claims.
