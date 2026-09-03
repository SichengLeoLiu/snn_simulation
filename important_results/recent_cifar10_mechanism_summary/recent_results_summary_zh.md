# 近期实验结果与机制结论

## 实验范围

核心结果均为 ANN 训练后转换到 SNN，CIFAR-10 VGG16，`L=T=16`，`rate_uniform`。

- pre-IF 与 post-IF 对比：MNE-L2、weight decay、L1，5 个训练 seed，固定绝对高斯噪声 `sigma=0...1`。
- 六方法尺度对照：seed 42，5 次噪声重复，噪声位于 `post_input_if`。比较 fixed absolute、lambda-relative、activation-relative 和 SNR-matched noise。
- 逐层机制统计：seed 42，`post_input_if`、固定 `sigma=1`，统计各 IF 层的 threshold、有效增益、margin-to-noise ratio 和 crossing probability。

五次 noise repeat 只衡量随机噪声采样波动，不能当作五个训练 seed。

## 1. 噪声位置会显著改变观察到的鲁棒性差距

| 噪声位置 | 方法 | clean | sigma=1 | 掉点 |
|---|---|---:|---:|---:|
| pre-IF | MNE-L2 | 90.378 | 90.156 | -0.222 |
| pre-IF | Weight decay | 90.594 | 83.216 | -7.378 |
| pre-IF | L1 | 89.874 | 89.184 | -0.690 |
| post-IF | MNE-L2 | 90.378 | 90.120 | -0.258 |
| post-IF | Weight decay | 90.594 | 23.200 | -67.394 |
| post-IF | L1 | 89.874 | 89.508 | -0.366 |

结论：MNE-L2 和 L1 在两个位置都稳定；weight decay 在 pre-IF 下仍明显下降，但 post-IF 下灾难性崩溃。post-IF Gaussian noise 绕过当前 IF 的量化边界并直接进入后续传播，因此对低尺度模型尤其严苛。

限制：pre-IF 与 post-IF 使用相同 `sigma` 并不代表相同有效扰动。当前 pre-IF 噪声在每个时间步独立采样，累计噪声标准差约为 `sqrt(T)*sigma_step`。两种位置必须分别做 threshold/SNR calibration，不能仅根据此图断言哪个位置本质上更容易。

## 2. BN gamma 是 all-parameter L2 崩溃的关键作用范围

seed 42、post-IF、`sigma=1` 的参数范围消融：

| L2/WD 作用范围 | clean | sigma=1 | 掉点 |
|---|---:|---:|---:|
| Conv/Linear weights only | 91.17 | 90.80 | -0.37 |
| Weights + BN beta | 90.71 | 90.30 | -0.41 |
| Weights + BN gamma | 90.97 | 22.56 | -68.41 |
| Weights + BN gamma + beta | 90.54 | 24.22 | -66.32 |

这说明在当前训练和评估协议下，正则 BN gamma 对崩溃近似是充分的，而正则 BN beta 不是。它定位了参数范围，但还不能单独说明因果路径是 `gamma -> lambda`。

## 3. lambda 重要，但 mean lambda 不是完整解释

六方法在输入 IF 位置的尺度和固定 `sigma=1` 结果：

| 方法 | input lambda | post-IF RMS | sigma=1 的实际 SNR | sigma/lambda | 掉点 |
|---|---:|---:|---:|---:|---:|
| MNE-standard | 2.096 | 0.838 | -1.54 dB | 0.48 | -0.03 |
| Weights-only | 1.012 | 0.499 | -6.05 dB | 0.99 | -0.37 |
| L1-all | 1.364 | 0.573 | -4.83 dB | 0.73 | -0.52 |
| MNE-all | 0.385 | 0.170 | -15.37 dB | 2.60 | -9.49 |
| L2-all | 0.234 | 0.103 | -19.77 dB | 4.28 | -68.80 |
| Weights + BN gamma | 0.222 | 0.102 | -19.79 dB | 4.51 | -68.47 |

L2-all 和 BN-gamma L2 同时压缩 lambda 与激活 RMS。固定 `sigma=1` 对它们相当于约 `-20 dB`、约 4.3--4.5 倍 threshold 的扰动，而 MNE-standard 只承受约 0.48 倍 threshold 的扰动。

因此，更准确的机制是：

`BN-gamma L2 -> 内部尺度压缩 -> 固定绝对噪声的相对强度上升 -> margin-to-noise ratio 降低 -> 量化边界 crossing 增多 -> accuracy 下降`。

lambda 是内部尺度的重要代理，但 lambda 单独不足以解释 MNE-all、L1-all 与 L2-all 的差异；还必须考虑 activation RMS、BN-folded effective gain、逐层 margin 和噪声传播。

## 4. SNR matching 几乎消除了灾难性差距

在 activation-SNR-matched `0 dB` 下，六种方法相对各自 clean 的掉点均不超过 0.26 个百分点：

| 方法 | clean | SNR=0 dB | 掉点 |
|---|---:|---:|---:|
| MNE-standard | 90.500 | 90.346 | -0.154 |
| Weights-only | 91.170 | 91.048 | -0.122 |
| L1-all | 89.850 | 89.594 | -0.256 |
| MNE-all | 89.550 | 89.470 | -0.080 |
| L2-all | 90.700 | 90.578 | -0.122 |
| Weights + BN gamma | 90.970 | 90.866 | -0.104 |

这是目前支持“尺度混杂”的最强证据：固定 absolute sigma 比较的不是相同相对强度的噪声。

但当前 SNR 只扫描到 `0 dB`，尚未覆盖 L2-all 在 absolute `sigma=1` 下实际经历的约 `-20 dB`。因此应该表述为“在当前测试范围内差距消失”，而不是宣布机制已经完全证明。

## 5. 逐层统计解释了固定绝对噪声如何演化成分类失败

在 VGG 后段，L2-all 和 Weights+BN-gamma 的 `rho=d50/sigma_eff` 降至约 `0.05--0.11`，对应 crossing probability 约 `0.62--0.74`。MNE-all 处于中间；MNE-standard、Weights-only 和 L1-all 通常保持更高 rho 与更低 crossing rate。

这里日志原字段 `P(E)` 实际由 `clean.ne(noisy)` 计算，因此表示的是 `P_cross=P(E^c)`，不是稳定事件概率。论文中必须改名。

逐层结果支持：尺度压缩只是起点，后续是否崩溃还取决于 BN-folded gain、有效噪声传播和量化 margin。它也解释了为什么简单的全局 `mean(gamma)/mean(lambda)` 无法排列六种方法的鲁棒性。

## 6. 对 MNE-L2 的当前评价

可以支持：

- MNE-standard 对固定绝对 post-IF noise 很稳定。
- MNE-standard、selective weights-only L2 和 L1 都避免了 all-parameter L2 的尺度崩溃。
- MNE-all 即使 lambda 较小，也比 L2-all 稳定，说明 MNE 对 effective weight/gain 有部分补偿作用。

暂时不能支持：

- MNE-L2 在所有公平归一化噪声条件下都优于 L1 或 weights-only L2。
- lambda 变小是鲁棒性下降的唯一原因。
- post-IF Gaussian noise 的结果可以直接推广到 raw-image noise、自然 corruption 或 adversarial robustness。

Hinge-MNE、Spectral-MNE 和其他已测试变体尚未稳定超过 MNE-standard。Fashion-MNIST 深窄网络扩展到 200 层也没有显示未正则 lambda 的 train-validation gap 随深度单调扩大，因此“lambda 自身明显过拟合”目前没有实验证据，不适合作为论文主线。

## 推荐论文叙事

当前最强的故事不是“新 MNE 一定优于所有 L2-family 方法”，而是：

1. all-parameter L2 尤其是 BN-gamma regularization 会选择低内部尺度；
2. 固定绝对 post-IF noise 打破近似尺度等价性，使低尺度模型承受更强的有效扰动；
3. 低尺度与高有效增益共同降低逐层 margin-to-noise ratio，并造成 late-layer boundary crossing；
4. MNE-standard、L1 和 selective L2 通过不同方式避免这一失败模式；
5. SNN robustness evaluation 必须同时报告 absolute noise 与 scale-normalized noise。

## 下一步实验优先级

1. 将 SNR 扩展到 `-5,-10,-15,-20,-25 dB`，检验六方法曲线在完整强度范围是否重合。
2. 实现严格的 pre-IF relative noise：独立时间噪声使用 `sigma_step=alpha*lambda/sqrt(T)`，并基于累计膜电位 `z` 计算 SNR。
3. 做 gauge-rescaling 因果实验：保持 clean function 近似不变，只改变内部尺度，比较 absolute 与 relative noise。
4. 将 RMS calibration 从 test loader 改为 train/validation calibration set。
5. 在 CIFAR-10 做 3--5 个训练 seed，再扩展 CIFAR-100、ResNet-18 和 ImageNet；不要用 noise repeat 替代训练 seed。

