# 实验清单（更新于 2026-09-03）

状态说明：**已完成**表示已有预定的多 seed 证据；**单 seed 试验完成**表示诊断实验已成功运行，但还不能作为确认性证据；**代码就绪**表示实现已经存在，但本地没有找到运行结果；**未完成**表示仍需实现或运行。

| 状态 | 实验 | 当前设置 | 这个实验说明了什么 | 下一步 |
| --- | --- | --- | --- | --- |
| 单 seed 试验完成 | L2 regularization scope | VGG16、CIFAR-10、单 seed；比较 weights only、加入 BN-$\beta$、加入 BN-$\gamma$ 和全部参数 | BN-$\gamma$ 是否进入 L2 scope 会显著改变转换后的噪声稳定性，为本文提供研究动机 | 补 3--5 seeds；作为动机消融，而不是主要性能比较 |
| 已完成 | MNE-L2 主实验 | VGG16、CIFAR-10/100、五 seeds、$\sigma=0\ldots5$ | MNE-L2 在高内部噪声下显著优于 L2-all、L2-wo 和 L1-wo，但存在一定 clean accuracy 代价 | 增加 clean-accuracy-matched 和基于验证集选择的比较 |
| 已完成 | VGG 深度迁移 | VGG13/16/19、CIFAR-10/100、五 seeds | MNE-L2 的高噪声优势不是 VGG16 的特例 | 在附录中压缩为表格或一张总览图 |
| 已完成 | ResNet 架构迁移 | ResNet-18、CIFAR-10/100、五 seeds | MNE-L2 可以迁移到残差网络；CIFAR-10 的 $\sigma=5$ accuracy 为 81.85%，L2-wo 为 53.62% | 增加 clean-matched Pareto 比较，并解释 CIFAR-100 的 clean gap |
| 已完成 | One-sided VGG | VGG16、CIFAR-10/100、五 seeds | 风险加权 L2 在串行 VGG 拓扑上有效；$\sigma=5$ accuracy 为 75.70%/45.78%，MNE-L2 为 72.06%/41.14% | 作为重要扩展保留，但限制其架构适用范围的表述 |
| 已完成，负面结果 | One-sided ResNet | ResNet-18、CIFAR-10/100、五 seeds | 原始 One-sided 不能稳定迁移到残差拓扑；$\sigma=5$ accuracy 仅为 28.96%/18.97% | 诊断从 MNE 到 One-sided 的变换，不再只调整 $\alpha/\tau$ |
| 单 seed 试验完成 | One-sided 强度归一化 | ResNet-18、CIFAR-10/100、单 seed | 对 $q$ 做均值归一化后，seed 42 的高噪声结果优于未归一化 One-sided，但仍未稳定恢复 MNE-L2 的表现 | 将归一化纳入 bridge 实验；只有通过 bridge 筛选后才扩展到多 seeds |
| 单 seed 试验完成 | One-sided q-assignment 消融 | VGG16、CIFAR-10/100、单 seed；risk、shuffle、strength 和 identity | risk 与 shuffle 几乎没有差异，而 strength-only 和 identity 明显崩溃；结果支持异质或层级正则分配，但不支持当前通道排序 | 结合已完成的敏感度相关性，并进行高/中/低风险因果干预 |
| 单 seed 试验完成 | One-sided q-cap 与层级控制 | VGG16、CIFAR-10/100、单 seed；$q_{\max}=4,8,16$、无上限和 layer-level control | 更严格的 cap 会逐步降低高噪声准确率，无上限配置最好；这说明系数尾部或总正则强度可能重要，但不能证明精确通道排序有效 | 在梯度强度或正则预算匹配后重复，才能把尾部解释为因果因素 |
| 单 seed 试验完成 | MNE-L2 component ablation | VGG16、CIFAR-10/100、单 seed；L2-wo、Effective L2、MNE-L2 和 no-detach | 加入 $1/\lambda^2$ 后优于 Effective L2；no-detach 在该筛选中最强（$\sigma=5$ 为 81.69%/54.49%），但尺度、聚合和梯度路径仍混在一起 | 完成 $\lambda/\gamma$ detach 析因消融，并匹配正则梯度强度 |
| 部分完成 | No-detach MNE-L2 多 seed 验证 | VGG16、CIFAR-10/100；输出目录名为 `5seed` | 当前汇总中 no-detach 仍为 `n_seeds=1`，因此目前只能确认 seed 42 的 81.69%/54.49% | 找到或补跑另外四个 no-detach checkpoints，然后重新生成汇总 |
| 未完成 | $\lambda/\gamma$ detach 析因消融 | detach both、仅 detach $\lambda$、仅 detach $\gamma$、均不 detach | 用于判断 no-detach 的收益来自 threshold、BN gain，还是两者交互 | P0：先单 seed 筛选并检查 scale escape，再对有效配置做五 seeds |
| 单 seed 试验完成，结论不足 | 跨量化等级的 $L^2$ scaling | VGG16、CIFAR-10/100、单 seed；$L=T\in\{4,8,16\}$，比较有无 $(L/16)^2$ | 固定系数下，低 $L$ 训练出现严重 clean collapse，scaled 形式也没有稳定胜出；当前结果不能验证理论中的 $L^2$ 因子 | 运行已经准备好的低 $L$ L2-wo/L2-all controls，并使用 clean 或梯度匹配系数 |
| 单 seed 试验完成 | VGG block/layer-role ablation | VGG16、CIFAR-10/100、单 seed | early/middle/late coverage 会影响稳健性，但这是定位实验，不是 component ablation | 放入附录，不将结果泛化为普遍的层重要性结论 |
| 单 seed 试验完成 | ResNet layer-role ablation | ResNet-18、CIFAR-10/100、单 seed；terminal、shortcut 和 preactivation roles | residual terminal 与 shortcut 周围的映射较重要，串行网络的局部风险不能直接迁移到 ResNet | 用于设计 path-aware objective，不作为某一 mapping 的最终证据 |
| 单 seed 试验完成，负面结果 | GA-MNE | VGG16/ResNet-18、CIFAR-10/100、单 seed；branch variance 与 covariance | 简单的分支方差分配不能解决 ResNet 问题；当前 probe 下 covariance 很小 | 不扩展到五 seeds；改用实测 boundary risk 和 downstream sensitivity |
| 单 checkpoint 诊断完成 | 理论风险与实际敏感度 | VGG16、CIFAR-10/100、每种方法一个 checkpoint；$\sigma=0.25,0.5,1$；层/通道 Spearman 和 clean/noisy paired measurements | 原始 $r$ 在层内跟踪 BN-folded amplification，但不能跟踪实际 crossing；$\widehat P_{\mathrm{bound}}$ 与 crossing 高度相关，$\widehat P_{\mathrm{bound}}D$ 与输出影响高度相关 | 将 $r$ 表述为正则启发式，而不是已验证的通道风险估计器；使用新分数做因果干预 |
| 部分完成 | Margin-aware One-sided | 已完成离线 $\widehat P_{\mathrm{bound}}=2Q(d/\sigma_{\mathrm{eff}})$ 诊断；尚未验证训练目标 | Margin-aware probability 明显比原始 MNE risk 更适合估计 crossing，但预测相关性不等于用它训练能够提高稳健性 | 在训练中实现或缓存 paired-probe estimator，并进行预算匹配的 seed-42 筛选 |
| 单 seed 试验完成，负面结果且 $D$ 测试无效 | Graph-Margin One-sided | ResNet-18、CIFAR-10/100、单 seed；$p$、$pD$、$pm$ 和 $pDm$ | $p$ 仅小幅改善 CIFAR-10，对 CIFAR-100 无稳定改善；branch multiplier $m$ 有害。由于极小的平方梯度被替换为 1，实际出现 $pD=p$、$pDm=pm$，因此 downstream sensitivity 并未被真正测试 | 修复尺度无关的 $D$ 归一化，使 $p$ 与已验证的 paired estimator 一致，移除 $m$，并固定总正则预算后重跑单 seed |
| 代码就绪，尚未运行 | MNE-to-One-sided bridge | ResNet-18；包含 uniform、raw、normalized、clipped 和 one-sided coefficient transforms，并统一为 layer-mean reduction | 用于定位 normalization、clipping、thresholding、coverage 或 aggregation 中哪一步破坏了 MNE-L2 在 ResNet 上的效果 | P0：首先运行 Raw，并要求它复现 Old MNE；成功后再运行后续 bridge cells |
| 已完成 | 稀疏度与估算算术能耗 | VGG16、CIFAR-10/100、$T=4,8,16$、五 seeds | MNE-L2 比 L2-all 发放更稀、估算算术能耗更低；CIFAR-10、$T=16$ 时为 0.472 对 0.585 mJ | 使用相同的 clean-only 能耗口径补充最终改进方法 |
| 已完成 | Clean/noisy firing 对照 | VGG16、CIFAR-10/100、五 seeds | 噪声会改变 firing，尤其会提高不稳定方法的发放率 | noisy firing 只作为机制诊断，不作为部署能耗 |
| 单 seed 试验完成，负面迁移结果 | CIFAR-C/CIFAR-100-C | VGG16、单 seed；L2-all、L2-wo、MNE-L2 和 One-sided；15 种 corruption、五个 severity | MNE-L2 和 One-sided 在标准输入 corruption 上没有超过 L2-wo，说明其优势针对内部 post-IF perturbation，而不是一般 corruption robustness | 只作为适用范围控制实验报告，不声称通用 corruption robustness |
| 已完成 | 替代噪声注入位置 | pre-first-conv noise、VGG16、CIFAR-10/100、五 seeds | MNE-L2 的优势具有协议依赖性；输入侧噪声足够强时所有方法都会崩溃 | 增加受控的 intermediate-IF injection sites，避免过度泛化 |
| 部分完成 | 多层 internal-IF noise | 有限模型与三 seed pilot | MNE-L2 在部分中间层协议下仍有优势，但覆盖不完整 | 统一 VGG16/ResNet18 的注入位置和噪声强度网格 |
| 部分完成 | ImageNet-1k 扩展 | ResNet-18/34、单 checkpoint；当前主要报告 Top-1 noise curve | 中等噪声下的排序大致与 CIFAR 一致，但高噪声下所有方法均会崩溃 | 至少评估三个独立训练的 checkpoints；统一报告 Top-1、Top-5、full/high-noise robust AUC，并补 NLL/ECE |
| 部分完成 | 无 BN 网络 | CIFAR MLP、单 seed pilot | 当前结果不能证明 MNE/One-sided 可以推广到无 BN 网络 | 将 BN-folded risk 替换为 margin/operator-gain 定义，并补 3--5 seeds |
| 未完成 | 风险分数因果干预 | 使用 $\widehat P_{\mathrm{bound}}D$ 或其他已验证分数选择等规模的 high/middle/low groups | 用于检验相关性分数是否真的定位了修改后会影响稳健性的通道 | P0：与等规模随机组比较，并报告 clean、crossing、logit 和 robust-accuracy 变化 |
| 未完成 | 公平超参数搜索 | 独立 validation split；每个 baseline 单独搜索 coefficient | 排除 MNE-L2 只是使用了更合适正则系数的解释 | P0 |
| 未完成 | Clean-accuracy-matched Pareto | 联合比较 clean accuracy、robust AUC 和 $\sigma=5$ accuracy | 判断稳健性提升是否只是通过牺牲 clean accuracy 获得 | P0 |
| 未完成 | Noise-augmentation baseline | 训练时在相同位置注入相同噪声 | 检验直接噪声训练是否比 MNE-based regularization 更有效 | P0 |
| 未完成 | 第二种 conversion backbone | QCFS 之外增加一种 quantized ANN-to-SNN conversion 方法 | 检验正则方法是否独立于 QCFS conversion rule | P1 |
| 未完成 | 更多架构 | MobileNetV2、WideResNet 或其他非 VGG/ResNet 模型 | 支撑 architecture-general claim | P1；选择一种架构深入验证即可 |
| 未完成 | 方差匹配的噪声分布 | post-input-IF、每 timestep 独立；Gaussian、bounded Uniform 和 Laplace 使用相同 per-step RMS | 判断优势是否依赖高斯尾部假设，还是来自更一般的 boundary-margin 稳定性 | P1：先在 VGG16/ResNet-18、CIFAR-10/100 上做单 seed 筛选，再对有差异的设置做五 seeds |
| 代码部分就绪，尚无可比结果 | 时间相关噪声 | VGG 已支持 Gaussian/pink；计划比较 iid Gaussian、AR(1)（$\rho=0.5,0.9$）、pink noise 和样本内静态 offset，并匹配边际 RMS | 检验独立 timestep 假设是否关键，以及时间相关性是否通过累积膜电位放大误差 | P1：先修复 ResNet 当前选择 pink 仍实际采样 white Gaussian 的实现，再固定同一 noise grid 运行 |
| 未完成 | IF threshold 失配与抖动 | 比较每个神经元/通道固定的 $\lambda' = \lambda(1+\eta)$ 与每 timestep 独立的 threshold jitter | 直接测试 boundary spacing 被器件失配或动态阈值波动改变时，MNE-L2 是否仍有效 | P1：分别报告 clean accuracy、robust AUC、crossing rate 和 firing density |
| 未完成 | 权重与 BN-gain 扰动 | inference-only relative multiplicative weight noise、absolute additive weight noise，以及静态 BN-$\gamma$ gain mismatch | 区分 MNE-L2 对相对放大误差、绝对参数误差和通道校准误差的适用性 | P1：按 layer-wise RMS 匹配强度，避免不同参数尺度造成不公平比较 |
| 未完成 | Spike deletion/insertion | 每 timestep 对 spike 做 Bernoulli deletion、insertion 或 bit-flip，建议 $p\in\{0.01,0.05,0.1\}$ | 检验方法能否应对离散通信错误，而不只是连续 feature-map 噪声 | P2：先测 input IF 后单点注入，再扩展到中间层；报告 error-rate curve 和 robust AUC |
| 未完成 | 静态通道 offset/gain | $z'_{t,c}=(1+g_c)z_{t,c}+b_c$，其中 $g_c,b_c$ 在整个样本的 $T$ 个 timestep 内固定 | 模拟通道级校准偏差，并检验基于 BN-folded amplification 的方法是否特别适合此类部署误差 | P2：与同 RMS 的 iid Gaussian 对照，区分静态失配与随机噪声 |
| 未完成 | 目标检测（主扩展） | Pascal VOC 2007+2012、SSD300-VGG16；clean/noisy mAP@0.5、per-class AP 和 robust-mAP AUC | 在保持 VGG 主干可比性的同时，检验分类之外的定位与分类联合任务 | P1：先完成 VOC 单 seed feasibility，再用至少三个 seeds 比较 L2-wo、MNE-L2 和最佳改进方法 |
| 未完成 | 目标检测（残差架构扩展） | COCO、RetinaNet 或 FCOS + ResNet-FPN；AP@[.5:.95]、AP50、AP75 和 robust-AP AUC | 同时测试 task generality 与 residual/multi-branch topology，约束只在 VGG 上有效的结论 | P2：仅在 ResNet bridge 或 MNE-L2 主线确定后开展 |
| 未完成 | 关键词识别 | Speech Commands v2、带 BN 的 DS-CNN/ResNet；Accuracy、macro-F1、NLL/ECE 和 robust AUC | 检验方法是否跨越视觉模态，并观察时间输入与内部 SNN 噪声的交互 | P2：统一音频前处理并明确噪声只注入转换后 feature map，不混同输入声学噪声 |
| 未完成 | 语义分割 | Pascal VOC 或 Cityscapes、可转换的 FCN/DeepLab；mIoU、pixel accuracy 和 robust-mIoU AUC | 检验像素级密集预测是否保留 boundary-margin 稳定性收益 | P2：可作为 COCO 检测的替代任务，不必同时完成两项大规模扩展 |
| 部分完成，尚未统一 | Robustness curve 汇总指标 | 现有部分 scorecard 已计算 full AUC 与 high-noise AUC；主实验仍多依赖单点 $\sigma=5$ accuracy | AUC 能避免只选择一个有利噪声强度，并区分全区间表现与高噪声尾部表现 | P0：所有方法统一报告 clean、$\sigma=1/3/5$、归一化 full rAUC 和 high-noise rAUC，AUC 除以积分区间长度后再跨实验比较 |
| 部分完成 | 预测与表示稳定性指标 | 已有 paired diagnostics 的 crossing 和输出影响；尚未在主五-seed比较中统一报告 | Prediction flip rate、logit RMS、KL/JS divergence、classification-margin drop 与 $P_{\mathrm{cross}}$ 可将准确率变化连接到机制 | P1：在相同样本和噪声 realization 下成对统计，正文保留 1--2 个机制指标，其余放附录 |
| 未完成 | 概率校准指标 | NLL、Brier score、ECE 和 noisy reliability diagram | 判断方法是在噪声下真正保持预测分布，还是仅维持 Top-1 decision | P1：至少在 CIFAR-10/100 与 ImageNet 上报告 clean/noisy NLL 和 ECE；固定 binning 规则 |
| 部分完成 | Accuracy--latency--energy 指标 | VGG16 已有 $T=4,8,16$ 的 accuracy、firing 和估算能耗；缺少每个 $T$ 的统一 robust AUC | 判断稳健性收益能否在低 timestep 与能耗约束下保持，而不是只在 $T=16$ 成立 | P1：每个 $T$ 报 clean、rAUC、$\sigma=5$、firing 和 clean-only arithmetic energy，并画 Pareto frontier |
| 未完成 | 配对效应量与不确定性 | 同一 seeds 下比较方法；报告 paired accuracy/rAUC difference、sample std 和 95% bootstrap CI | 量化改进幅度及其跨 seed 稳定性，避免只比较均值和最佳曲线 | P0：预先固定主指标与噪声区间；五 seeds 不依赖单独的显著性阈值下结论 |
| 未完成 | 真实神经形态硬件测量 | device-level energy、latency 和 reliability | 支撑硬件部署结论 | P2；没有硬件时只能保留算术能耗估算 |

## 当前决策点

当前最有价值的下一步是运行 ResNet 的 MNE-to-One-sided bridge。应先运行 **Raw** 条件；如果它不能在相同 coverage、reduction、detach policy、训练 schedule 和梯度强度下复现 Old MNE，就需要先修复两者的目标函数等价性，再解释 normalization、clipping 或 thresholding 的影响。

同时需要补齐 no-detach 的其余 seeds：尽管输出目录名包含 `5seed`，当前汇总文件中 no-detach 仍然只有一个 seed。
