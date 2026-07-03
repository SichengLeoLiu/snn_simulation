# 实验汇总（QCFS_simulation_FI）

- 更新时间：2026-07-02 21:27:22
- 统计范围：`/Users/S4142196/Documents/Phd_Document/codes/QCFS_simulation_FI` 当前可见结果文件（主要是 `noise3_exp` 下 CSV/PNG）

## 1) CIFAR-10 VGG16 三路正则噪声扫描（strict-seed, L=16, T=16, rate_uniform）

- 结果文件：`QCFS_simulation/noise3_exp/cifar10_vgg16_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16/cifar10_vgg16_strict_seed_three_regs_noise_sweep_mean_std.csv`
- 核心结果（5 seeds mean）：

| 方法 | sigma=0 acc(mean±std) | sigma=1 acc(mean±std) | Δ(sigma1-sigma0) |
|---|---:|---:|---:|
| L2 (`weight_decay`) | 89.234 ± 0.800 | 22.577 ± 14.000 | -66.657 |
| MNE L2 (`mne_l2`) | 89.200 ± 0.350 | 88.800 ± 0.200 | -0.400 |
| MNE L2 + WD (`mne_l2_wd`) | 89.300 ± 0.400 | 85.000 ± 0.550 | -4.300 |

## 2) MNIST CNN2 三路正则噪声扫描（strict-seed, L=16, T=16, rate_uniform）

- 结果目录：`cnn_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16/`（每个 arch × 正则 × seed 的 `noise_sweep_combined_L_T.csv`）
- 关键对比（聚合 5 seeds 后，sigma=0 与 sigma=1）：

| 架构 | 方法 | sigma=0 acc(mean) | sigma=1 acc(mean) | Δ(sigma1-sigma0) |
|---|---|---:|---:|---:|
| `cnn2_c2_c4` | weight_decay | 95.902 | 76.990 | -18.912 |
| `cnn2_c2_c4` | mne_l2 | 93.678 | 92.068 | -1.610 |
| `cnn2_c2_c4` | no_regularization | 95.384 | 75.468 | -19.916 |
| `cnn2_c4_c8` | weight_decay | 98.452 | 86.288 | -12.164 |
| `cnn2_c4_c8` | mne_l2 | 96.140 | 95.828 | -0.312 |
| `cnn2_c4_c8` | no_regularization | 98.222 | 83.362 | -14.860 |
| `cnn2_c8_c16` | weight_decay | 98.816 | 80.326 | -18.490 |
| `cnn2_c8_c16` | mne_l2 | 95.098 | 93.432 | -1.666 |
| `cnn2_c8_c16` | no_regularization | 98.804 | 89.052 | -9.752 |
| `cnn2_c16_c32` | weight_decay | 98.912 | 82.340 | -16.572 |
| `cnn2_c16_c32` | mne_l2 | 97.818 | 97.152 | -0.666 |
| `cnn2_c16_c32` | no_regularization | 99.070 | 93.048 | -6.022 |

## 3) MNIST FC3 三路正则噪声扫描（strict-seed, rate_uniform）

- 结果文件：`QCFS_simulation/noise3_exp/ablation_mne_l2_vs_weight_decay_l16_fc3_h4_h8_h16_h32_h64_h128/strict_seed_train_rate_uniform_L16_T16/strict_seed_train_noise_sweep_fc3_h4_h8_h16_h32_h64_h128_mean_std.csv`
- 各模型在高噪声端（sigma=1）最优正则（按每个 hidden size 比较）：

| 架构 | 最优正则 | sigma=0 acc(mean±std) | sigma=1 acc(mean±std) | Δ |
|---|---|---:|---:|---:|
| `fc3_h4` | mne_l2 | 67.074 ± 18.155 | 65.824 ± 18.719 | -1.250 |
| `fc3_h8` | mne_l2 | 81.394 ± 11.242 | 80.318 ± 11.854 | -1.076 |
| `fc3_h16` | mne_l2 | 93.444 ± 1.225 | 92.740 ± 1.399 | -0.704 |
| `fc3_h32` | weight_decay | 97.168 ± 0.292 | 96.424 ± 0.321 | -0.744 |
| `fc3_h64` | mne_l2 | 97.584 ± 0.101 | 97.130 ± 0.141 | -0.454 |
| `fc3_h128` | mne_l2 | 97.774 ± 0.093 | 97.504 ± 0.148 | -0.270 |

## 4) MNIST FC3（weight_decay）L×T 精度扫描

- 结果文件：`QCFS_simulation/noise3_exp/fc3_wd_strict_seed_normal_L_T_acc/fc3_wd_strict_seed_normal_L_T_acc_mean_std.csv`
- 全局最佳：`fc3_h64` 在 L=4, T=32，acc=97.944% ± 0.074
- L16 相对 L2 提升最大：`fc3_h4` T=32，Δ=11.664
- L16 相对 L2 下降最大：`fc3_h8` T=2，Δ=-30.426

### 4.0) FC3 各 hidden size 的完整 L×T 表（acc_mean±acc_std）

### `fc3_h4`
| L \ T | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 2 | 55.420±30.109 | 55.812±30.499 | 55.688±30.420 | 55.658±30.480 | 55.616±30.511 |
| 4 | 59.278±11.586 | 71.054±12.250 | 73.018±12.217 | 73.516±12.334 | 73.496±12.401 |
| 8 | 42.178±24.100 | 56.922±30.682 | 60.110±32.410 | 60.300±32.511 | 60.538±32.429 |
| 16 | 39.684±20.704 | 50.744±22.697 | 63.158±19.029 | 66.798±16.163 | 67.280±15.800 |
| 32 | 50.098±14.715 | 61.298±10.218 | 74.014±8.118 | 77.736±5.987 | 78.268±5.856 |

### `fc3_h8`
| L \ T | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 2 | 86.254±6.726 | 87.272±6.395 | 87.482±6.361 | 87.544±6.398 | 87.520±6.384 |
| 4 | 82.978±7.675 | 88.218±4.500 | 88.790±4.290 | 88.930±4.141 | 89.068±4.173 |
| 8 | 67.992±17.816 | 80.316±12.626 | 85.324±7.398 | 85.864±6.614 | 86.066±6.390 |
| 16 | 55.828±22.557 | 70.356±21.397 | 74.808±22.455 | 77.160±19.233 | 77.524±18.714 |
| 32 | 59.854±19.252 | 73.996±18.852 | 81.388±11.887 | 84.022±8.130 | 84.276±7.902 |

### `fc3_h16`
| L \ T | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 2 | 94.318±0.393 | 95.098±0.530 | 95.342±0.471 | 95.342±0.537 | 95.336±0.489 |
| 4 | 91.496±3.043 | 94.276±1.864 | 94.840±1.582 | 94.916±1.644 | 94.918±1.607 |
| 8 | 87.344±2.948 | 92.906±1.524 | 94.040±1.297 | 94.320±1.096 | 94.340±1.117 |
| 16 | 82.568±5.551 | 91.748±1.894 | 93.562±1.410 | 93.876±1.353 | 93.912±1.340 |
| 32 | 79.540±7.052 | 90.902±2.124 | 93.662±1.205 | 94.198±1.093 | 94.324±1.068 |

### `fc3_h32`
| L \ T | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 2 | 96.502±0.266 | 97.016±0.224 | 97.184±0.238 | 97.236±0.232 | 97.254±0.205 |
| 4 | 95.756±0.256 | 97.014±0.232 | 97.176±0.213 | 97.258±0.217 | 97.264±0.221 |
| 8 | 95.154±0.792 | 96.852±0.404 | 97.240±0.254 | 97.314±0.245 | 97.326±0.213 |
| 16 | 94.230±0.829 | 96.584±0.434 | 97.040±0.315 | 97.176±0.316 | 97.142±0.244 |
| 32 | 94.180±0.703 | 96.506±0.272 | 97.038±0.205 | 97.142±0.214 | 97.206±0.171 |

### `fc3_h64`
| L \ T | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 2 | 97.248±0.170 | 97.680±0.068 | 97.780±0.101 | 97.840±0.100 | 97.824±0.107 |
| 4 | 96.806±0.129 | 97.748±0.108 | 97.850±0.106 | 97.912±0.061 | 97.944±0.074 |
| 8 | 96.088±0.214 | 97.342±0.160 | 97.660±0.166 | 97.620±0.135 | 97.636±0.113 |
| 16 | 95.980±0.177 | 97.314±0.127 | 97.590±0.126 | 97.710±0.118 | 97.740±0.110 |
| 32 | 95.222±1.160 | 96.408±1.124 | 96.688±1.052 | 96.812±1.106 | 96.944±0.864 |

### `fc3_h128`
| L \ T | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 2 | 95.440±2.169 | 96.084±1.785 | 96.562±1.289 | 96.532±1.267 | 96.576±1.258 |
| 4 | 95.596±0.886 | 96.392±1.063 | 96.450±1.040 | 96.496±1.069 | 96.536±1.098 |
| 8 | 95.682±0.578 | 96.742±0.853 | 96.968±0.944 | 97.048±0.902 | 97.054±0.932 |
| 16 | 96.108±0.157 | 97.468±0.121 | 97.718±0.160 | 97.796±0.130 | 97.778±0.126 |
| 32 | 95.532±0.797 | 96.542±1.080 | 96.798±1.052 | 96.814±1.086 | 96.842±1.105 |

### 4.1) MNIST CNN2（weight_decay）L×T 精度扫描（补充）

- 结果文件：`cnn_wd_strict_seed_normal_L_T_acc_mean_std.csv`
- 说明：表内为 `acc_mean ± acc_std`（5 seeds）

### `cnn2_c2_c4`
| L \ T | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 2 | 95.932±0.535 | 92.090±4.912 | 92.034±7.044 | 91.034±9.151 | 89.986±10.965 |
| 4 | 94.132±1.542 | 96.136±0.912 | 96.748±0.589 | 96.910±0.326 | 97.002±0.324 |
| 8 | 85.490±18.120 | 92.558±6.846 | 95.380±2.913 | 95.512±3.116 | 95.668±3.171 |
| 16 | 85.806±8.064 | 87.798±6.937 | 91.538±4.390 | 90.680±6.270 | 90.664±6.596 |
| 32 | 76.642±13.824 | 82.412±8.691 | 89.158±6.677 | 90.396±6.891 | 90.332±7.286 |

### `cnn2_c4_c8`
| L \ T | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 2 | 97.756±0.188 | 96.582±1.476 | 97.130±1.364 | 97.096±1.479 | 97.116±1.440 |
| 4 | 96.994±1.296 | 98.002±0.258 | 97.970±0.532 | 98.130±0.424 | 98.178±0.385 |
| 8 | 95.508±2.868 | 97.452±0.309 | 98.066±0.115 | 98.190±0.178 | 98.272±0.131 |
| 16 | 95.862±0.752 | 97.010±0.603 | 97.698±0.392 | 97.954±0.255 | 98.058±0.266 |
| 32 | 96.066±1.047 | 96.626±1.598 | 97.796±0.427 | 97.996±0.292 | 98.118±0.199 |

### `cnn2_c8_c16`
| L \ T | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 2 | 98.420±0.165 | 97.126±1.254 | 97.824±0.667 | 97.598±0.829 | 97.638±0.784 |
| 4 | 98.326±0.281 | 98.666±0.131 | 98.764±0.094 | 98.788±0.129 | 98.822±0.144 |
| 8 | 97.730±0.273 | 98.328±0.138 | 98.576±0.060 | 98.676±0.122 | 98.726±0.105 |
| 16 | 97.634±0.401 | 98.224±0.366 | 98.520±0.188 | 98.640±0.136 | 98.642±0.090 |
| 32 | 97.736±0.237 | 98.298±0.113 | 98.630±0.116 | 98.786±0.047 | 98.822±0.031 |

### `cnn2_c16_c32`
| L \ T | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 2 | 98.544±0.048 | 98.284±0.314 | 98.280±0.301 | 98.400±0.246 | 98.476±0.208 |
| 4 | 98.278±0.160 | 98.672±0.122 | 98.704±0.144 | 98.756±0.107 | 98.796±0.112 |
| 8 | 98.188±0.291 | 98.580±0.170 | 98.792±0.142 | 98.848±0.104 | 98.906±0.117 |
| 16 | 98.024±0.354 | 98.424±0.208 | 98.728±0.118 | 98.806±0.097 | 98.842±0.088 |
| 32 | 98.112±0.204 | 98.478±0.196 | 98.796±0.119 | 98.856±0.080 | 98.896±0.055 |

## 5) MNIST ANN（T=0）L=2/4/8/16/32 扫描（已完成）

- 结果文件：
  - `QCFS_simulation/noise3_exp/mnist_ann_T0_L_acc/mnist_ann_T0_L_acc_raw.csv`
  - `QCFS_simulation/noise3_exp/mnist_ann_T0_L_acc/mnist_ann_T0_L_acc_mean_std.csv`
  - `QCFS_simulation/noise3_exp/mnist_ann_T0_L_acc/mnist_ann_T0_L_acc_L2_vs_L16_summary.csv`
- 任务规模：10 个模型（FC3 6个 + CNN2 4个）× 5 个 L × 5 seeds = **250** 条记录（已完成）

- `cnn2` 最优：`cnn2_c16_c32` @ L=8, acc=99.088±0.049
- `fc3` 最优：`fc3_h128` @ L=32, acc=97.822±0.110

### 5.1) CNN2 全量结果（acc_mean±acc_std）

| arch | size | L=2 | L=4 | L=8 | L=16 | L=32 |
|---|---|---:|---:|---:|---:|---:|
| `cnn2_c2_c4` | `c2c4` | 96.408±0.413 | 97.530±0.067 | 97.884±0.191 | 97.822±0.139 | 97.956±0.136 |
| `cnn2_c4_c8` | `c4c8` | 98.166±0.146 | 98.670±0.159 | 98.740±0.061 | 98.776±0.071 | 98.798±0.058 |
| `cnn2_c8_c16` | `c8c16` | 98.732±0.039 | 98.914±0.030 | 99.020±0.067 | 99.058±0.068 | 99.040±0.052 |
| `cnn2_c16_c32` | `c16c32` | 98.886±0.053 | 99.018±0.079 | 99.088±0.049 | 99.080±0.042 | 99.050±0.037 |

### 5.2) FC3 全量结果（acc_mean±acc_std）

| arch | size | L=2 | L=4 | L=8 | L=16 | L=32 |
|---|---|---:|---:|---:|---:|---:|
| `fc3_h4` | `h4` | 55.286±30.208 | 71.916±12.233 | 60.166±32.424 | 67.164±15.607 | 78.334±5.770 |
| `fc3_h8` | `h8` | 86.376±6.634 | 88.278±4.613 | 85.772±6.603 | 77.424±18.499 | 84.262±7.941 |
| `fc3_h16` | `h16` | 94.440±0.442 | 94.372±1.827 | 94.188±1.184 | 93.966±1.361 | 94.390±1.080 |
| `fc3_h32` | `h32` | 96.480±0.242 | 97.028±0.230 | 97.296±0.227 | 97.168±0.292 | 97.214±0.174 |
| `fc3_h64` | `h64` | 97.262±0.125 | 97.752±0.087 | 97.664±0.172 | 97.726±0.114 | 97.746±0.115 |
| `fc3_h128` | `h128` | 97.342±0.037 | 97.690±0.080 | 97.818±0.081 | 97.812±0.105 | 97.822±0.110 |

### 5.3) L2 vs L16 对比（来自 summary）

| family | arch | L2(mean±std) | L16(mean±std) | Δ(L16-L2) |
|---|---|---:|---:|---:|
| `cnn2` | `cnn2_c16_c32` | 98.886±0.053 | 99.080±0.042 | 0.194 |
| `cnn2` | `cnn2_c2_c4` | 96.408±0.413 | 97.822±0.139 | 1.414 |
| `cnn2` | `cnn2_c4_c8` | 98.166±0.146 | 98.776±0.071 | 0.610 |
| `cnn2` | `cnn2_c8_c16` | 98.732±0.039 | 99.058±0.068 | 0.326 |
| `fc3` | `fc3_h128` | 97.342±0.037 | 97.812±0.105 | 0.470 |
| `fc3` | `fc3_h16` | 94.440±0.442 | 93.966±1.361 | -0.474 |
| `fc3` | `fc3_h32` | 96.480±0.242 | 97.168±0.292 | 0.688 |
| `fc3` | `fc3_h4` | 55.286±30.208 | 67.164±15.607 | 11.878 |
| `fc3` | `fc3_h64` | 97.262±0.125 | 97.726±0.114 | 0.464 |
| `fc3` | `fc3_h8` | 86.376±6.634 | 77.424±18.499 | -8.952 |

## 6) 其他已沉淀的历史/对照实验（已提取）

- `noise3_exp` 下检测到 **32** 个 `*summary*.csv` 文件。
- 下表是目前本地可直接读取到的代表性实验（含具体指标）：

| 实验文件 | 方法 | 指标1 | 指标2 | 备注 |
|---|---|---:|---:|---|
| `.../ablation_mne_l2_vs_weight_decay_l16/noise_sweep_sigma_0_1_T16_rate_uniform/summary_sigma_hit.csv` | weight_decay | clean_acc=91.230 | sigma_hit@90=0.040 | seed42 |
| 同上 | no_regularization | clean_acc=90.900 | sigma_hit@90=0.040 | seed42 |
| 同上 | mne_l2 | clean_acc=87.570 | sigma_hit@90=0.000 | clean 已<90 |
| `.../ablation_mne_l2_vs_weight_decay_l16_c4_c8/.../summary_sigma_hit.csv` | weight_decay | sigma_hit@90=0.46 | acc_at_hit=89.89 | |
| 同上 | no_regularization | sigma_hit@90=0.20 | acc_at_hit=89.89 | |
| 同上 | mne_l2 | sigma_hit@90=NA | acc_at_hit=NA | 未命中阈值 |
| `.../ablation_mne_l2_vs_weight_decay_l16_c16_c32/.../summary_sigma_hit.csv` | weight_decay | sigma_hit@90=0.94 | acc_at_hit=89.36 | |
| 同上 | mne_l2 | sigma_hit@90=NA | acc_at_hit=NA | |
| 同上 | no_regularization | sigma_hit@90=NA | acc_at_hit=NA | |
| `.../ablation_mne_l2_vs_weight_decay_l16_fc2_h8/.../noise_sweep_summary_h8_rate_uniform_T16_L16.csv` | mne_l2 | acc(s0)=80.190 | acc(s1)=79.130 | min=79.130@1.0 |
| 同上 | weight_decay | acc(s0)=82.010 | acc(s1)=71.570 | min=71.570@1.0 |
| 同上 | no_regularization | acc(s0)=70.580 | acc(s1)=32.560 | min=32.560@1.0 |
| `.../ablation_mne_l2_vs_weight_decay_l16_fc2_h16/.../noise_sweep_summary_h16_rate_uniform_T16_L16.csv` | mne_l2 | acc(s0)=92.760 | acc(s1)=92.150 | min=92.150@1.0 |
| 同上 | weight_decay | acc(s0)=94.240 | acc(s1)=92.480 | min=92.430@0.9 |
| 同上 | no_regularization | acc(s0)=93.650 | acc(s1)=71.670 | min=71.670@1.0 |

### 6.1) 本地无完整具体数值时，服务器查找位置

如果本地仅有图或部分 summary，完整结果请到服务器下列目录找 `noise_sweep_matrix_*.csv`、`noise_sweep_combined_L_T.csv`、`*_raw.csv`：

- 服务器项目根目录：`/home/595/sl9144/codes/snn_simulation/QCFS_simulation`
- 主要结果目录：`/home/595/sl9144/codes/snn_simulation/QCFS_simulation/noise3_exp`
- 常用查找命令（服务器上）：
  - `cd /home/595/sl9144/codes/snn_simulation/QCFS_simulation/noise3_exp`
  - `rg "summary|mean_std|raw" -g "*.csv"`
  - `rg "noise_sweep_matrix|noise_sweep_combined_L_T" -g "*.csv"`

## 7) 来自 Gadi 打包文件的补充结果（`all_results_from_gadi`）

- 数据来源：`all_results_from_gadi/_csv_collect_20260702_235428_csv_all_files.tar.gz`（共 1178 个 CSV）

### 7.1) CIFAR-100 VGG16 三路正则噪声扫描（L=16, T=16, rate_uniform）

| 方法 | sigma=0 acc(mean±std) | sigma=1 acc(mean±std) | Δ(sigma1-sigma0) |
|---|---:|---:|---:|
| L2 (`weight_decay`) | 62.272 ± 0.373 | 14.642 ± 2.989 | -47.630 |
| MNE L2 (`mne_l2`) | 59.620 ± 0.225 | 59.438 ± 0.228 | -0.182 |
| MNE L2 + WD (`mne_l2_wd`) | 63.940 ± 0.258 | 60.340 ± 0.384 | -3.600 |

来源文件：`noise3_exp/cifar100_vgg16_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16/cifar100_vgg16_strict_seed_three_regs_noise_sweep_mean_std.csv`

### 7.2) CIFAR-100 VGG16 组合实验（best test summary）

| label | reg_coeff | weight_decay | acc_sigma0 |
|---|---:|---:|---:|
| weight_decay | - | 5e-04 | 62.270 |
| mne_l2 rc=1e-4 | 1e-04 | 0 | 59.560 |
| mne_l2+wd rc=1e-4 wd=5e-4 | 1e-04 | 5e-04 | 50.400 |
| mne_l2+wd rc=1e-4 wd=1e-4 | 1e-04 | 1e-04 | 64.240 |
| mne_l2+wd rc=3e-5 wd=5e-4 | 3e-05 | 5e-04 | 62.190 |
| mne_l2+wd rc=1e-5 wd=5e-4 | 1e-05 | 5e-04 | 61.670 |

来源文件：`noise3_exp/cifar100_vgg16_mne_l2_wd_combo_noise_sweep_rate_uniform_L16_T16/cifar100_vgg16_mne_l2_wd_combo_best_test_summary.csv`

### 7.3) CIFAR-100 VGG16 `mne_l2` 系数扫描（best test summary）

粗扫（`...mne_reg_coeff_scan...`）：

| method | reg_coeff | acc_sigma0 |
|---|---:|---:|
| weight_decay | 5e-4(wd) | 62.270 |
| mne_l2:1em04 | 1e-04 | 59.560 |
| mne_l2:1em03 | 1e-03 | 58.330 |
| mne_l2:3em03 | 3e-03 | 39.830 |
| mne_l2:3em02 | 3e-02 | 8.750 |

细扫（`...mne_reg_coeff_fine_scan...`）：

| method | reg_coeff | acc_sigma0 |
|---|---:|---:|
| weight_decay | 5e-4(wd) | 62.270 |
| mne_l2:3em05 | 3e-05 | 59.520 |
| mne_l2:1em05 | 1e-05 | 59.090 |
| mne_l2:3em04 | 3e-04 | 59.620 |
| mne_l2:1em04 | 1e-04 | 59.560 |

### 7.4) ImageNet ResNet18 新一轮实验（MNE L2）

- 新数据文件：`imagenet_resnet18_mne_l2_wd_combo_noise_sweep_raw.csv`
- 你确认图 `imagenet_resnet18_mne_l2_wd_combo_noise_sweep.png` 对应 **MNE L2（rc=1e-4）**。
- 备注：当前 raw 的 `label` 列显示为 `weight_decay`，与图例不一致；此处按图与实验设定记为 `mne_l2`。

| method | sigma | acc |
|---|---:|---:|
| mne_l2 (rc=1e-4) | 0.0 | 55.338 |
| mne_l2 (rc=1e-4) | 0.1 | 36.262 |
| mne_l2 (rc=1e-4) | 0.2 | 11.362 |
| mne_l2 (rc=1e-4) | 0.3 | 0.462 |
| mne_l2 (rc=1e-4) | 0.4 | 0.126 |
| mne_l2 (rc=1e-4) | 0.5 | 0.100 |
| mne_l2 (rc=1e-4) | 0.6 | 0.098 |
| mne_l2 (rc=1e-4) | 0.7 | 0.096 |
| mne_l2 (rc=1e-4) | 0.8 | 0.096 |
| mne_l2 (rc=1e-4) | 0.9 | 0.100 |
| mne_l2 (rc=1e-4) | 1.0 | 0.100 |

- 图文件：`imagenet_resnet18_mne_l2_wd_combo_noise_sweep.png`
- 若要画 **MNE L2 vs L2 双折线**，仍需同一轮的 L2 曲线数据（`weight_decay` 的 raw/summary）。

### 7.5) 当前包内缺失/未打包项

- `mnist_ann_T0_L_acc` 在此 Gadi 包中未出现（`CSV_COUNT=1178` 中检索不到该路径）。
- 如需补到本汇总，请在服务器单独打包并下载：
  - 目标目录：`/home/595/sl9144/codes/snn_simulation/QCFS_simulation/noise3_exp/mnist_ann_T0_L_acc/`
  - 关键文件：`mnist_ann_T0_L_acc_raw.csv`（和后续生成的 `...mean_std.csv` / `...summary.csv`）

## 8) 完整数据覆盖核查（来自 Gadi 打包）

- 数据源：`all_results_from_gadi/_csv_collect_20260702_235428_csv_all_files.tar.gz`
- CSV 总数：**1178**
- `*summary*.csv`：**37**
- `*mean_std*.csv`：**12**
- `*raw*.csv`：**18**

### 8.1 已在正文展开为完整数据表的核心文件

| 文件 | 状态 |
|---|---|
| `cifar10_vgg16_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16/cifar10_vgg16_strict_seed_three_regs_noise_sweep_mean_std.csv` | 已展开 |
| `cifar100_vgg16_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16/cifar100_vgg16_strict_seed_three_regs_noise_sweep_mean_std.csv` | 已展开 |
| `fc3_wd_strict_seed_normal_L_T_acc/fc3_wd_strict_seed_normal_L_T_acc_mean_std.csv` | 已展开 |
| `cnn_wd_strict_seed_normal_L_T_acc/cnn_wd_strict_seed_normal_L_T_acc_mean_std.csv` | 已展开 |
| `ablation_mne_l2_vs_weight_decay_l16_fc3_h4_h8_h16_h32_h64_h128/strict_seed_train_rate_uniform_L16_T16/strict_seed_train_noise_sweep_fc3_h4_h8_h16_h32_h64_h128_mean_std.csv` | 已展开 |
| `cifar100_vgg16_mne_l2_wd_combo_noise_sweep_rate_uniform_L16_T16/cifar100_vgg16_mne_l2_wd_combo_best_test_summary.csv` | 已展开 |
| `cifar100_vgg16_mne_reg_coeff_scan_noise_sweep_rate_uniform_L16_T16/cifar100_vgg16_mne_reg_coeff_scan_best_test_summary.csv` | 已展开 |
| `cifar100_vgg16_mne_reg_coeff_fine_scan_noise_sweep_rate_uniform_L16_T16/cifar100_vgg16_mne_reg_coeff_fine_scan_best_test_summary.csv` | 已展开 |
| `imagenet_resnet18_mne_l2_wd_combo_noise_sweep_rate_uniform_L16_T16/imagenet_resnet18_mne_l2_wd_combo_best_test_summary.csv` | 已展开 |

### 8.2 全部 summary CSV 清单（完整）

| 文件 | 行数 | 列数 |
|---|---:|---:|
| `noise3_exp/ablation_conv_mne_l2_vs_baselines_mnist_c2_c4_l16_rerun_v2/noise_sweep_sigma_0_1_T16_rate_uniform_step0.1/summary_metrics.csv` | 4 | 5 |
| `noise3_exp/ablation_conv_mne_l2_vs_baselines_mnist_c4_c8_rerun_v1/noise_sweep_sigma_0_1_T16_rate_uniform_step0.1/summary_metrics.csv` | 4 | 5 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_cifar10_vgg16_l16_full100_mps/accuracy_summary_full100.csv` | 3 | 9 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_cifar10_vgg16_l16_quick3ep/accuracy_summary.csv` | 3 | 6 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_cifar10_vgg16_l16_quick3ep/t16_eval/accuracy_t16_summary.csv` | 3 | 5 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16/noise_sweep_sigma_0_1_T16/summary_sigma_hit.csv` | 3 | 4 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16/noise_sweep_sigma_0_1_T16_rate_uniform/summary_sigma_hit.csv` | 3 | 6 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16/noise_sweep_sigma_0_1_T16_rate_uniform_reproduce_plus_conv/summary_sigma_hit.csv` | 4 | 4 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16/noise_sweep_sigma_0_1_T16_rate_uniform_reproduce_plus_conv_no_detach/summary_sigma_hit_no_detach.csv` | 4 | 4 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16/train_summary.csv` | 3 | 5 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16_c16_c32/noise_sweep_sigma_0_1_T16_rate_uniform/summary_sigma_hit.csv` | 3 | 4 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16_c16_c32/noise_sweep_sigma_0_1_T16_rate_uniform_reproduce_plus_conv/summary_sigma_hit.csv` | 4 | 5 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16_c16_c32/noise_sweep_sigma_0_1_T16_rate_uniform_reproduce_plus_conv_no_detach/summary_sigma_hit_no_detach.csv` | 4 | 4 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16_c4_c8/noise_sweep_sigma_0_1_T16_rate_uniform/summary_sigma_hit.csv` | 3 | 4 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16_c4_c8/noise_sweep_sigma_0_1_T16_rate_uniform_reproduce_plus_conv/summary_sigma_hit.csv` | 4 | 4 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16_c4_c8/noise_sweep_sigma_0_1_T16_rate_uniform_reproduce_plus_conv_no_detach/summary_sigma_hit_no_detach.csv` | 4 | 4 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16_fc2/noise_sweep_sigma_0_1_T16_rate_uniform/summary_sigma_hit.csv` | 3 | 4 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16_fc2_h128/noise_sweep_sigma_0_1_T16_rate_uniform/summary_sigma_hit.csv` | 3 | 4 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16_fc2_h16/noise_sweep_sigma_0_1_T16_rate_uniform/noise_sweep_summary_h16_rate_uniform_T16_L16.csv` | 3 | 6 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16_fc2_h16_h32/normal_T16_L16_acc_summary.csv` | 6 | 10 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16_fc2_h16_h32/normal_T16_L16_acc_summary_final_rc5em02.csv` | 6 | 10 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16_fc2_h16_h32/normal_T16_L16_acc_summary_mne_rc5em02.csv` | 2 | 9 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16_fc2_h16_h32/normal_T16_L4_acc_summary_final_rc5em02.csv` | 6 | 10 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16_fc2_h512/noise_sweep_sigma_0_1_T16_rate_uniform/summary_sigma_hit.csv` | 3 | 4 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16_fc2_h64/noise_sweep_sigma_0_1_T16_rate_uniform/summary_sigma_hit.csv` | 3 | 4 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16_fc2_h64/noise_sweep_sigma_0_1_T16_rate_uniform_mne_rc_compare/noise_sweep_mne_rc_compare_summary.csv` | 15 | 2 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16_fc2_h8/noise_sweep_sigma_0_1_T16_rate_uniform/noise_sweep_summary_h8_rate_uniform_T16_L16.csv` | 3 | 6 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16_plus_conv_mne_from_old/noise_sweep_sigma_0_1_T16_rate_uniform/summary_sigma_hit.csv` | 4 | 4 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16_plus_conv_mne_rerun_seed42_v1/noise_sweep_sigma_0_1_T16_rate_uniform/summary_sigma_hit.csv` | 4 | 5 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16_plus_conv_mne_schemeA_seed42_rerun_v1/noise_sweep_sigma_0_1_T16_rate_uniform/summary_sigma_hit.csv` | 4 | 4 |
| `noise3_exp/ablation_mne_l2_vs_weight_decay_l16_seed123/train_summary.csv` | 3 | 6 |
| `noise3_exp/cifar100_vgg16_mne_l2_wd_combo_noise_sweep_rate_uniform_L16_T16/cifar100_vgg16_mne_l2_wd_combo_best_test_summary.csv` | 6 | 5 |
| `noise3_exp/cifar100_vgg16_mne_reg_coeff_fine_scan_noise_sweep_rate_uniform_L16_T16/cifar100_vgg16_mne_reg_coeff_fine_scan_best_test_summary.csv` | 5 | 5 |
| `noise3_exp/cifar100_vgg16_mne_reg_coeff_scan_noise_sweep_rate_uniform_L16_T16/cifar100_vgg16_mne_reg_coeff_scan_best_test_summary.csv` | 5 | 5 |
| `noise3_exp/cifar10_vgg16_mne_l2_wd_combo_noise_sweep_rate_uniform_L16_T16/cifar10_vgg16_mne_l2_wd_combo_best_test_summary.csv` | 3 | 5 |
| `noise3_exp/fc3_wd_strict_seed_normal_L_T_acc/fc3_wd_strict_seed_normal_L2_vs_L16_acc_summary.csv` | 30 | 8 |
| `noise3_exp/imagenet_resnet18_mne_l2_wd_combo_noise_sweep_rate_uniform_L16_T16/imagenet_resnet18_mne_l2_wd_combo_best_test_summary.csv` | 1 | 5 |

## 结论速览

- 在 CIFAR-10 VGG16 三路对比中，`mne_l2` 对噪声最稳（sigma=0 到 1 几乎不降）。
- 在 MNIST CNN2 三路对比中，`mne_l2` 在各宽度下都显著更抗噪；`weight_decay/no_regularization` 在 sigma=1 有明显下滑。
- FC3 的 L×T 扫描显示：中大模型（h32/h64/h128）在更高 L/T 下精度与稳定性更好，极小模型方差较大。
- 新增 ANN(T=0) 的 L 扫描正在运行，当前已接近完成，可在跑完后补一版最终表。
