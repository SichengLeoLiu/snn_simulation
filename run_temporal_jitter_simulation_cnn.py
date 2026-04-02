"""
Temporal Jitter -> Magnitude Error simulation (CNN version)

Features:
- Two-layer CNN SNN / ANN-equivalent simulation on MNIST
- Supports reconstruction / classification tasks
- Input spike encoding modes:
  uniform / front_loaded / back_loaded / natural /
  weight_sign_pos_front / weight_sign_neg_front
- weight_sign_* uses synapse-level policy at conv1:
  each conv1 synapse (o,c,u,v) uses front/back by sign of W[o,c,u,v]
- 可视化 `run_viz` 会输出两张同布局大图：IF2 后（原 `*_feature_maps_*.png`）与 IF1 后（`*_if1_feature_maps_*.png`）
- 差分图 `RdBu_r`：diff=SNN−ANN；Matplotlib 下 RdBu_r(0)=蓝、RdBu_r(1)=红，且 vmin→0、vmax→1，故 **蓝=SNN<ANN（更弱）**，**红=SNN>ANN（更强）**（相对 ANN 归一化特征）

控制变量（推荐）：
- 仅用 --train_ann 训练一个 ANN（L 控制量化）；权重存入 checkpoint（from_ann=True, 含 T_ann_ref）。
- 之后多次运行可视化并只改 --T：SNN 用当前 T，ANN 对照始终用 x_mean ∝ 1/T_ann_ref（与训练一致），
  不因扫 T 而改变，唯一变量为 SNN 时间步。
- 第一层输入编码为有符号率编码 q∈[-T,T]、脉冲幅 ±thresh，与 Normalize 后可能为负的像素及有符号 ANN 输入一致。
- natural：离散脉冲，各步独立 Bernoulli(p=|q|/T)，无前后载/均匀等人为时序干预（标准率编码）。
- --input_codec if：第一层用独立 IF 从分步电流生成脉冲；uniform 与 direct 同为「t_i=round(i*T/|q|) 占位」；
  natural 为每步等电流、无额外时序干预（direct 的 natural 仍为 Bernoulli，二者不同）。
- **SNN 隐藏层**：`TwoLayerCNNSNN` 使用 `Models.layer.IF`（`--if_mode`，默认 `normal`）逐步积分发放。
- **ANN2SNN / QCFS 与 VGG 一致**：每层 IF 前设置 `ann_input = conv(·)`（**量化前**，与 `VGG._forward_with_ann` 相同）；ANN 支路用 `_ann_quantize` 递推，与 `TwoLayerCNNANN` 一致。
  默认前向传入 `ann_input_mean = images*scale/T_ann_ref`（与训练 ANN 输入一致）；加 `--no_qcfs_ann_mean` 则改为对脉冲序列时间平均（与 VGG 中 `x_reshaped.mean(0)` 类似）。
- **`--if_mode compare_ann`**：每层 IF 内可计算 `snn_ann_error`（发放率 vs ANN 量化值），见 `Models/layer.py`；加 `--print_if_ann_error` 在 viz 末打印 IF1/IF2 摘要。
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from Models.layer import IF, myfloor
from utils import resolve_torch_device


MNIST_NUM_CLASSES = 10

# 与分类可视化网格中的 SNN 前向方式一致（按模式评估准确率时使用）
CNN_CLASSIFICATION_SNN_MODES = (
    "uniform",
    "front_loaded",
    "back_loaded",
    "natural",
    "weight_sign_pos_front",
    "weight_sign_neg_front",
)


def _ann_quantize(h, thresh, L):
    h = h / thresh
    h = torch.clamp(h, 0, 1)
    h = myfloor(h * L + 0.5) / L
    return h * thresh


def _qcfs_ann_input_mean(images, scale, t_ann_ref, T):
    """与 ANN 训练输入一致：x_mean = images * scale / T_ann_ref（VGG 中 ann 支路起点）。"""
    tr = int(t_ann_ref) if t_ann_ref is not None else int(T)
    return images * scale / float(max(tr, 1))


def _ann_mean_kw(net, images, scale, t_ann_ref, T, qcfs_ann_mean):
    if qcfs_ann_mean and isinstance(net, TwoLayerCNNSNN):
        return {"ann_input_mean": _qcfs_ann_input_mean(images, scale, t_ann_ref, T)}
    return {}


@torch.no_grad()
def _print_compare_ann_errors_if_any(net, images, scale, thresh, t_ann_ref, T, input_codec, device):
    """在 if_mode=compare_ann 下跑一次 uniform 编码并打印 IF1/IF2 的 snn_ann_error。"""
    if getattr(net, "if_mode", "") != "compare_ann":
        return
    net.eval()
    b = min(8, images.shape[0])
    im = images[:b].to(device)
    x = _encode_first_layer_input(
        im, T, scale, thresh, "uniform", t_ann_ref, input_codec,
    )
    x_flat = to_flat_seq(x, T, b)
    am = _qcfs_ann_input_mean(im, scale, t_ann_ref, T)
    _ = net(x_flat, ann_input_mean=am)
    for tag, layer in (("IF1", net.if1), ("IF2", net.if2)):
        if hasattr(layer, "snn_ann_error") and layer.snn_ann_error:
            e = layer.snn_ann_error
            print(
                f"  [compare_ann {tag}] MSE={e['mse']:.6f} MAE={e['mae']:.6f} | "
                f"SNN>ANN {e.get('snn_greater_ratio', 0)*100:.1f}%  SNN<ANN {e.get('snn_less_ratio', 0)*100:.1f}%  "
                f"SNN=ANN {e.get('snn_equal_ratio', 0)*100:.1f}%"
            )


class TwoLayerCNNSNN(nn.Module):
    """Conv -> IF -> Conv -> IF。QCFS：在每次调用 IF 前写入 ann_input（Conv 后、量化前），对齐 VGG._forward_with_ann。"""
    def __init__(
        self,
        T,
        in_ch=1,
        hidden_ch=16,
        out_ch=1,
        thresh=1.0,
        L=8,
        task="reconstruction",
        if_mode="normal",
    ):
        super().__init__()
        self.T = T
        self.task = task
        self.if_mode = if_mode
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1)
        self.if1 = IF(T=T, L=L, thresh=thresh)
        self.if1.T, self.if1.L = T, L
        self.if1.mode = if_mode
        self.conv2 = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, padding=1)
        self.if2 = IF(T=T, L=L, thresh=thresh)
        self.if2.T, self.if2.L = T, L
        self.if2.mode = if_mode
        if task == "classification":
            self.cls_head = nn.Linear(out_ch, MNIST_NUM_CLASSES)

    def forward_features(self, x_flat, ann_input_mean=None):
        """
        SNN 支路：conv1(x_flat) → IF1 → conv2 → IF2。
        ANN 支路（与 VGG._forward_with_ann 一致）：x_ann 经 conv → 写入对应 IF 的 ann_input → _ann_quantize 再进下一层 conv。
        ann_input_mean: [B,C,H,W]，默认 None 时用 x_flat 的时间维平均作为 x_ann 起点。
        """
        T = self.T
        tb = x_flat.shape[0]
        B = tb // T
        x_seq = x_flat.view(T, B, *x_flat.shape[1:])
        if ann_input_mean is not None:
            x_ann = ann_input_mean
        else:
            x_ann = x_seq.mean(dim=0)

        th1, th2 = self.if1.thresh.data, self.if2.thresh.data
        L1, L2 = self.if1.L, self.if2.L

        x_ann_h1_pre = self.conv1(x_ann)
        self.if1.ann_input = x_ann_h1_pre.detach().clone()

        c1 = self.conv1(x_flat)
        s1 = self.if1(c1)

        x_ann_h1 = _ann_quantize(x_ann_h1_pre, th1, L1)
        x_ann_h2_pre = self.conv2(x_ann_h1)
        self.if2.ann_input = x_ann_h2_pre.detach().clone()

        s2 = self.conv2(s1)
        s2 = self.if2(s2)
        return s2

    def forward(self, x_flat, ann_input_mean=None):
        x = self.forward_features(x_flat, ann_input_mean=ann_input_mean)
        if self.task == "classification":
            # 与 ANN 一致：cls_head 只作用在时间平均后的特征图上（ANN 是单次 h2→GAP→Linear）
            # 旧实现「每步 GAP+Linear 再对 logits 求和」与训练目标不等价，会导致准确率崩溃
            T, tb = self.T, x.shape[0]
            b = tb // T
            x = x.view(T, b, *x.shape[1:]).mean(dim=0)
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
            x = self.cls_head(x)
        return x

    def forward_and_magnitude(self, x_flat, ann_input_mean=None):
        """
        跑一次 forward_features，从 IF1/IF2 的 spike_sequence 得到时间累计归一化幅度。
        返回 (unused, norm_if2, norm_if1)，其中 norm_* = sum_t(spike_t) / thresh，形状 [B,C,H,W]。
        norm_if1 为 IF1 后（第一层脉冲累计）；norm_if2 为 IF2 后（与旧版第三返回值一致）。
        """
        _ = self.forward_features(x_flat, ann_input_mean=ann_input_mean)
        norm1, norm2 = None, None
        if getattr(self.if1, "spike_sequence", None) is not None:
            mag1 = self.if1.spike_sequence.sum(dim=0)
            norm1 = mag1 / self.if1.thresh.data
        if getattr(self.if2, "spike_sequence", None) is not None:
            mag2 = self.if2.spike_sequence.sum(dim=0)
            norm2 = mag2 / self.if2.thresh.data
        return _, norm2, norm1


class TwoLayerCNNANN(nn.Module):
    """ANN-equivalent for CNN: Conv -> quant -> Conv -> quant (+ cls head)."""
    def __init__(self, in_ch=1, hidden_ch=16, out_ch=1, thresh=1.0, L=8, task="reconstruction"):
        super().__init__()
        self.thresh = thresh
        self.L = L
        self.task = task
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, padding=1)
        if task == "classification":
            self.cls_head = nn.Linear(out_ch, MNIST_NUM_CLASSES)

    def forward(self, x):
        h1 = _ann_quantize(self.conv1(x), self.thresh, self.L)
        h2 = _ann_quantize(self.conv2(h1), self.thresh, self.L)
        if self.task == "classification":
            logits = self.cls_head(F.adaptive_avg_pool2d(h2, 1).flatten(1))
            return logits, h2 / self.thresh
        return h2, h2 / self.thresh


def get_mnist_loaders(batch_size, data_dir=None):
    data_dir = data_dir or os.path.expanduser("~/datasets")
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=trans)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=trans)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def to_flat_seq(x_batch_T_C_H_W, T, B):
    return x_batch_T_C_H_W.permute(1, 0, 2, 3, 4).reshape(T * B, *x_batch_T_C_H_W.shape[2:])


def _spike_charge_units(images, scale, thresh, T, t_ann_ref):
    """
    有符号离散电荷 q ∈ [-T, T]（单位：「一次 ±thresh 脉冲」），满足
        q * thresh ≈ images * scale * T / T_ann_ref
    即与 ANN 的 x_mean = images*scale/T_ann_ref 在 T 步内总注入量一致。

    说明：MNIST 使用 Normalize 后像素大量为负。旧版仅 n∈[0,T] 会把负侧 clamp 成 0 脉冲，
    而 ANN 仍用有符号 x_mean，导致 SNN 信息严重丢失（测试准确率≈随机），ANN 仍可达 ~30%。
    """
    tr = int(t_ann_ref) if t_ann_ref is not None else int(T)
    tr = max(tr, 1)
    base = images * scale / thresh  # images * I_total
    return (base * T / float(tr)).round().clamp(-T, T).long()


def encode_spikes(images, T, scale, thresh, mode="uniform", conv1_weight=None, t_ann_ref=None):
    """
    images: [B, C, H, W]
    returns: [B, T, C, H, W]，离散取值于 {-thresh, 0, thresh}。

    natural（仅 direct）：离散脉冲；各步独立 Bernoulli，P=|q|/T，幅值 sign(q)*thresh。可复现：viz/eval 对 torch manual_seed。
    若用 --input_codec if，natural 为每步等电流、由 IF 自然发放，见 encode_if_input。

    t_ann_ref: 与 ANN 训练参考窗一致；扫不同 SNN 的 T 时应传入 ckpt 中的 T_ann_ref。
              None 等价于 T（与旧版兼容）。纯 SNN 训练可显式传 t_ann_ref=T 保持原脉冲预算。
    """
    B, C, H, W = images.shape
    device = images.device
    charge = _spike_charge_units(images, scale, thresh, T, t_ann_ref)
    sign_f = torch.sign(charge.to(torch.float32))  # [B,C,H,W] in {-1,0,1}

    t_idx = torch.arange(T, device=device).view(1, T, 1, 1, 1)
    pq = charge.clamp(min=0)
    nq = (-charge).clamp(min=0)
    pq_bt = pq.unsqueeze(1)
    nq_bt = nq.unsqueeze(1)

    if mode == "front_loaded":
        return (t_idx < pq_bt).float() * thresh + (t_idx < nq_bt).float() * (-thresh)
    if mode == "back_loaded":
        return (t_idx >= T - pq_bt).float() * thresh + (t_idx >= T - nq_bt).float() * (-thresh)
    if mode == "uniform":
        # 在 T 个时间槽上放 k 个脉冲：t_i = round(i*T/k)，i=0..k-1（例如 T=8,k=2 → 0 与 4）。
        # 旧版用 i*(T-1)/(k-1) 会把 k=2 钉在 0 与 T-1（0 与 7），时间跨度拉满两端。
        uniform_mask = torch.zeros(T + 1, T, device=device)
        for k in range(1, T + 1):
            idx = torch.round(torch.arange(k, device=device, dtype=torch.float32) * T / k).long().clamp(0, T - 1)
            uniform_mask[k, idx] = 1
        abs_c = charge.abs().clamp(0, T)
        pat = uniform_mask[abs_c, :].permute(0, 4, 1, 2, 3)  # [B,T,C,H,W]
        return pat * sign_f.unsqueeze(1) * thresh
    if mode == "natural":
        # 离散脉冲；各步独立 Bernoulli(p=|q|/T)，无前后载/均匀等人为时序干预（标准率编码）
        abs_c = charge.abs().float().clamp(0, T)
        p = (abs_c / float(T)).clamp(0.0, 1.0)  # [B,C,H,W]
        rand = torch.rand(B, T, C, H, W, device=device)
        mask = (rand < p.unsqueeze(1)).float()
        return mask * sign_f.unsqueeze(1) * thresh
    if mode == "weight_sign":
        if conv1_weight is None:
            raise ValueError("mode='weight_sign' requires conv1_weight")
        # conv1 weight: [out_ch, in_ch, k, k] -> aggregate by input channel
        agg = conv1_weight.to(device=device, dtype=torch.float32).sum(dim=(0, 2, 3))  # [C]
        use_front = (agg >= 0).view(1, 1, C, 1, 1).expand(B, T, C, H, W)
        abs_n = charge.abs().clamp(0, T)
        n_btchw = abs_n.unsqueeze(1)
        front_mask = (t_idx < n_btchw).float()
        back_mask = (t_idx >= T - n_btchw).float()
        sched = torch.where(use_front, front_mask, back_mask)
        return sched * sign_f.unsqueeze(1) * thresh

    raise ValueError(f"Unknown mode: {mode}")


def _temporal_drive_weights(mode, T, B, C, H, W, device, dtype=torch.float32, conv1_weight=None, charge=None):
    """
    IF 输入编码：各时间步相对电流 w（与 base=charge*thresh/T 相乘），满足 sum_t inp_t = charge*thresh。

    - uniform：与 encode_spikes 相同——|q|=k 时在 t_i=round(i*T/k) 占位（k=2,T=8→0,4），0/1 模板再归一化使 sum_t w=T；
      只在「应有脉冲」的时间步上有电流，**不是**每步同权全 1。
    - natural：**不添加任何时序干预**，w≡1（每步等强度 q*thresh/T），脉冲形态完全由 IF 积分-发放决定。
    """
    if T < 1:
        raise ValueError("T must be >= 1")

    def _norm_vec(vec):
        s = vec.sum()
        if float(s.item()) <= 0:
            return torch.ones_like(vec) * (T / max(T, 1))
        return vec / s * T

    if mode == "natural":
        return torch.ones(B, T, C, H, W, device=device, dtype=dtype)

    if mode == "uniform":
        if charge is None:
            raise ValueError("IF+uniform 需要 charge")
        abs_c = charge.abs().clamp(0, T).long()
        uniform_mask = torch.zeros(T + 1, T, device=device, dtype=dtype)
        for k in range(1, T + 1):
            idx = torch.round(torch.arange(k, device=device, dtype=dtype) * T / k).long().clamp(0, T - 1)
            uniform_mask[k, idx] = 1.0
        pat = uniform_mask[abs_c, :].permute(0, 4, 1, 2, 3)
        s = pat.sum(dim=1, keepdim=True)
        w = torch.where(s > 0, pat / s.clamp_min(1e-9) * T, torch.zeros_like(pat))
        return w

    if mode == "weight_sign":
        if conv1_weight is None:
            raise ValueError("IF+weight_sign 需要 conv1_weight")
        wf = _norm_vec(torch.arange(T, 0, -1, device=device, dtype=dtype))
        wb = _norm_vec(torch.arange(1, T + 1, device=device, dtype=dtype))
        agg = conv1_weight.to(device=device, dtype=dtype).sum(dim=(0, 2, 3))
        w_ct = torch.where(
            agg.view(C, 1) >= 0,
            wf.view(1, T).expand(C, T),
            wb.view(1, T).expand(C, T),
        )
        return w_ct.view(1, T, C, 1, 1).expand(B, T, C, H, W).contiguous()

    if mode == "front_loaded":
        w1 = _norm_vec(torch.arange(T, 0, -1, device=device, dtype=dtype))
    elif mode == "back_loaded":
        w1 = _norm_vec(torch.arange(1, T + 1, device=device, dtype=dtype))
    else:
        raise ValueError(
            f"IF 输入编码不支持 mode={mode!r}（请用 uniform/front_loaded/back_loaded/natural/weight_sign；"
            f"突触级 weight_sign_* 仍走手写前向）"
        )
    return w1.view(1, T, 1, 1, 1).expand(B, T, C, H, W).contiguous()


def encode_if_input(
    images,
    T,
    scale,
    thresh,
    mode,
    t_ann_ref=None,
    conv1_weight=None,
):
    """
    第一层用 IF 做「传感器编码」：先按 mode 分配各步电流（见 _temporal_drive_weights），再对每像素独立
    膜电位积分 → 过 ±thresh 发放 → 复位（单步最多各一发正/负脉冲，与主网络 IF 同构）。
    输出与 encode_spikes 相同形状 [B,T,C,H,W]，取值为 0 或 ±thresh。

    uniform：电流只加在「与 direct uniform 相同的均匀时间格点」上；natural：每步等电流，无额外时序整形。

    注意：膜电位初值必须为 **0**，不能与主网络 IF 一样用 0.5*thresh。
    后者会破坏有符号积分的对称性：Normalize 下大量 q<0 时，natural 等模式在 mem0=0.5*th 下
    往往只能产生少于 |q| 的负脉冲（总注入虽仍为 q*thresh，但与 direct 的 |q| 个 ±thresh 脉冲不等价），
    分类易跌至随机水平；mem0=0 时，在 sum_t(inp)=q*thresh 且每步 |inp| 不过大时，正负脉冲数与 |q| 对齐。
    """
    B, C, H, W = images.shape
    device = images.device
    dtype = torch.float32
    charge = _spike_charge_units(images, scale, thresh, T, t_ann_ref).to(dtype)
    th = torch.as_tensor(thresh, device=device, dtype=dtype)
    if th.numel() != 1:
        th = th.reshape(-1)[0]
    thv = th.item()

    w = _temporal_drive_weights(
        mode, T, B, C, H, W, device, dtype, conv1_weight=conv1_weight, charge=charge,
    )
    base = charge * thv / float(T)
    inp = base.unsqueeze(1) * w

    # 有符号传感器：初值 0（见函数说明）。勿用 0.5*thresh，否则负 q 下负脉冲数系统性偏少。
    mem = torch.zeros((B, C, H, W), device=device, dtype=dtype)
    outs = []
    for t in range(T):
        mem = mem + inp[:, t]
        sp = torch.zeros_like(mem)
        pos = mem >= thv
        sp_p = pos.to(dtype) * thv
        mem = mem - sp_p
        neg = mem <= -thv
        sp_n = neg.to(dtype) * (-thv)
        mem = mem - sp_n
        sp = sp_p + sp_n
        outs.append(sp)
    return torch.stack(outs, dim=1)


def _encode_first_layer_input(
    images,
    T,
    scale,
    thresh,
    mode,
    t_ann_ref,
    input_codec,
    conv1_weight=None,
):
    """direct：encode_spikes；if：encode_if_input（突触级模式勿调用本函数）。"""
    if input_codec == "if":
        return encode_if_input(
            images,
            T,
            scale,
            thresh,
            mode,
            t_ann_ref=t_ann_ref,
            conv1_weight=conv1_weight,
        )
    return encode_spikes(images, T, scale, thresh, mode, t_ann_ref=t_ann_ref, conv1_weight=conv1_weight)


def _feature_2d(x):
    """x: [B,C,H,W] -> [B,H,W] by channel mean."""
    return x.mean(axis=1)


def _forward_weight_sign_synapse_level(
    net, images, T, scale, thresh, sign_policy="pos_front", synapse_subset=1.0, t_ann_ref=None,
):
    """
    Synapse-level weight-sign forward at conv1.
    For each conv1 synapse W[o,c,u,v]:
      W>=0 -> front-loaded spikes for that synapse
      W<0  -> back-loaded spikes for that synapse
    After the manual weighted sum, adds net.conv1.bias per output channel (same as nn.Conv2d).
    t_ann_ref: 与 encode_spikes 一致，用于有符号 charge 时间预算。
    """
    device = images.device
    B, C, H, W = images.shape
    W1 = net.conv1.weight  # [O, C, k, k]
    O = W1.shape[0]
    k = W1.shape[2]
    pad = k // 2
    L = H * W
    W1_flat = W1.view(O, -1)  # [O, C*k*k]
    if sign_policy == "pos_front":
        use_front_sign = (W1_flat >= 0).view(1, O, -1, 1)  # [1,O,KK,1]
    elif sign_policy == "neg_front":
        use_front_sign = (W1_flat < 0).view(1, O, -1, 1)   # [1,O,KK,1]
    else:
        raise ValueError(f"Unknown sign_policy: {sign_policy}")
    if synapse_subset < 1.0:
        # only a subset uses sign-based schedule, the rest fallback to uniform(front-like here)
        total = O * W1_flat.shape[1]
        keep = max(1, int(total * max(0.0, synapse_subset)))
        flat_idx = torch.randperm(total, device=device)[:keep]
        subset_mask = torch.zeros(total, device=device, dtype=torch.bool)
        subset_mask[flat_idx] = True
        subset_mask = subset_mask.view(1, O, -1, 1)
    else:
        subset_mask = torch.ones(1, O, W1_flat.shape[1], 1, device=device, dtype=torch.bool)

    charge = _spike_charge_units(images, scale, thresh, T, t_ann_ref)
    abs_patch = F.unfold(charge.abs().float(), kernel_size=k, padding=pad, stride=1)
    sign_patch = F.unfold(torch.sign(charge.to(torch.float32)), kernel_size=k, padding=pad, stride=1)

    thre1 = net.if1.thresh.data
    thre2 = net.if2.thresh.data
    mem1 = 0.5 * thre1
    mem2 = 0.5 * thre2
    spike1_seq = []
    spike2_seq = []

    for t in range(T):
        front = (t < abs_patch).unsqueeze(1)            # [B,1,KK,L]
        back = (t >= (T - abs_patch)).unsqueeze(1)      # [B,1,KK,L]
        # subset_mask=True: use sign(front/back); False: use uniform-like front schedule as baseline
        sign_spike = torch.where(use_front_sign, front, back).float()
        base_sp = torch.where(subset_mask, sign_spike, front.float())
        syn_spike = base_sp * sign_patch.unsqueeze(1) * thresh  # [B,O,KK,L]，与有符号 encode 对齐
        curr1_flat = (syn_spike * W1_flat.view(1, O, -1, 1)).sum(dim=2)   # [B,O,L]
        curr1 = curr1_flat.view(B, O, H, W)
        # 与 nn.Conv2d 一致：手动乘加后须加上 conv1 偏置（否则与 uniform/ANN 不等价）
        if net.conv1.bias is not None:
            curr1 = curr1 + net.conv1.bias.view(1, O, 1, 1)

        mem1 = mem1 + curr1
        spike1 = (mem1 >= thre1).float() * thre1
        mem1 = mem1 - spike1
        spike1_seq.append(spike1)

        curr2 = net.conv2(spike1)
        mem2 = mem2 + curr2
        spike2 = (mem2 >= thre2).float() * thre2
        mem2 = mem2 - spike2
        spike2_seq.append(spike2)

    spike1_seq = torch.stack(spike1_seq, dim=0)  # [T,B,C1,H,W]
    spike2_seq = torch.stack(spike2_seq, dim=0)  # [T,B,C2,H,W]
    mag1 = spike1_seq.sum(dim=0)
    mag2 = spike2_seq.sum(dim=0)
    norm1 = mag1 / thre1
    norm2 = mag2 / thre2
    if net.task == "classification":
        feat_t = spike2_seq.mean(dim=0)
        logits_sum = net.cls_head(F.adaptive_avg_pool2d(feat_t, 1).flatten(1))
    else:
        logits_sum = None
    return norm2, logits_sum, norm1


def _select_vis_indices(num_show, gt, pred_list):
    gt = np.asarray(gt)
    preds = [np.asarray(p) for p in pred_list]
    correct_mat = np.stack([(p == gt) for p in preds], axis=0)
    diff_mask = np.any(correct_mat != correct_mat[0:1, :], axis=0)
    diff_idx = np.where(diff_mask)[0]
    if diff_idx.size == 0:
        pred_mat = np.stack(preds, axis=0)
        pred_diff_mask = np.any(pred_mat != pred_mat[0:1, :], axis=0)
        diff_idx = np.where(pred_diff_mask)[0]
        diff_mask = pred_diff_mask
    same_idx = np.where(~diff_mask)[0]
    return np.concatenate([diff_idx, same_idx])[:num_show]


@torch.no_grad()
def eval_snn_classification_one_mode(
    net,
    test_loader,
    T,
    scale,
    thresh,
    device,
    mode,
    max_batches=None,
    t_ann_ref=None,
    synapse_subset=1.0,
    natural_seed=0,
    input_codec="direct",
    qcfs_ann_mean=True,
):
    """
    某一种输入编码 / 前向方式的测试集分类准确率；mode 名称与可视化 `CNN_CLASSIFICATION_SNN_MODES` 一致。
    natural：若 natural_seed>=0，在整段评估开始前设 torch.manual_seed（各 batch 连续消耗 RNG）；natural_seed<0 不重置。
    input_codec：direct 或 if（第一层 IF 传感器编码，见 --input_codec）。
    qcfs_ann_mean：为 True 时传入 ann_input_mean（与 ANN x_mean 一致），供各 IF 写入 ann_input（VGG/QCFS）。
    """
    if mode == "natural" and natural_seed >= 0:
        torch.manual_seed(int(natural_seed))
    net.eval()
    correct, total = 0, 0
    for bi, (images, labels) in enumerate(test_loader):
        if max_batches is not None and bi >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        b = images.shape[0]
        if mode == "weight_sign_pos_front":
            _, logits, _ = _forward_weight_sign_synapse_level(
                net,
                images,
                T,
                scale,
                thresh,
                sign_policy="pos_front",
                synapse_subset=synapse_subset,
                t_ann_ref=t_ann_ref,
            )
        elif mode == "weight_sign_neg_front":
            _, logits, _ = _forward_weight_sign_synapse_level(
                net,
                images,
                T,
                scale,
                thresh,
                sign_policy="neg_front",
                synapse_subset=synapse_subset,
                t_ann_ref=t_ann_ref,
            )
        else:
            x = _encode_first_layer_input(
                images,
                T,
                scale,
                thresh,
                mode,
                t_ann_ref,
                input_codec,
                conv1_weight=net.conv1.weight if mode == "weight_sign" else None,
            )
            x_flat = to_flat_seq(x, T, b)
            if qcfs_ann_mean and isinstance(net, TwoLayerCNNSNN):
                am = _qcfs_ann_input_mean(images, scale, t_ann_ref, T)
                logits = net(x_flat, ann_input_mean=am)
            else:
                logits = net(x_flat)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += b
    return 100.0 * correct / max(total, 1), correct, total


@torch.no_grad()
def eval_snn_classification_accuracy(
    net,
    test_loader,
    T,
    scale,
    thresh,
    device,
    encoding_mode="uniform",
    max_batches=None,
    t_ann_ref=None,
    natural_seed=0,
    input_codec="direct",
    qcfs_ann_mean=True,
):
    """
    测试集分类准确率（仅 encode_spikes 类模式，默认 uniform）。
    与 ANN 一致：SNN forward 内部已对时间维特征取平均再分类。
    """
    return eval_snn_classification_one_mode(
        net,
        test_loader,
        T,
        scale,
        thresh,
        device,
        encoding_mode,
        max_batches=max_batches,
        t_ann_ref=t_ann_ref,
        synapse_subset=1.0,
        natural_seed=natural_seed,
        input_codec=input_codec,
        qcfs_ann_mean=qcfs_ann_mean,
    )


@torch.no_grad()
def eval_ann_classification_accuracy(
    snn_net, test_loader, scale, thresh, L, task, device, T_ann_ref, max_batches=None,
):
    """
    与可视化一致的 ANN：从当前 SNN 拷贝 conv/cls_head，输入 x_mean = images*scale/T_ann_ref。
    用于判断「≈10%」来自弱 ANN 还是 SNN 前向/转换问题。
    """
    if task != "classification":
        return None
    ann = TwoLayerCNNANN(
        in_ch=1,
        hidden_ch=snn_net.conv1.out_channels,
        out_ch=snn_net.conv2.out_channels,
        thresh=thresh,
        L=L,
        task=task,
    ).to(device)
    ann_sd = ann.state_dict()
    snn_sd = snn_net.state_dict()
    for k in ann_sd:
        if k in snn_sd and ("conv" in k or "cls_head" in k):
            ann_sd[k] = snn_sd[k].clone()
    ann.load_state_dict(ann_sd, strict=False)
    ann.eval()
    correct, total = 0, 0
    for bi, (images, labels) in enumerate(test_loader):
        if max_batches is not None and bi >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        x_mean = images * scale / float(max(int(T_ann_ref), 1))
        logits, _ = ann(x_mean)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += images.shape[0]
    return 100.0 * correct / max(total, 1), correct, total


@torch.no_grad()
def _validate_epoch(net, data_loader, args, device, scale):
    """
    在 data_loader 上评估（默认用 MNIST test_loader 作「验证」）。
    分类：返回 (acc_pct, None)；重建：返回 (None, avg_mse)。
    """
    was_training = net.training
    net.eval()
    task = args.task
    if task == "classification":
        correct, total = 0, 0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            b = images.shape[0]
            if args.train_ann:
                x_mean = images * scale / float(max(int(args.T_ann_ref), 1))
                out, _ = net(x_mean)
            else:
                x_flat = to_flat_seq(
                    _encode_first_layer_input(
                        images,
                        args.T,
                        scale,
                        args.thresh,
                        "uniform",
                        args.T_ann_ref,
                        args.input_codec,
                    ),
                    args.T,
                    b,
                )
                kw = _ann_mean_kw(
                    net, images, scale, args.T_ann_ref, args.T,
                    not getattr(args, "no_qcfs_ann_mean", False),
                )
                out = net(x_flat, **kw)
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += b
        acc = 100.0 * correct / max(total, 1)
        if was_training:
            net.train()
        return acc, None
    # reconstruction
    running = 0.0
    n_batches = 0
    for images, labels in data_loader:
        images = images.to(device)
        b = images.shape[0]
        if args.train_ann:
            x_mean = images * scale / float(max(int(args.T_ann_ref), 1))
            out, norm_map = net(x_mean)
        else:
            x_flat = to_flat_seq(
                _encode_first_layer_input(
                    images,
                    args.T,
                    scale,
                    args.thresh,
                    "uniform",
                    args.T_ann_ref,
                    args.input_codec,
                ),
                args.T,
                b,
            )
            kw = _ann_mean_kw(
                net, images, scale, args.T_ann_ref, args.T,
                not getattr(args, "no_qcfs_ann_mean", False),
            )
            _, norm_map, _ = net.forward_and_magnitude(x_flat, **kw)
        target = torch.clamp(images * scale / args.thresh, 0, 1)
        if norm_map.shape[1] != 1:
            target = target.repeat(1, norm_map.shape[1], 1, 1)
        loss = F.mse_loss(norm_map / args.T if not args.train_ann else norm_map, target)
        running += loss.item()
        n_batches += 1
    avg_mse = running / max(n_batches, 1)
    if was_training:
        net.train()
    return None, avg_mse


def train(args):
    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)
    train_loader, test_loader = get_mnist_loaders(args.batch_size, args.data_dir)
    out_ch = 1 if args.task == "reconstruction" else args.out_ch_cls
    scale = args.I_total * args.thresh

    if args.train_ann:
        net = TwoLayerCNNANN(in_ch=1, hidden_ch=args.hidden_ch, out_ch=out_ch, thresh=args.thresh, L=args.L, task=args.task).to(device)
    else:
        net = TwoLayerCNNSNN(
            T=args.T,
            in_ch=1,
            hidden_ch=args.hidden_ch,
            out_ch=out_ch,
            thresh=args.thresh,
            L=args.L,
            task=args.task,
            if_mode=args.if_mode,
        ).to(device)
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    net.train()
    for epoch in range(args.epochs):
        running = 0.0
        correct, total = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            if args.train_ann:
                # ANN 输入与 SNN 的 --T 解耦：用固定参考 T_ann_ref（写入 ckpt），便于之后只改 SNN 的 T 做对比
                x_mean = images * scale / float(max(int(args.T_ann_ref), 1))
                out, norm_map = net(x_mean)
            else:
                b = images.shape[0]
                x_flat = to_flat_seq(
                    _encode_first_layer_input(
                        images,
                        args.T,
                        scale,
                        args.thresh,
                        "uniform",
                        args.T_ann_ref,
                        args.input_codec,
                    ),
                    args.T,
                    b,
                )
                kw = _ann_mean_kw(
                    net, images, scale, args.T_ann_ref, args.T,
                    not getattr(args, "no_qcfs_ann_mean", False),
                )
                if args.task == "reconstruction":
                    _, norm_map, _ = net.forward_and_magnitude(x_flat, **kw)
                    out = None
                else:
                    out = net(x_flat, **kw)

            if args.task == "reconstruction":
                target = torch.clamp(images * scale / args.thresh, 0, 1)
                if norm_map.shape[1] != 1:
                    target = target.repeat(1, norm_map.shape[1], 1, 1)
                loss = F.mse_loss(norm_map / args.T if not args.train_ann else norm_map, target)
            else:
                loss = F.cross_entropy(out, labels)
                with torch.no_grad():
                    pred = out.argmax(dim=1)
                    correct += (pred == labels).sum().item()
                    total += labels.size(0)
            loss.backward()
            opt.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        msg = f"Epoch {epoch+1} avg loss = {running / len(train_loader):.4f}"
        if args.task == "classification":
            msg += f", train acc = {100.0 * correct / max(total,1):.2f}%"
        val_acc, val_mse = _validate_epoch(net, test_loader, args, device, scale)
        if args.task == "classification":
            msg += f", val(test) acc = {val_acc:.2f}%"
        else:
            msg += f", val(test) mse = {val_mse:.4f}"
        print(msg)

    if args.train_ann:
        print(f"ANN 训练: x_mean ∝ 1/{args.T_ann_ref} (T_ann_ref)；SNN 若另训则用 --T={args.T}，二者已解耦。")

    ckpt_path = os.path.join(args.out_dir, args.ckpt_name)
    state = {
        "state_dict": net.state_dict(),
        "from_ann": bool(args.train_ann),
        "task": args.task,
        "hidden_ch": args.hidden_ch,
        "out_ch": out_ch,
        "T": args.T,
        "T_ann_ref": args.T_ann_ref,
        "L": args.L,
        "thresh": args.thresh,
        "if_mode": args.if_mode,
    }
    torch.save(state, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")


def _save_cnn_feature_map_grid(
    map_2d,
    ann_2d,
    diff_2d,
    mode_order,
    suffix,
    vis_idx,
    vmin,
    vmax,
    L_signed,
    suptitle,
    out_path,
):
    """与 run_viz 相同布局：6 种 SNN 模式 + ANN + 6 行 diff。
    diff=SNN−ANN，RdBu_r：vmin(-)→蓝=SNN<ANN，vmax(+)→红=SNN>ANN。"""
    n_rows = 13
    num_show = len(vis_idx)
    fig, axes = plt.subplots(n_rows, num_show, figsize=(6 * num_show, 4.8 * n_rows))
    if num_show == 1:
        axes = axes.reshape(-1, 1)
    for col, s in enumerate(vis_idx):
        axes[0, col].imshow(map_2d["uniform"][s], cmap="hot", aspect="equal", vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f"Sample {s}: uniform" + suffix["uniform"][s])
        axes[0, col].axis("off")
        axes[1, col].imshow(map_2d["front_loaded"][s], cmap="hot", aspect="equal", vmin=vmin, vmax=vmax)
        axes[1, col].set_title(f"Sample {s}: front_loaded" + suffix["front_loaded"][s])
        axes[1, col].axis("off")
        axes[2, col].imshow(map_2d["back_loaded"][s], cmap="hot", aspect="equal", vmin=vmin, vmax=vmax)
        axes[2, col].set_title(f"Sample {s}: back_loaded" + suffix["back_loaded"][s])
        axes[2, col].axis("off")
        axes[3, col].imshow(map_2d["natural"][s], cmap="hot", aspect="equal", vmin=vmin, vmax=vmax)
        axes[3, col].set_title(f"Sample {s}: natural" + suffix["natural"][s])
        axes[3, col].axis("off")
        axes[4, col].imshow(map_2d["weight_sign_pos_front"][s], cmap="hot", aspect="equal", vmin=vmin, vmax=vmax)
        axes[4, col].set_title(f"Sample {s}: weight_sign_pos_front" + suffix["weight_sign_pos_front"][s])
        axes[4, col].axis("off")
        axes[5, col].imshow(map_2d["weight_sign_neg_front"][s], cmap="hot", aspect="equal", vmin=vmin, vmax=vmax)
        axes[5, col].set_title(f"Sample {s}: weight_sign_neg_front" + suffix["weight_sign_neg_front"][s])
        axes[5, col].axis("off")
        axes[6, col].imshow(ann_2d[s], cmap="hot", aspect="equal", vmin=vmin, vmax=vmax)
        axes[6, col].set_title(f"Sample {s}: ANN")
        axes[6, col].axis("off")
        for r, k in enumerate(mode_order, start=7):
            axes[r, col].imshow(diff_2d[k][s], cmap="RdBu_r", aspect="equal", vmin=-L_signed, vmax=L_signed)
            axes[r, col].set_title(f"Sample {s}: {k}−ANN")
            axes[r, col].axis("off")
    plt.suptitle(suptitle, fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def run_viz(args):
    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)
    B = max(args.batch_size, args.num_show)
    B = min(B, 64)
    _, test_loader = get_mnist_loaders(B, args.data_dir)
    scale = args.I_total * args.thresh

    ckpt_path = args.ckpt or os.path.join(args.out_dir, args.ckpt_name)
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        task = ckpt.get("task", args.task)
        hidden_ch = ckpt.get("hidden_ch", args.hidden_ch)
        out_ch = ckpt.get("out_ch", 1 if task == "reconstruction" else args.out_ch_cls)
        thresh = ckpt.get("thresh", args.thresh)
        L = ckpt.get("L", args.L)
        # ANN 对照：from_ann 的 ckpt 用保存的 T_ann_ref（旧 ckpt 无该键则用 T）；纯 SNN ckpt 用命令行 T_ann_ref
        if ckpt.get("from_ann", False):
            T_ann_ref = int(ckpt.get("T_ann_ref", ckpt.get("T", args.T_ann_ref)))
        else:
            T_ann_ref = int(args.T_ann_ref)
        if_m = ckpt.get("if_mode", args.if_mode)
        if ckpt.get("from_ann", False):
            net = TwoLayerCNNSNN(
                T=args.T,
                in_ch=1,
                hidden_ch=hidden_ch,
                out_ch=out_ch,
                thresh=thresh,
                L=L,
                task=task,
                if_mode=if_m,
            ).to(device)
            # copy conv/fc weights from ANN checkpoint
            sd = net.state_dict()
            for k, v in ckpt["state_dict"].items():
                if k in sd and ("conv" in k or "cls_head" in k):
                    sd[k] = v.clone()
            net.load_state_dict(sd, strict=False)
        else:
            net = TwoLayerCNNSNN(
                T=args.T,
                in_ch=1,
                hidden_ch=hidden_ch,
                out_ch=out_ch,
                thresh=thresh,
                L=L,
                task=task,
                if_mode=if_m,
            ).to(device)
            net.load_state_dict(ckpt["state_dict"])
    else:
        task = args.task
        out_ch = 1 if task == "reconstruction" else args.out_ch_cls
        net = TwoLayerCNNSNN(
            T=args.T,
            in_ch=1,
            hidden_ch=args.hidden_ch,
            out_ch=out_ch,
            thresh=args.thresh,
            L=args.L,
            task=task,
            if_mode=args.if_mode,
        ).to(device)
        thresh = args.thresh
        L = args.L
        T_ann_ref = int(args.T_ann_ref)
        print("No checkpoint found, using random init.")

    if not getattr(args, "no_qcfs_ann_mean", False):
        print(
            "  [diag] QCFS: ann_input_mean = images*scale/T_ann_ref（与 ANN / VGG 支路一致）；"
            "加 --no_qcfs_ann_mean 则 ann 起点改为脉冲序列时间平均。"
        )

    qcfs_eval = not getattr(args, "no_qcfs_ann_mean", False)

    if task == "classification" and not args.no_eval_snn_acc:
        _, test_loader_eval = get_mnist_loaders(args.eval_batch_size, args.data_dir)
        max_b = args.eval_max_batches if args.eval_max_batches > 0 else None
        # 若未出现此行，说明运行的不是当前仓库脚本（或未保存）
        print(
            f"  [diag] CNN SNN eval: IF1/IF2=Models.layer.IF(mode={getattr(net, 'if_mode', 'normal')}) "
            "逐步积分发放 → merge → … → time-mean(IF2) → GAP → cls_head"
        )
        mb_note = f", max_batches={args.eval_max_batches}" if args.eval_max_batches > 0 else ""
        ann_acc, ann_cor, ann_tot = eval_ann_classification_accuracy(
            net,
            test_loader_eval,
            scale,
            thresh,
            L,
            task,
            device,
            T_ann_ref,
            max_batches=max_b,
        )
        print(
            f"[ANN test acc]  x_mean∝1/{T_ann_ref} (rate, same weights): "
            f"{ann_acc:.2f}% ({ann_cor}/{ann_tot} samples{mb_note})"
        )
        acc_uniform = None
        if args.eval_each_mode:
            print(
                f"[SNN test acc by mode] T={args.T}, input_codec={args.input_codec}, "
                f"有符号 q≈round(I_tot·T/{T_ann_ref}) (|q|≤T), 脉冲∈{{±thresh,0}}{mb_note}"
            )
            for m in CNN_CLASSIFICATION_SNN_MODES:
                acc, n_cor, n_tot = eval_snn_classification_one_mode(
                    net,
                    test_loader_eval,
                    args.T,
                    scale,
                    thresh,
                    device,
                    m,
                    max_batches=max_b,
                    t_ann_ref=T_ann_ref,
                    synapse_subset=args.synapse_subset,
                    natural_seed=args.natural_seed,
                    input_codec=args.input_codec,
                    qcfs_ann_mean=qcfs_eval,
                )
                if m == "uniform":
                    acc_uniform = acc
                print(f"  {m}: {acc:.2f}% ({n_cor}/{n_tot})")
        else:
            acc_uniform, n_cor, n_tot = eval_snn_classification_accuracy(
                net,
                test_loader_eval,
                args.T,
                scale,
                thresh,
                device,
                encoding_mode="uniform",
                max_batches=max_b,
                t_ann_ref=T_ann_ref,
                natural_seed=args.natural_seed,
                input_codec=args.input_codec,
                qcfs_ann_mean=qcfs_eval,
            )
            print(
                f"[SNN test acc] T={args.T} input_codec={args.input_codec} uniform, "
                f"有符号 q≈round(I_tot·T/{T_ann_ref}) (|q|≤T), 脉冲∈{{±thresh,0}}: "
                f"{acc_uniform:.2f}% ({n_cor}/{n_tot} samples{mb_note})"
            )
            print("  （需要各模式分别统计时加 --eval_each_mode，约为 6 倍测试集耗时）")
        if ann_acc > acc_uniform + 5.0:
            print(
                "  → 提示: ANN 明显高于 SNN 时，重点查 IF/脉冲与率编码对齐、Normalize 下负像素是否用有符号编码；"
                "若两者都≈10%，多为训练不足或 checkpoint 非 from_ann。"
            )
        elif ann_acc < 15.0 and acc_uniform < 15.0:
            print(
                "  → 提示: ANN 与 SNN 均接近 10%（随机猜）：请加长 --epochs、检查是否用 --train_ann 训练并加载正确 ckpt。"
            )

    if task == "classification":
        if args.natural_seed >= 0:
            torch.manual_seed(int(args.natural_seed))
        mode_order_pre = list(CNN_CLASSIFICATION_SNN_MODES)
        if args.vis_sample_mode == "sequential":
            # 测试集顺序取前 num_show 个，跨 T 可视化同一批样本
            images, labels = next(iter(test_loader))
            images, labels = images.to(device), labels.to(device)
            n = min(args.num_show, images.shape[0])
            images, labels = images[:n], labels[:n]
            B = n
        else:
            images_all, labels_all = [], []
            max_batches = args.search_batches if args.search_batches > 0 else 10**9
            net.eval()
            with torch.no_grad():
                for bi, (images, labels) in enumerate(test_loader):
                    if bi >= max_batches or len(images_all) >= args.num_show:
                        break
                    images, labels = images.to(device), labels.to(device)
                    b = images.shape[0]
                    modes = {}
                    for m in mode_order_pre:
                        if m == "weight_sign_pos_front":
                            _, logits, _ = _forward_weight_sign_synapse_level(
                                net,
                                images,
                                args.T,
                                scale,
                                thresh,
                                sign_policy="pos_front",
                                synapse_subset=args.synapse_subset,
                                t_ann_ref=T_ann_ref,
                            )
                        elif m == "weight_sign_neg_front":
                            _, logits, _ = _forward_weight_sign_synapse_level(
                                net,
                                images,
                                args.T,
                                scale,
                                thresh,
                                sign_policy="neg_front",
                                synapse_subset=args.synapse_subset,
                                t_ann_ref=T_ann_ref,
                            )
                        else:
                            x = _encode_first_layer_input(
                                images,
                                args.T,
                                scale,
                                thresh,
                                m,
                                T_ann_ref,
                                args.input_codec,
                                conv1_weight=net.conv1.weight if m == "weight_sign" else None,
                            )
                            x_flat = to_flat_seq(x, args.T, b)
                            kw = _ann_mean_kw(
                                net, images, scale, T_ann_ref, args.T, qcfs_eval,
                            )
                            logits = net(x_flat, **kw)
                        modes[m] = logits.argmax(dim=1).cpu().numpy()

                    idx = _select_vis_indices(
                        b,
                        labels.cpu().numpy(),
                        [modes[k] for k in mode_order_pre],
                    )
                    for s in idx:
                        images_all.append(images[s].detach().cpu())
                        labels_all.append(labels[s].detach().cpu())
                        if len(images_all) >= args.num_show:
                            break
            images = torch.stack(images_all, dim=0).to(device)
            labels = torch.stack(labels_all, dim=0).to(device)
            B = images.shape[0]
    else:
        if args.natural_seed >= 0:
            torch.manual_seed(int(args.natural_seed))
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.to(device)
        B = images.shape[0]

    mode_order = [
        "uniform", "front_loaded", "back_loaded", "natural",
        "weight_sign_pos_front", "weight_sign_neg_front"
    ]
    feats = {}
    feats_if1 = {}
    preds = {}
    net.eval()
    with torch.no_grad():
        for m in mode_order:
            if m == "weight_sign_pos_front":
                norm, logits_ws, norm_if1_ws = _forward_weight_sign_synapse_level(
                    net,
                    images,
                    args.T,
                    scale,
                    thresh,
                    sign_policy="pos_front",
                    synapse_subset=args.synapse_subset,
                    t_ann_ref=T_ann_ref,
                )
                feats[m] = (norm / args.T).cpu().numpy()  # IF2 后 [B,C,H,W]
                feats_if1[m] = (norm_if1_ws / args.T).cpu().numpy()
                if task == "classification":
                    preds[m] = logits_ws.argmax(dim=1).cpu().numpy()
            elif m == "weight_sign_neg_front":
                norm, logits_ws, norm_if1_ws = _forward_weight_sign_synapse_level(
                    net,
                    images,
                    args.T,
                    scale,
                    thresh,
                    sign_policy="neg_front",
                    synapse_subset=args.synapse_subset,
                    t_ann_ref=T_ann_ref,
                )
                feats[m] = (norm / args.T).cpu().numpy()
                feats_if1[m] = (norm_if1_ws / args.T).cpu().numpy()
                if task == "classification":
                    preds[m] = logits_ws.argmax(dim=1).cpu().numpy()
            else:
                x = _encode_first_layer_input(
                    images,
                    args.T,
                    scale,
                    thresh,
                    m,
                    T_ann_ref,
                    args.input_codec,
                    conv1_weight=net.conv1.weight if m == "weight_sign" else None,
                )
                x_flat = to_flat_seq(x, args.T, B)
                kw = _ann_mean_kw(net, images, scale, T_ann_ref, args.T, qcfs_eval)
                _, norm_if2, norm_if1 = net.forward_and_magnitude(x_flat, **kw)
                feats[m] = (norm_if2 / args.T).cpu().numpy()
                feats_if1[m] = (norm_if1 / args.T).cpu().numpy()
                if task == "classification":
                    logits = net(x_flat, **kw)
                    preds[m] = logits.argmax(dim=1).cpu().numpy()

        # ANN 对照：x_mean 用 T_ann_ref（与 --train_ann 一致），与当前 SNN 的 args.T 解耦
        x_mean = images * scale / float(max(int(T_ann_ref), 1))
        ann = TwoLayerCNNANN(in_ch=1, hidden_ch=net.conv1.out_channels, out_ch=net.conv2.out_channels, thresh=thresh, L=L, task=task).to(device)
        ann_sd = ann.state_dict()
        net_sd = net.state_dict()
        for k in ann_sd:
            if k in net_sd and ("conv" in k or "cls_head" in k):
                ann_sd[k] = net_sd[k]
        ann.load_state_dict(ann_sd, strict=False)
        _, ann_norm = ann(x_mean)
        ann_norm_np = ann_norm.cpu().numpy()
        # IF1 对照：ANN 第一层量化后 h1 / thresh（与 SNN 的 IF1 累计脉冲/thresh 同一量级）
        h1_ann = _ann_quantize(ann.conv1(x_mean), thresh, L)
        ann_norm_if1_np = (h1_ann / thresh).cpu().numpy()

    print(
        f"  [viz] SNN T={args.T}, input_codec={args.input_codec}, ANN 对照 x_mean ∝ 1/{T_ann_ref} (T_ann_ref), "
        f"输入有符号 charge q≈round(I_tot·T/{T_ann_ref}) (|q|≤T), vis_sample_mode={args.vis_sample_mode}"
    )

    # convert to 2D map by channel mean（IF2 与 IF1 各一套）
    map_2d = {k: _feature_2d(v) for k, v in feats.items()}
    map_if1_2d = {k: _feature_2d(v) for k, v in feats_if1.items()}
    ann_2d = _feature_2d(ann_norm_np)
    ann_if1_2d = _feature_2d(ann_norm_if1_np)
    diff_2d = {k: map_2d[k] - ann_2d for k in mode_order}
    diff_if1_2d = {k: map_if1_2d[k] - ann_if1_2d for k in mode_order}

    gt = labels.cpu().numpy()
    suffix = {k: [""] * B for k in mode_order}
    if task == "classification":
        for k in mode_order:
            p = preds[k]
            suffix[k] = [f" | g={int(gt[i])} p={int(p[i])} {'OK' if p[i] == gt[i] else 'ERR'}" for i in range(B)]

    vis_idx = np.arange(min(args.num_show, B), dtype=np.int64)
    if task == "classification" and args.vis_sample_mode == "diverse":
        vis_idx = _select_vis_indices(args.num_show, gt, [preds[k] for k in mode_order])

    diff_abs_max_data = max([np.abs(diff_2d[k]).max() for k in mode_order] + [0.01])
    vmin_data = min([map_2d[k].min() for k in mode_order] + [ann_2d.min()])
    vmax_data = max([map_2d[k].max() for k in mode_order] + [ann_2d.max(), 1e-6])
    if getattr(args, "viz_diff_abs_max", None) is not None:
        L_signed = max(float(args.viz_diff_abs_max), 1e-6)
    else:
        L_signed = diff_abs_max_data
    fvmin, fvmax = getattr(args, "viz_feat_vmin", None), getattr(args, "viz_feat_vmax", None)
    if fvmin is not None and fvmax is not None:
        vmin, vmax = float(fvmin), float(fvmax)
    else:
        vmin, vmax = vmin_data, vmax_data
    print(
        f"  [viz] batch |diff|_max={diff_abs_max_data:.4f}; diff colormap ±{L_signed:.4f} "
        f"({'fixed --viz_diff_abs_max' if args.viz_diff_abs_max is not None else 'dynamic'})"
    )
    print(
        f"  [viz] feature hot [{vmin:.4f}, {vmax:.4f}] "
        f"({'fixed feat vmin/vmax' if fvmin is not None and fvmax is not None else 'dynamic'})"
    )

    diff_abs_max_if1 = max([np.abs(diff_if1_2d[k]).max() for k in mode_order] + [0.01])
    vmin_if1 = min([map_if1_2d[k].min() for k in mode_order] + [ann_if1_2d.min()])
    vmax_if1 = max([map_if1_2d[k].max() for k in mode_order] + [ann_if1_2d.max(), 1e-6])
    if args.viz_diff_abs_max is not None:
        L_if1 = max(float(args.viz_diff_abs_max), 1e-6)
    else:
        L_if1 = diff_abs_max_if1
    if fvmin is not None and fvmax is not None:
        vmin1, vmax1 = float(fvmin), float(fvmax)
    else:
        vmin1, vmax1 = vmin_if1, vmax_if1
    print(
        f"  [viz] IF1 layer: hot [{vmin1:.4f}, {vmax1:.4f}], |diff|_max={diff_abs_max_if1:.4f}, diff colormap ±{L_if1:.4f}"
    )

    out_if2 = os.path.join(args.out_dir, f"mnist_cnn_two_layer_feature_maps_T{args.T}_L{args.L}_{task}.png")
    _save_cnn_feature_map_grid(
        map_2d,
        ann_2d,
        diff_2d,
        mode_order,
        suffix,
        vis_idx,
        vmin,
        vmax,
        L_signed,
        suptitle=(
            f"CNN two-layer ({task}): IF2 后 — feature maps + diff vs ANN(h2) (diff ±{L_signed:.4f}"
            f"{', fixed' if args.viz_diff_abs_max is not None else ', dynamic'})"
        ),
        out_path=out_if2,
    )

    out_if1 = os.path.join(args.out_dir, f"mnist_cnn_two_layer_if1_feature_maps_T{args.T}_L{args.L}_{task}.png")
    _save_cnn_feature_map_grid(
        map_if1_2d,
        ann_if1_2d,
        diff_if1_2d,
        mode_order,
        suffix,
        vis_idx,
        vmin1,
        vmax1,
        L_if1,
        suptitle=(
            f"CNN two-layer ({task}): IF1 后 — feature maps + diff vs ANN(h1) (diff ±{L_if1:.4f}"
            f"{', fixed' if args.viz_diff_abs_max is not None else ', dynamic'})"
        ),
        out_path=out_if1,
    )

    if getattr(args, "print_if_ann_error", False):
        _print_compare_ann_errors_if_any(
            net, images, scale, thresh, T_ann_ref, args.T, args.input_codec, device,
        )


def main():
    parser = argparse.ArgumentParser(description="Temporal Jitter simulation (CNN)")
    parser.add_argument("--T", type=int, default=8, help="SNN 仿真时间步长（可扫不同 T 做对比）")
    parser.add_argument(
        "--T_ann_ref",
        type=int,
        default=8,
        help="ANN 训练与可视化对照的参考时间步：x_mean=images*I_total*thresh/T_ann_ref。"
             "仅 --train_ann 时参与训练；from_ann 的 ckpt 会覆盖为保存值。与 --T 解耦以实现控制变量。",
    )
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    parser.add_argument("--thresh", type=float, default=1.0)
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--I_total", type=float, default=2.5)
    parser.add_argument("--hidden_ch", type=int, default=64)
    parser.add_argument("--out_ch_cls", type=int, default=32, help="conv2 out channels for classification")
    parser.add_argument("--task", type=str, default="reconstruction", choices=["reconstruction", "classification"])
    parser.add_argument("--out_dir", type=str, default="./temporal_jitter_simulation")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train_ann", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_show", type=int, default=6)
    parser.add_argument("--search_batches", type=int, default=1, help="classification sample search batches; 0 means full set")
    parser.add_argument(
        "--vis_sample_mode",
        type=str,
        default="diverse",
        choices=["diverse", "sequential"],
        help="diverse: 按各编码预测差异挑选样本（默认）；sequential: 测试集顺序前 num_show 个，便于跨 T 固定同一样本",
    )
    parser.add_argument("--synapse_subset", type=float, default=1.0,
                        help="weight_sign中使用synapse-level符号调制的连接比例，范围(0,1]；1.0表示全部连接")
    parser.add_argument("--ckpt_name", type=str, default="mnist_cnn_two_layer.pth")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument(
        "--viz_diff_abs_max",
        type=float,
        default=None,
        help="差分图 RdBu_r 对称色标 ±该值（固定，便于不同 T 跨图比较）；不设则按本 batch 动态",
    )
    parser.add_argument(
        "--viz_feat_vmin",
        type=float,
        default=None,
        help="特征图 hot 下界（需与 --viz_feat_vmax 同时设置）",
    )
    parser.add_argument(
        "--viz_feat_vmax",
        type=float,
        default=None,
        help="特征图 hot 上界（例如与 vmin=0,vmax=1 对应归一化特征理论范围）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="auto: 优先 CUDA，其次 Apple MPS，否则 CPU；mps 需 Apple Silicon + 较新 PyTorch",
    )
    parser.add_argument(
        "--no_eval_snn_acc",
        action="store_true",
        help="分类任务下默认会在 viz 前用测试集评估 SNN(uniform)；加此 flag 可跳过",
    )
    parser.add_argument(
        "--eval_each_mode",
        action="store_true",
        help="分类任务：对 uniform/front_loaded/back_loaded/natural/weight_sign_pos_front/weight_sign_neg_front "
             "各跑一遍测试集并分别打印准确率（比仅 uniform 慢约 6 倍）",
    )
    parser.add_argument(
        "--natural_seed",
        type=int,
        default=0,
        help="natural 伯努利编码：>=0 时在 viz/eval(natural) 入口设 torch.manual_seed 以复现；<0 不固定",
    )
    parser.add_argument(
        "--input_codec",
        type=str,
        default="direct",
        choices=["direct", "if"],
        help="第一层：direct=encode_spikes；if=各步电流×IF 传感器（uniform=与 direct 相同的均匀时间格；"
             "natural=每步等电流无干预；突触级 weight_sign_* 仍手写前向）",
    )
    parser.add_argument(
        "--if_mode",
        type=str,
        default="normal",
        help="TwoLayerCNNSNN 的 IF1/IF2.mode，与 Models/layer.py 中 IF.forward 分支一致（如 normal、multi_spike 等）。",
    )
    parser.add_argument(
        "--no_qcfs_ann_mean",
        action="store_true",
        help="不传 ann_input_mean：ANN 支路起点改为脉冲序列时间平均（关闭与 VGG 一致的 x_mean=images*scale/T_ann_ref）。",
    )
    parser.add_argument(
        "--print_if_ann_error",
        action="store_true",
        help="若 --if_mode 为 compare_ann，viz 结束打印 IF1/IF2 的 snn_ann_error 摘要（需先做一次 uniform 前向）。",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=512,
        help="评估 SNN 测试准确率时 DataLoader 的 batch_size",
    )
    parser.add_argument(
        "--eval_max_batches",
        type=int,
        default=0,
        help="0=完整测试集；>0 只跑前若干个 batch（快速估计）",
    )
    args = parser.parse_args()

    args.device = resolve_torch_device(args.device)
    print("Using device:", args.device)

    if args.train:
        train(args)
    run_viz(args)


if __name__ == "__main__":
    main()

