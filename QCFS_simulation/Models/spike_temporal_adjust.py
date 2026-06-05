"""
第一层 IF 之后、第一层 Conv 之前，对时间维脉冲序列的手动重排（仅 T>0 有效）。
"""
import torch
import torch.nn.functional as F

SPIKE_SCHEDULE_MODES = frozenset(
    {
        "normal",
        "uniform",
        "front_load",
        "back_load",
        "weight_sign_pos_front",
        "weight_sign_neg_front",
    }
)

_UNIFORM_TABLE_CACHE = {}


def _strict_uniform_spike_indices(k: int, T: int, device):
    """
    ``uniform`` 模板：先定脉冲个数 ``k``，再按「首尾占满 ``[0,T-1]``」的均匀间隔落点。

    1. 脉冲个数 ``k``（已 ``<= T``）：由调用方从 IF 输出数出的 ``K`` 决定。
    2. 间隔（连续索引上）：``k>=2`` 时 ``Δ = (T-1)/(k-1)``，即相邻脉冲在**时间步索引轴**上的平均间距。
    3. 第 ``i`` 个脉冲时刻：``t_i = round(i * Δ)``，再钳制到 ``[0,T-1]``。
    4. 若 ``round`` 导致重复时刻，按时间顺序用尚未占用的时间步补足，保证 **恰好 k 个互异时刻**。

    这样得到长度 ``T`` 的 0/1 模板，且 **非零步数严格为 k**。
    """
    k = int(k)
    if k <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    k = min(k, T)
    if k == 1:
        return torch.tensor([T // 2], device=device, dtype=torch.long).clamp(0, T - 1)

    # 平均间隔（单位：时间步索引）；首尾脉冲分别在第 0 与第 T-1 步附近
    delta = float(T - 1) / float(k - 1)
    raw = [int(round(i * delta)) for i in range(k)]
    raw = [max(0, min(t, T - 1)) for t in raw]

    seen = set()
    order = []
    for t in raw:
        if t not in seen:
            seen.add(t)
            order.append(t)
    for t in range(T):
        if len(order) >= k:
            break
        if t not in seen:
            seen.add(t)
            order.append(t)
    order = sorted(order)[:k]
    return torch.tensor(order, device=device, dtype=torch.long)


def _build_uniform_table(T: int, device, dtype):
    """
    ``uniform_table[k, t]``：第 k 行恰有 **k** 个 1，对应「k 个脉冲 + 均匀间隔 Δ=(T-1)/(k-1)」
    离散落点（见 ``_strict_uniform_spike_indices``）。
    """
    tbl = torch.zeros(T + 1, T, device=device, dtype=dtype)
    for k in range(1, T + 1):
        idx = _strict_uniform_spike_indices(k, T, device)
        tbl[k, idx] = 1.0
    return tbl


def _get_uniform_table(T: int, device, dtype):
    key = (T, str(device), dtype, "uniform_strict_k")
    if key not in _UNIFORM_TABLE_CACHE:
        _UNIFORM_TABLE_CACHE[key] = _build_uniform_table(T, device, dtype)
    return _UNIFORM_TABLE_CACHE[key]


def _tb_to_tbt(x_tb: torch.Tensor, T: int):
    """[T*B, C, H, W] -> [T, B, C, H, W]"""
    n = x_tb.shape[0]
    assert n % T == 0, "batch 维须为 T 的整数倍"
    B = n // T
    return x_tb.view(T, B, *x_tb.shape[1:])


def _tbt_to_tb(x_tbt: torch.Tensor):
    """[T, B, C, H, W] -> [T*B, C, H, W]"""
    T, B = x_tbt.shape[0], x_tbt.shape[1]
    return x_tbt.reshape(T * B, *x_tbt.shape[2:])


def temporal_rearrange_after_first_if(
    x_tb: torch.Tensor,
    T: int,
    mode: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    对第一层 IF 输出 [T*B, C, H, W] 在长度 T 上重排脉冲时刻。
    T=0 或非 SNN 路径不应调用；normal 原样返回。
    """
    if mode == "normal" or T <= 0:
        return x_tb

    x = _tb_to_tbt(x_tb, T)
    # 每个 (b,c,h,w) 上的脉冲个数（幅值超过 eps 的时间步数）
    K = (x > eps).sum(dim=0).long()
    K = K.clamp(0, T)
    th = x.amax(dim=0).clamp_min(0.0)

    if mode == "front_load":
        t_idx = torch.arange(T, device=x.device, dtype=torch.long).view(T, 1, 1, 1, 1)
        mask = t_idx < K.unsqueeze(0)
        y = mask.to(x.dtype) * th.unsqueeze(0)
        return _tbt_to_tb(y)

    if mode == "back_load":
        t_idx = torch.arange(T, device=x.device, dtype=torch.long).view(T, 1, 1, 1, 1)
        start = (T - K).unsqueeze(0).clamp(min=0)
        mask = t_idx >= start
        y = mask.to(x.dtype) * th.unsqueeze(0)
        return _tbt_to_tb(y)

    if mode == "uniform":
        tbl = _get_uniform_table(T, x.device, x.dtype)
        # K: [B,C,H,W] -> 取每点模板 [T]，再乘 th
        pat = tbl[K]  # [B,C,H,W,T]
        pat = pat.permute(4, 0, 1, 2, 3).contiguous()
        y = pat * th.unsqueeze(0)
        return _tbt_to_tb(y)

    raise ValueError("temporal_rearrange_after_first_if 不支持模式: %s（请用 first_conv_weight_sign）" % mode)


def first_conv_with_weight_sign_schedule(
    x_tb: torch.Tensor,
    T: int,
    conv1: torch.nn.Conv2d,
    mode: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    weight_sign_pos_front / weight_sign_neg_front：
    先得到 front_load / back_load 两套时间序列，再按 conv1 权重正负分别与 front/back 做卷积后相加。
    """
    if T <= 0:
        return conv1(x_tb)

    if mode not in ("weight_sign_pos_front", "weight_sign_neg_front"):
        raise ValueError(mode)

    x = _tb_to_tbt(x_tb, T)
    K = (x > eps).sum(dim=0).long().clamp(0, T)
    th = x.amax(dim=0).clamp_min(0.0)
    t_idx = torch.arange(T, device=x.device, dtype=torch.long).view(T, 1, 1, 1, 1)

    mask_front = t_idx < K.unsqueeze(0)
    y_front = mask_front.to(x.dtype) * th.unsqueeze(0)
    start = (T - K).unsqueeze(0).clamp(min=0)
    mask_back = t_idx >= start
    y_back = mask_back.to(x.dtype) * th.unsqueeze(0)

    front_tb = _tbt_to_tb(y_front)
    back_tb = _tbt_to_tb(y_back)

    W = conv1.weight
    b = conv1.bias
    stride, pad = conv1.stride, conv1.padding

    W_pos = W * (W > 0).to(W.dtype)
    W_neg = W * (W < 0).to(W.dtype)

    if mode == "weight_sign_pos_front":
        # 正权用 front，负权用 back
        y = F.conv2d(front_tb, W_pos, None, stride, pad) + F.conv2d(
            back_tb, W_neg, None, stride, pad
        )
    else:
        # weight_sign_neg_front：负权 front，正权 back
        y = F.conv2d(front_tb, W_neg, None, stride, pad) + F.conv2d(
            back_tb, W_pos, None, stride, pad
        )

    if b is not None:
        y = y + b.view(1, -1, 1, 1)
    return y


def first_conv1d_with_weight_sign_schedule(
    x_tb: torch.Tensor,
    T: int,
    conv1: torch.nn.Conv1d,
    mode: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    与 ``first_conv_with_weight_sign_schedule`` 相同语义，针对 ``nn.Conv1d`` / ``[N,C,L]``，
    避免在模型侧把 Conv1d 伪装成 Conv2d。
    """
    if T <= 0:
        return conv1(x_tb)

    if mode not in ("weight_sign_pos_front", "weight_sign_neg_front"):
        raise ValueError(mode)

    x = _tb_to_tbt(x_tb, T)
    K = (x > eps).sum(dim=0).long().clamp(0, T)
    th = x.amax(dim=0).clamp_min(0.0)
    t_idx = torch.arange(T, device=x.device, dtype=torch.long).view(T, 1, 1, 1)

    mask_front = t_idx < K.unsqueeze(0)
    y_front = mask_front.to(x.dtype) * th.unsqueeze(0)
    start = (T - K).unsqueeze(0).clamp(min=0)
    mask_back = t_idx >= start
    y_back = mask_back.to(x.dtype) * th.unsqueeze(0)

    front_tb = _tbt_to_tb(y_front)
    back_tb = _tbt_to_tb(y_back)

    W = conv1.weight
    b = conv1.bias
    s = conv1.stride
    p = conv1.padding
    stride = s[0] if isinstance(s, tuple) else s
    pad = p[0] if isinstance(p, tuple) else p

    W_pos = W * (W > 0).to(W.dtype)
    W_neg = W * (W < 0).to(W.dtype)

    if mode == "weight_sign_pos_front":
        y = F.conv1d(front_tb, W_pos, None, stride, pad) + F.conv1d(
            back_tb, W_neg, None, stride, pad
        )
    else:
        y = F.conv1d(front_tb, W_neg, None, stride, pad) + F.conv1d(
            back_tb, W_pos, None, stride, pad
        )

    if b is not None:
        y = y + b.view(1, -1, 1)
    return y


def first_linear_with_weight_sign_schedule(
    x_tb: torch.Tensor,
    T: int,
    linear: torch.nn.Linear,
    mode: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    与 ``first_conv1d_with_weight_sign_schedule`` 相同语义，针对 ``nn.Linear``。
    ``x_tb`` 为 ``[N, C, L]`` 且 ``C * L == linear.in_features``（diff1d 为 ``[N, 1, 2]``）。
    输出 ``[N, 1, linear.out_features]``。
    """

    def _apply_linear(x_3d, lin):
        b = x_3d.size(0)
        o = lin(x_3d.reshape(b, -1))
        return o.view(b, 1, lin.out_features)

    if T <= 0:
        return _apply_linear(x_tb, linear)

    if mode not in ("weight_sign_pos_front", "weight_sign_neg_front"):
        raise ValueError(mode)

    x_flat = x_tb.reshape(x_tb.size(0), -1)
    x = _tb_to_tbt(x_flat, T)
    K = (x > eps).sum(dim=0).long().clamp(0, T)
    th = x.amax(dim=0).clamp_min(0.0)
    t_idx = torch.arange(T, device=x.device, dtype=torch.long).view(T, 1, 1)

    mask_front = t_idx < K.unsqueeze(0)
    y_front = mask_front.to(x.dtype) * th.unsqueeze(0)
    start = (T - K).unsqueeze(0).clamp(min=0)
    mask_back = t_idx >= start
    y_back = mask_back.to(x.dtype) * th.unsqueeze(0)

    front_tb = _tbt_to_tb(y_front)
    back_tb = _tbt_to_tb(y_back)

    W = linear.weight
    b = linear.bias

    W_pos = W * (W > 0).to(W.dtype)
    W_neg = W * (W < 0).to(W.dtype)

    if mode == "weight_sign_pos_front":
        y = F.linear(front_tb, W_pos, None) + F.linear(back_tb, W_neg, None)
    else:
        y = F.linear(front_tb, W_neg, None) + F.linear(back_tb, W_pos, None)

    if b is not None:
        y = y + b.unsqueeze(0)
    return y.view(y.size(0), 1, linear.out_features)
