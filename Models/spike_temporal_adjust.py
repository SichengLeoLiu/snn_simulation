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


def _build_uniform_table(T: int, device, dtype):
    """uniform_table[k, t]：k 个脉冲在 T 步上的 0/1 模板（与 layer.get_rate_table 中 base 部分一致）。"""
    tbl = torch.zeros(T + 1, T, device=device, dtype=dtype)
    for k in range(1, T + 1):
        interval = T / k
        indices = torch.floor(
            torch.arange(k, device=device, dtype=torch.float32) * interval
        ).long()
        indices = torch.clamp(indices, 0, T - 1)
        tbl[k, indices] = 1.0
    return tbl


def _get_uniform_table(T: int, device, dtype):
    key = (T, str(device), dtype)
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
