"""
diff1d 玩具回归：全程 ``[N, C, L]``（默认 L=2），无假高度维。

``T>0`` 时先将 batch 沿时间重复为 ``[T*N, C, L]``；仅在调用 ``spike_temporal_adjust`` 时临时
``unsqueeze(2)`` 得到 ``[N, C, 1, L]`` 以兼容现有 4D 实现，返回后再 ``squeeze(2)``。

``input_if`` → ``linear1`` → ``if1`` → ``linear2``（无 bias，写死为 ``x1-x2``；数据上 ``x1>x2`` 时 ``y=x1-x2``∈``[0,1]`` 与输出直接对齐）。
"""
import torch
import torch.nn as nn

from Models.layer import IF, MergeTemporalDim, ExpandTemporalDim
from Models.spike_temporal_adjust import (
    SPIKE_SCHEDULE_MODES,
    first_linear_with_weight_sign_schedule,
    temporal_rearrange_after_first_if,
)


def _repeat_time_merge(x, T):
    """``[B, C, L]`` → ``[T*B, C, L]``（与 ``add_dimention``+``merge`` 对 4D 输入的语义一致）。"""
    return (
        x.unsqueeze(0)
        .expand(T, -1, -1, -1)
        .reshape(T * x.size(0), *x.shape[1:])
    )


def _apply_linear_1x2(x, linear):
    """``[B, 1, 2]`` + ``Linear(2, out)`` → ``[B, 1, out]``。"""
    b = x.size(0)
    o = linear(x.reshape(b, -1))
    return o.view(b, 1, linear.out_features)


class ToyDiff1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = 0
        self.merge = MergeTemporalDim(0)
        self.expand = ExpandTemporalDim(0)
        self.spike_schedule = "normal"

        self.input_if = IF()
        self.linear1 = nn.Linear(2, 2, bias=False)
        self.if1 = IF()
        self.linear2 = nn.Linear(2, 1, bias=False)

        with torch.no_grad():
            self.linear1.weight.copy_(
                torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float32)
            )
            self.linear2.weight.copy_(torch.tensor([[0.5, 0.5]], dtype=torch.float32))
        self.linear1.weight.requires_grad_(False)
        self.linear2.weight.requires_grad_(False)

    def set_spike_schedule(self, mode: str):
        if mode not in SPIKE_SCHEDULE_MODES:
            raise ValueError(
                "spike_schedule 须为 %s，收到: %s"
                % (sorted(SPIKE_SCHEDULE_MODES), mode)
            )
        self.spike_schedule = mode

    def set_T(self, T):
        self.T = T
        for module in self.modules():
            if isinstance(module, (IF, ExpandTemporalDim)):
                module.T = T
                if T > 0:
                    module.spike_counts = [0] * T
                    if hasattr(module, "total_elements"):
                        module.total_elements = [0] * T

    def set_L(self, L):
        for module in self.modules():
            if isinstance(module, IF):
                module.L = L

    def set_scaling_factor(self, scaling_factor=1.0):
        for module in self.modules():
            if isinstance(module, IF):
                module.scaling_factor = scaling_factor

    def set_mode(self, mode="normal"):
        for module in self.modules():
            if isinstance(module, IF):
                module.mode = mode

    def forward(self, x):
        # 兼容旧 checkpoint / 仍传 [B,1,1,2] 的情况
        if x.dim() == 4:
            x = x.squeeze(2)

        if self.T > 0:
            x = _repeat_time_merge(x, self.T)

        x = self.input_if(x)

        if self.T > 0:
            sch = self.spike_schedule
            if sch in ("weight_sign_pos_front", "weight_sign_neg_front"):
                x = first_linear_with_weight_sign_schedule(
                    x, self.T, self.linear1, sch
                )
            else:
                x4 = temporal_rearrange_after_first_if(x.unsqueeze(2), self.T, sch)
                x = _apply_linear_1x2(x4.squeeze(2), self.linear1)
        else:
            x = _apply_linear_1x2(x, self.linear1)

        x = self.if1(x)
        x = _apply_linear_1x2(x, self.linear2)
        x = torch.flatten(x, 1)
        if self.T > 0:
            x = self.expand(x)
        return x

    @torch.no_grad()
    def forward_trace_dict(self, x):
        """
        与 ``forward`` 同步的中间结果，用于测试时打印。
        键：``x_in`` / ``after_time_merge`` / ``after_input_if`` / ``after_linear1`` /
        ``after_if1`` / ``after_linear2`` / ``after_x1mx2``（flatten 后标量，即 ``x1-x2``）/ ``y_out``。
        """
        steps = {}
        if x.dim() == 4:
            x = x.squeeze(2)
        steps["x_in"] = x.detach().clone()
        if self.T > 0:
            x = _repeat_time_merge(x, self.T)
            steps["after_time_merge"] = x.detach().clone()
        x = self.input_if(x)
        steps["after_input_if"] = x.detach().clone()
        if self.T > 0:
            sch = self.spike_schedule
            if sch in ("weight_sign_pos_front", "weight_sign_neg_front"):
                x = first_linear_with_weight_sign_schedule(
                    x, self.T, self.linear1, sch
                )
            else:
                x4 = temporal_rearrange_after_first_if(
                    x.unsqueeze(2), self.T, sch
                )
                x = _apply_linear_1x2(x4.squeeze(2), self.linear1)
        else:
            x = _apply_linear_1x2(x, self.linear1)
        steps["after_linear1"] = x.detach().clone()
        x = self.if1(x)
        steps["after_if1"] = x.detach().clone()
        x = _apply_linear_1x2(x, self.linear2)
        steps["after_linear2"] = x.detach().clone()
        x = torch.flatten(x, 1)
        steps["after_x1mx2"] = x.detach().clone()
        if self.T > 0:
            x = self.expand(x)
        steps["y_out"] = x.detach().clone()
        return steps


def _format_tensor_short(t, max_elems=8):
    """CPU numpy 风格单行，过长则截断。"""
    if t is None:
        return "None"
    a = t.detach().float().cpu().reshape(-1).numpy()
    if a.size <= max_elems:
        return str(a.tolist())
    return str(a[:max_elems].tolist()) + " ... (共 %d 元素)" % a.size


def format_diff1d_trace(steps, n_samples, T, y_true=None):
    """
    将 ``forward_trace_dict`` 结果格式化为可读字符串（前 ``n_samples`` 条 batch 样本）。
    ``T>0`` 时 ``after_*`` 为 ``[T*B,...]``，按原 batch 下标展示各时间步。
    """
    lines = []
    x_in = steps["x_in"]
    B = x_in.size(0)
    n = min(int(n_samples), B)
    for i in range(n):
        lines.append("--- sample i=%d ---" % i)
        x1 = float(x_in[i, 0, 0].item())
        x2 = float(x_in[i, 0, 1].item())
        d = x1 - x2
        lines.append(
            "  x = [x1=%.6f, x2=%.6f]  (约束 x1>=x2)  (x1-x2)=%.6f"
            % (x1, x2, d)
        )
        if y_true is not None and i < y_true.shape[0]:
            lines.append(
                "  y_true = %.6f" % (float(y_true[i].view(-1)[0].item()),)
            )
        if T <= 0:
            lines.append(
                "  after_input_if: %s"
                % (_format_tensor_short(steps["after_input_if"][i]),)
            )
            lines.append(
                "  after_linear1: %s"
                % (_format_tensor_short(steps["after_linear1"][i]),)
            )
            lines.append(
                "  after_if1: %s" % (_format_tensor_short(steps["after_if1"][i]),)
            )
            lines.append(
                "  after_linear2: %s"
                % (_format_tensor_short(steps["after_linear2"][i]),)
            )
            if "after_x1mx2" in steps:
                lines.append(
                    "  after_x1mx2 (flatten): %s"
                    % (_format_tensor_short(steps["after_x1mx2"][i]),)
                )
            yo = steps["y_out"][i].view(-1)
            lines.append("  y_pred = %.6f" % (float(yo[0].item()),))
        else:
            # [T*B, 1, L]：index = t*B + b；每步为 [1,L]（L=2 为两维输入），勿与「T 个标量」混淆
            def row(tb, b):
                rows = []
                for t in range(T):
                    rows.append(tb[t * B + b])
                return torch.stack(rows, dim=0)

            lines.append(
                "  (T=%d：每步一行；每步 [1,L] 共 L=2 个数对应两维特征，非 2T 个时间步)"
                % (T,)
            )
            for name in (
                "after_input_if",
                "after_linear1",
                "after_if1",
                "after_linear2",
                "after_x1mx2",
            ):
                if name not in steps:
                    continue
                rt = row(steps[name], i)
                lines.append("  %s:" % (name,))
                for t in range(T):
                    lines.append(
                        "    t=%d: %s"
                        % (t, _format_tensor_short(rt[t]))
                    )
            y = steps["y_out"]
            if y.dim() == 3:
                lines.append(
                    "  y_pred [T,B,1] 对 (t,b=i): %s"
                    % (
                        str(
                            [
                                float(y[t, i, 0].item())
                                for t in range(T)
                            ]
                        ),
                    )
                )
                lines.append(
                    "  y_pred 时间平均 (与 val_reg 一致): %.6f"
                    % (float(y[:, i, 0].mean().item()),)
                )
    return "\n".join(lines)


def toy_diff1d():
    return ToyDiff1D()
