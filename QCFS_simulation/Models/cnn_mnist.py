import torch
import torch.nn as nn
from Models.layer import IF, MergeTemporalDim, ExpandTemporalDim, add_dimention
from Models.spike_temporal_adjust import (
    SPIKE_SCHEDULE_MODES,
    temporal_rearrange_after_first_if,
    first_conv_with_weight_sign_schedule,
)


class CNN2MNIST(nn.Module):
    """两层卷积 + 全连接，用于 MNIST（1×28×28）。

    输入先经 IF，第一层 Conv 仅见 IF 输出。T>0 时可在第一层 Conv 前对时间维脉冲序列重排
    （见 ``set_spike_schedule``）：normal / uniform / front_load / back_load /
    weight_sign_pos_front / weight_sign_neg_front。
    """

    def __init__(self, num_classes=10, c1=2, c2=4):
        super().__init__()
        self.T = 0
        self.merge = MergeTemporalDim(0)
        self.expand = ExpandTemporalDim(0)
        self.spike_schedule = "normal"
        self.first_layer_input_noise_sigma = 0.0
        self.first_layer_input_noise_type = "gaussian"
        self.c1 = int(c1)
        self.c2 = int(c2)

        self.input_if = IF()
        self.conv1 = nn.Conv2d(1, self.c1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.c1)
        self.if1 = IF()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(self.c1, self.c2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.c2)
        self.if2 = IF()
        self.pool2 = nn.MaxPool2d(2)
        self.classifier = nn.Linear(self.c2 * 7 * 7, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

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
        """IF 神经元前向模式（与脉冲时间重排 ``spike_schedule`` 无关）。"""
        for module in self.modules():
            if isinstance(module, IF):
                module.mode = mode

    def set_first_layer_input_noise_sigma(self, sigma=0.0):
        """设置第一层输入噪声标准差（input_if 后、conv1 前）。"""
        self.first_layer_input_noise_sigma = max(0.0, float(sigma))

    def set_first_layer_input_noise_type(self, noise_type="gaussian"):
        """设置第一层输入噪声类型：gaussian | pink。"""
        nt = str(noise_type).strip().lower()
        if nt not in ("gaussian", "pink"):
            raise ValueError("noise_type 必须为 gaussian 或 pink，收到: %s" % (noise_type,))
        self.first_layer_input_noise_type = nt

    def resolution_aware_noise_regularization(self, T=None, eps=1e-8):
        """
        R_rho = sum_l [ L_l^2 / (C_l * T * lambda_l^2) ] * ||W_l||_2^2

        当前 CNN2 采用两层卷积对应两个 IF 阈值：
        - conv1.weight <-> if1.thresh
        - conv2.weight <-> if2.thresh
        """
        t_eff = int(self.T if T is None else T)
        t_eff = max(t_eff, 1)

        def _term(weight, if_layer):
            c_l = weight.shape[0]
            l_l = float(if_layer.L)
            lambda_l = float(if_layer.thresh.detach().clamp(min=eps).item())
            omega = (l_l * l_l) / (float(c_l) * float(t_eff) * (lambda_l * lambda_l))
            return weight.pow(2).sum() * omega

        reg = _term(self.conv1.weight, self.if1) + _term(self.conv2.weight, self.if2)
        return reg

    def _pink_noise_like(self, x, T):
        """
        在时间维生成近似 1/f 粉红噪声（每个像素/通道独立），返回与 x 同形状噪声。
        x 形状要求为 [T*B, C, H, W]。
        """
        if T <= 1:
            return torch.randn_like(x)
        tb, c, h, w = x.shape
        if tb % T != 0:
            return torch.randn_like(x)
        b = tb // T
        # MPS 对 FFT 轴有限制：时间维放到最后一维再做 rfft。
        white = torch.randn((b, c, h, w, T), device=x.device, dtype=x.dtype)
        freq = torch.fft.rfft(white, dim=-1)
        n_freq = freq.shape[-1]
        f = torch.arange(n_freq, device=x.device, dtype=torch.float32)
        scale = torch.ones_like(f)
        scale[1:] = 1.0 / torch.sqrt(f[1:])
        scale = scale.to(dtype=x.dtype).view(1, 1, 1, 1, n_freq)
        pink = torch.fft.irfft(freq * scale, n=T, dim=-1)
        std = pink.std(dim=-1, unbiased=False, keepdim=True).clamp(min=1e-6)
        pink = pink / std
        pink = pink.permute(4, 0, 1, 2, 3).contiguous()  # [T, B, C, H, W]
        return pink.reshape(tb, c, h, w)

    def _inject_first_layer_input_noise(self, x):
        sigma = self.first_layer_input_noise_sigma
        if sigma <= 0:
            return x
        if self.first_layer_input_noise_type == "pink":
            noise = self._pink_noise_like(x, self.T) if self.T > 0 else torch.randn_like(x)
        else:
            noise = torch.randn_like(x)
        return x + noise * sigma

    @staticmethod
    def _if_out_to_firing_map(x_tb, if_layer, T):
        """
        x_tb: IF 输出 [T*B,C,H,W]（T>0）或 [B,C,H,W]（T=0）
        返回 [B,C,H,W]，语义近似 sum_t(spike_t)/thresh（T>0）或 h/thresh（T=0）。
        """
        th = if_layer.thresh.data.clamp(min=1e-8)
        if T and T > 0:
            tb, c, h, w = x_tb.shape
            b = tb // T
            s = x_tb.view(T, b, c, h, w).sum(dim=0)
            return s / th
        return x_tb / th

    def forward_with_if_features(self, x):
        """
        与 ``forward`` 相同计算图，额外返回 IF1 / IF2 后的归一化特征图（池化前），用于可视化。
        返回 (logits, feat_if1, feat_if2)，feat_* 形状 [B,C,H,W]；T>0 时 logits 为 [T,B,num_classes]。
        """
        T = self.T
        if T > 0:
            x = x.clone()
            x = add_dimention(x, T)
            x = self.merge(x)

        x = self.input_if(x)
        x = self._inject_first_layer_input_noise(x)

        if T > 0:
            sch = self.spike_schedule
            if sch in ("weight_sign_pos_front", "weight_sign_neg_front"):
                x = first_conv_with_weight_sign_schedule(x, T, self.conv1, sch)
            else:
                x = temporal_rearrange_after_first_if(x, T, sch)
                x = self.conv1(x)
        else:
            x = self.conv1(x)

        x = self.bn1(x)
        x = self.if1(x)
        feat_if1 = self._if_out_to_firing_map(x, self.if1, T)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.if2(x)
        feat_if2 = self._if_out_to_firing_map(x, self.if2, T)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if T > 0:
            x = self.expand(x)
        return x, feat_if1, feat_if2

    def forward(self, x):
        if self.T > 0:
            x = add_dimention(x, self.T)
            x = self.merge(x)

        x = self.input_if(x)
        x = self._inject_first_layer_input_noise(x)

        if self.T > 0:
            sch = self.spike_schedule
            if sch in ("weight_sign_pos_front", "weight_sign_neg_front"):
                x = first_conv_with_weight_sign_schedule(
                    x, self.T, self.conv1, sch
                )
            else:
                x = temporal_rearrange_after_first_if(x, self.T, sch)
                x = self.conv1(x)
        else:
            x = self.conv1(x)

        x = self.bn1(x)
        x = self.if1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.if2(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if self.T > 0:
            x = self.expand(x)
        return x


def cnn2_mnist(num_classes=10, c1=2, c2=4):
    return CNN2MNIST(num_classes=num_classes, c1=c1, c2=c2)


# 旧版 checkpoint：nn.Sequential 命名为 features，下标与当前子模块对应关系
_LEGACY_FEATURES_IDX_MAP = {
    0: "input_if",
    1: "conv1",
    2: "bn1",
    3: "if1",
    5: "conv2",
    6: "bn2",
    7: "if2",
}


def remap_legacy_cnn2_state_dict(state_dict):
    """
    将旧版 ``features.{idx}.*`` 键名映射为 ``input_if`` / ``conv1`` / …
    若未检测到可映射键，返回原字典且 remapped=False。
    """
    if not any(k.startswith("features.") for k in state_dict):
        return state_dict, False
    new_sd = {}
    remapped = False
    for k, v in state_dict.items():
        if not k.startswith("features."):
            new_sd[k] = v
            continue
        rest = k[len("features.") :]
        dot = rest.find(".")
        if dot < 0:
            new_sd[k] = v
            continue
        try:
            idx = int(rest[:dot])
        except ValueError:
            new_sd[k] = v
            continue
        tail = rest[dot + 1 :]
        if idx in _LEGACY_FEATURES_IDX_MAP:
            new_k = _LEGACY_FEATURES_IDX_MAP[idx] + "." + tail
            new_sd[new_k] = v
            remapped = True
        else:
            # MaxPool 等无参数层不应有 state；若有残留键则跳过
            pass
    return new_sd, remapped
