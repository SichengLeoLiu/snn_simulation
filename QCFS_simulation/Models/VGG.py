import torch.nn as nn
import torch
from Models.layer import *
from Models.spike_temporal_adjust import (
    SPIKE_SCHEDULE_MODES,
    first_conv_with_weight_sign_schedule,
    temporal_rearrange_after_first_if,
)


def _first_if_and_next_conv_idx(seq):
    if_idx = None
    for i, m in enumerate(seq):
        if isinstance(m, IF):
            if_idx = i
            break
    if if_idx is None:
        raise RuntimeError("VGG 子模块中未找到 IF，无法应用 spike_schedule")
    conv_idx = None
    for j in range(if_idx + 1, len(seq)):
        if isinstance(seq[j], nn.Conv2d):
            conv_idx = j
            break
    if conv_idx is None:
        raise RuntimeError("第一个 IF 之后未找到 Conv2d")
    return if_idx, conv_idx


def _pink_noise_like(x, T):
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


def _inject_noise_tensor(x, sigma, noise_type, T):
    if sigma <= 0:
        return x
    if noise_type == "pink":
        noise = _pink_noise_like(x, T) if T > 0 else torch.randn_like(x)
    else:
        noise = torch.randn_like(x)
    return x + noise * sigma


def _forward_sequential_first_if_spike_schedule(
    seq,
    x,
    T,
    spike_schedule,
    noise_sigma=0.0,
    noise_type="gaussian",
    noise_position="post_input_if",
):
    if_idx, conv_idx = _first_if_and_next_conv_idx(seq)
    for i in range(if_idx):
        x = seq[i](x)
    if noise_position == "pre_input_if":
        x = _inject_noise_tensor(x, noise_sigma, noise_type, T)
        x = seq[if_idx](x)
    else:
        x = seq[if_idx](x)
        x = _inject_noise_tensor(x, noise_sigma, noise_type, T)
    sch = spike_schedule
    if sch in ("weight_sign_pos_front", "weight_sign_neg_front"):
        x = first_conv_with_weight_sign_schedule(x, T, seq[conv_idx], sch)
        for j in range(conv_idx + 1, len(seq)):
            x = seq[j](x)
    else:
        x = temporal_rearrange_after_first_if(x, T, sch)
        for j in range(if_idx + 1, len(seq)):
            x = seq[j](x)
    return x


def _forward_sequential_first_if_no_schedule(
    seq,
    x,
    noise_sigma=0.0,
    noise_type="gaussian",
    noise_position="post_input_if",
):
    """
    T=0 路径：在 layer1 的第一个 IF 后注入噪声，再继续后续层。
    """
    if_idx, _ = _first_if_and_next_conv_idx(seq)
    for i in range(if_idx):
        x = seq[i](x)
    if noise_position == "pre_input_if":
        x = _inject_noise_tensor(x, noise_sigma, noise_type, 0)
        x = seq[if_idx](x)
    else:
        x = seq[if_idx](x)
        x = _inject_noise_tensor(x, noise_sigma, noise_type, 0)
    for j in range(if_idx + 1, len(seq)):
        x = seq[j](x)
    return x


cfg = {
    'VGG11': [
        [64, 'M'],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG13': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ]
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes, dropout):
        super(VGG, self).__init__()
        self.init_channels = 3
        self.T = 0
        self.merge = MergeTemporalDim(0)
        self.expand = ExpandTemporalDim(0)
        self.spike_schedule = "normal"
        self.first_layer_input_noise_sigma = 0.0
        self.first_layer_input_noise_type = "gaussian"
        self.first_layer_input_noise_position = "post_input_if"
        self.loss = 0
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout)
        if num_classes == 1000:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512*7*7, 4096),
                IF(),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                IF(),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 4096),
                IF(),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                IF(),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes)
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layers(self, cfg, dropout):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(IF())
                layers.append(nn.Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers)

    def set_spike_schedule(self, mode):
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
        return

    def set_scaling_factor(self, scaling_factor=1.0):
        """ Set scaling factor for the warmup x0 / scaling_factor at timestep 1
        """
        for module in self.modules():
            if isinstance(module, IF):
                module.scaling_factor = scaling_factor

    def set_mode(self, mode='normal'):
        for module in self.modules():
            if isinstance(module, IF):
                module.mode = mode
                # break # only set the first IF module's mode

    def set_first_layer_input_noise_sigma(self, sigma=0.0):
        """设置第一层输入噪声标准差（layer1 第一个 IF 后、后续 Conv 前）。"""
        self.first_layer_input_noise_sigma = max(0.0, float(sigma))

    def set_first_layer_input_noise_type(self, noise_type="gaussian"):
        """设置第一层输入噪声类型：gaussian | pink。"""
        nt = str(noise_type).strip().lower()
        if nt not in ("gaussian", "pink"):
            raise ValueError("noise_type 必须为 gaussian 或 pink，收到: %s" % (noise_type,))
        self.first_layer_input_noise_type = nt

    def set_first_layer_input_noise_position(self, position="post_input_if"):
        pos = str(position).strip().lower()
        if pos not in ("post_input_if", "pre_input_if"):
            raise ValueError(
                "first_layer_input_noise_position 必须为 post_input_if 或 pre_input_if，收到: %s"
                % (position,)
            )
        self.first_layer_input_noise_position = pos

    def forward(self, x):
        if self.T > 0:
            x = add_dimention(x, self.T)
            x = self.merge(x)
            out = _forward_sequential_first_if_spike_schedule(
                self.layer1,
                x,
                self.T,
                self.spike_schedule,
                self.first_layer_input_noise_sigma,
                self.first_layer_input_noise_type,
                self.first_layer_input_noise_position,
            )
        else:
            out = _forward_sequential_first_if_no_schedule(
                self.layer1,
                x,
                self.first_layer_input_noise_sigma,
                self.first_layer_input_noise_type,
                self.first_layer_input_noise_position,
            )
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.classifier(out)
        if self.T > 0:
            out = self.expand(out)
        return out

    def get_firing_rate(self, save_to_file=None, experiment_name=None):
        """the percentage of 1s out of a spike tensor

        Args:
            save_to_file (str, optional): File path, for saving spike counts. Default is None.
            experiment_name (str, optional): Experiment name (e.g., 'Baseline' or 'TPP'). Default is None.
        """
        count = 0
        layer_spike_counts = []

        for module in self.modules():
            if isinstance(module, IF):
                count += 1
                spike_count = sum(module.spike_counts)
                layer_spike_counts.append(spike_count)
                print(f"layer {count} spike counts: {spike_count}")

        if save_to_file:
            import os
            import csv

            os.makedirs(os.path.dirname(save_to_file), exist_ok=True)

            file_exists = os.path.isfile(save_to_file)

            with open(save_to_file, 'a', newline='') as f:
                writer = csv.writer(f)

                if not file_exists:
                    headers = ['experiment_name', 'T'] + [f'layer_{i+1}' for i in range(count)]
                    writer.writerow(headers)

                row_data = [experiment_name or 'unnamed', self.T] + layer_spike_counts
                writer.writerow(row_data)

            print(f"Spike counts saved to {save_to_file}")


class VGG_woBN(nn.Module):
    def __init__(self, vgg_name, num_classes, dropout):
        super(VGG_woBN, self).__init__()
        self.init_channels = 3
        self.T = 0
        self.merge = MergeTemporalDim(0)
        self.expand = ExpandTemporalDim(0)
        self.spike_schedule = "normal"
        self.first_layer_input_noise_sigma = 0.0
        self.first_layer_input_noise_type = "gaussian"
        self.first_layer_input_noise_position = "post_input_if"
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout)
        if num_classes == 1000:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512*7*7, 4096),
                IF(),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                IF(),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 4096),
                IF(),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                IF(),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layers(self, cfg, dropout):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(IF())
                layers.append(nn.Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers)

    def set_spike_schedule(self, mode):
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
        return

    def set_scaling_factor(self, scaling_factor=1.0):
        for module in self.modules():
            if isinstance(module, IF):
                module.scaling_factor = scaling_factor

    def set_mode(self, mode="normal"):
        for module in self.modules():
            if isinstance(module, IF):
                module.mode = mode

    def set_first_layer_input_noise_sigma(self, sigma=0.0):
        self.first_layer_input_noise_sigma = max(0.0, float(sigma))

    def set_first_layer_input_noise_type(self, noise_type="gaussian"):
        nt = str(noise_type).strip().lower()
        if nt not in ("gaussian", "pink"):
            raise ValueError("noise_type 必须为 gaussian 或 pink，收到: %s" % (noise_type,))
        self.first_layer_input_noise_type = nt

    def set_first_layer_input_noise_position(self, position="post_input_if"):
        pos = str(position).strip().lower()
        if pos not in ("post_input_if", "pre_input_if"):
            raise ValueError(
                "first_layer_input_noise_position 必须为 post_input_if 或 pre_input_if，收到: %s"
                % (position,)
            )
        self.first_layer_input_noise_position = pos

    def forward(self, x):
        if self.T > 0:
            x = add_dimention(x, self.T)
            x = self.merge(x)
            out = _forward_sequential_first_if_spike_schedule(
                self.layer1,
                x,
                self.T,
                self.spike_schedule,
                self.first_layer_input_noise_sigma,
                self.first_layer_input_noise_type,
                self.first_layer_input_noise_position,
            )
        else:
            out = _forward_sequential_first_if_no_schedule(
                self.layer1,
                x,
                self.first_layer_input_noise_sigma,
                self.first_layer_input_noise_type,
                self.first_layer_input_noise_position,
            )
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.classifier(out)
        if self.T > 0:
            out = self.expand(out)
        return out

def vgg16(num_classes, dropout=0.0):
    return VGG("VGG16", num_classes, dropout)


def remap_legacy_vgg_state_dict(state_dict):
    sd = dict(state_dict)
    keys = list(sd.keys())
    for k in keys:
        if k not in sd:
            continue
        if "relu.up" in k:
            sd[k[:-7] + "act.thresh"] = sd.pop(k)
        elif "up" in k:
            sd[k[:-2] + "thresh"] = sd.pop(k)
    return sd

def vgg16_wobn(num_classes, dropout=0.1):
    return VGG_woBN('VGG16', num_classes, dropout)

def vgg19(num_classes, dropout):
    return VGG('VGG19', num_classes, dropout)