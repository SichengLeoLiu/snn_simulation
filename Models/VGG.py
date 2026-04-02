import torch.nn as nn
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


def _forward_sequential_first_if_spike_schedule(seq, x, T, spike_schedule):
    if_idx, conv_idx = _first_if_and_next_conv_idx(seq)
    for i in range(if_idx):
        x = seq[i](x)
    x = seq[if_idx](x)
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

    def forward(self, x):
        if self.T > 0:
            x = add_dimention(x, self.T)
            x = self.merge(x)
            out = _forward_sequential_first_if_spike_schedule(
                self.layer1, x, self.T, self.spike_schedule
            )
        else:
            out = self.layer1(x)
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

    def forward(self, x):
        if self.T > 0:
            x = add_dimention(x, self.T)
            x = self.merge(x)
            out = _forward_sequential_first_if_spike_schedule(
                self.layer1, x, self.T, self.spike_schedule
            )
        else:
            out = self.layer1(x)
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