"""BN-free MLP for CIFAR-10/100 with QCFS IF layers and first-layer noise hooks."""

from __future__ import annotations

import torch
import torch.nn as nn

from Models.layer import IF, MergeTemporalDim, ExpandTemporalDim, add_dimention


class _CIFARMLPBase(nn.Module):
    """Shared IF / noise / T-L-mode API for CIFAR fully-connected models."""

    def __init__(self):
        super().__init__()
        self.T = 0
        self.merge = MergeTemporalDim(0)
        self.expand = ExpandTemporalDim(0)
        self.first_layer_input_noise_sigma = 0.0
        self.first_layer_input_noise_type = "gaussian"
        self.first_layer_input_noise_position = "post_input_if"

    def set_T(self, T):
        self.T = int(T)
        self.merge.T = self.T
        self.expand.T = self.T
        for module in self.modules():
            if isinstance(module, (IF, ExpandTemporalDim)):
                module.T = self.T
                if self.T > 0:
                    module.spike_counts = [0] * self.T
                    module.total_elements = [0] * self.T

    def set_L(self, L):
        for module in self.modules():
            if isinstance(module, IF):
                module.L = int(L)

    def set_scaling_factor(self, scaling_factor=1.0):
        for module in self.modules():
            if isinstance(module, IF):
                module.scaling_factor = float(scaling_factor)

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

    def _inject_first_layer_input_noise(self, x):
        sigma = self.first_layer_input_noise_sigma
        if sigma <= 0:
            return x
        noise = torch.randn_like(x)
        return x + noise * sigma

    def _apply_input_if_and_noise(self, x):
        if self.first_layer_input_noise_position == "pre_input_if":
            x = self._inject_first_layer_input_noise(x)
            x = self.input_if(x)
            return x
        x = self.input_if(x)
        x = self._inject_first_layer_input_noise(x)
        return x

    def _init_linears(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class FC2CIFAR(_CIFARMLPBase):
    """CIFAR BN-free 2-layer MLP: 3072 -> h -> C."""

    def __init__(self, num_classes=10, hidden_dim=512):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.input_if = IF()
        self.fc1 = nn.Linear(3 * 32 * 32, self.hidden_dim)
        self.if1 = IF()
        self.fc2 = nn.Linear(self.hidden_dim, num_classes)
        self._init_linears()

    def forward(self, x):
        if self.T > 0:
            x = add_dimention(x, self.T)
            x = self.merge(x)
        x = torch.flatten(x, 1)
        x = self._apply_input_if_and_noise(x)
        x = self.fc1(x)
        x = self.if1(x)
        x = self.fc2(x)
        if self.T > 0:
            x = self.expand(x)
        return x


class FC3CIFAR(_CIFARMLPBase):
    """CIFAR BN-free 3-layer MLP: 3072 -> h -> 2h -> C (no BatchNorm)."""

    def __init__(self, num_classes=10, hidden_dim=512):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.hidden_dim2 = int(hidden_dim) * 2
        self.input_if = IF()
        self.fc1 = nn.Linear(3 * 32 * 32, self.hidden_dim)
        self.if1 = IF()
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim2)
        self.if2 = IF()
        self.fc3 = nn.Linear(self.hidden_dim2, num_classes)
        self._init_linears()

    def forward(self, x):
        if self.T > 0:
            x = add_dimention(x, self.T)
            x = self.merge(x)
        x = torch.flatten(x, 1)
        x = self._apply_input_if_and_noise(x)
        x = self.fc1(x)
        x = self.if1(x)
        x = self.fc2(x)
        x = self.if2(x)
        x = self.fc3(x)
        if self.T > 0:
            x = self.expand(x)
        return x


def fc2_cifar(num_classes=10, hidden_dim=512):
    return FC2CIFAR(num_classes=num_classes, hidden_dim=hidden_dim)


def fc3_cifar(num_classes=10, hidden_dim=512):
    return FC3CIFAR(num_classes=num_classes, hidden_dim=hidden_dim)
