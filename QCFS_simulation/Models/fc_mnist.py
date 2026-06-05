import torch
import torch.nn as nn

from Models.layer import IF, MergeTemporalDim, ExpandTemporalDim, add_dimention


class FC2MNIST(nn.Module):
    """MNIST 两层全连接网络（单隐藏层）+ IF 量化/脉冲支持。"""

    def __init__(self, num_classes=10, hidden_dim=256):
        super().__init__()
        self.T = 0
        self.hidden_dim = int(hidden_dim)
        self.merge = MergeTemporalDim(0)
        self.expand = ExpandTemporalDim(0)
        self.first_layer_input_noise_sigma = 0.0
        self.first_layer_input_noise_type = "gaussian"

        self.input_if = IF()
        self.fc1 = nn.Linear(28 * 28, self.hidden_dim)
        self.if1 = IF()
        self.fc2 = nn.Linear(self.hidden_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

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

    def _inject_first_layer_input_noise(self, x):
        sigma = self.first_layer_input_noise_sigma
        if sigma <= 0:
            return x
        # 对全连接输入，pink 退化为同尺度高斯噪声。
        noise = torch.randn_like(x)
        return x + noise * sigma

    def forward(self, x):
        if self.T > 0:
            x = add_dimention(x, self.T)
            x = self.merge(x)
        x = torch.flatten(x, 1)
        x = self.input_if(x)
        x = self._inject_first_layer_input_noise(x)
        x = self.fc1(x)
        x = self.if1(x)
        x = self.fc2(x)
        if self.T > 0:
            x = self.expand(x)
        return x


def fc2_mnist(num_classes=10, hidden_dim=256):
    return FC2MNIST(num_classes=num_classes, hidden_dim=hidden_dim)


class FC3MNIST(nn.Module):
    """MNIST 三层全连接网络（双隐藏层）+ IF 量化/脉冲支持。"""

    def __init__(self, num_classes=10, hidden_dim=64):
        super().__init__()
        self.T = 0
        self.hidden_dim = int(hidden_dim)
        self.hidden_dim2 = int(hidden_dim) * 2
        self.merge = MergeTemporalDim(0)
        self.expand = ExpandTemporalDim(0)
        self.first_layer_input_noise_sigma = 0.0
        self.first_layer_input_noise_type = "gaussian"

        self.input_if = IF()
        self.fc1 = nn.Linear(28 * 28, self.hidden_dim)
        self.if1 = IF()
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim2)
        self.if2 = IF()
        self.fc3 = nn.Linear(self.hidden_dim2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

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

    def _inject_first_layer_input_noise(self, x):
        sigma = self.first_layer_input_noise_sigma
        if sigma <= 0:
            return x
        noise = torch.randn_like(x)
        return x + noise * sigma

    def forward(self, x):
        if self.T > 0:
            x = add_dimention(x, self.T)
            x = self.merge(x)
        x = torch.flatten(x, 1)
        x = self.input_if(x)
        x = self._inject_first_layer_input_noise(x)
        x = self.fc1(x)
        x = self.if1(x)
        x = self.fc2(x)
        x = self.if2(x)
        x = self.fc3(x)
        if self.T > 0:
            x = self.expand(x)
        return x


def fc3_mnist(num_classes=10, hidden_dim=64):
    return FC3MNIST(num_classes=num_classes, hidden_dim=hidden_dim)
