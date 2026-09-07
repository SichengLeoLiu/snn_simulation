"""Oxford-IIIT Pet VGG16-BN encoder with linear-readout classification / segmentation.

Shared encoder
--------------
    Conv → BN → QCFS/IF, AvgPool, no residual, no ASPP, no skip connections.
    13 encoder IFs. ImageNet VGG-16 BN conv/BN init.

Tasks
-----
    classification : GAP + Linear(512, 37). Final class scores are not quantized.
    segmentation   : 5× (bilinear ×2, Conv-BN-IF) decoder, then 1×1 Linear Conv to
                     foreground/background. No skip fusion. Final scores are not quantized.

``head_if=True`` puts BN-IF (seg) or IF (cls) on the readout; that is an ablation,
not the default protocol.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.FCN import _load_vgg16_bn_features
from Models.layer import ExpandTemporalDim, IF, MergeTemporalDim, add_dimention
from Models.VGG import (
    NOISE_POSITIONS,
    SPIKE_SCHEDULE_MODES,
    _forward_sequential_first_if_no_schedule,
    _forward_sequential_first_if_spike_schedule,
)

NUM_CLS = 37
NUM_SEG = 2
IGNORE_INDEX = 255
STRIDE = 32
DECODER_CHANNELS = (256, 128, 64, 64, 64)


def conv_bn_if(cin, cout, k=3, stride=1, padding=1):
    return [
        nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(cout),
        IF(),
    ]


class VGG16BNEncoder(nn.Module):
    """VGG-16 BN stages 1–5. Identical for classification and segmentation."""

    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(*conv_bn_if(3, 64), *conv_bn_if(64, 64), nn.AvgPool2d(2, 2))
        self.stage2 = nn.Sequential(*conv_bn_if(64, 128), *conv_bn_if(128, 128), nn.AvgPool2d(2, 2))
        self.stage3 = nn.Sequential(
            *conv_bn_if(128, 256), *conv_bn_if(256, 256), *conv_bn_if(256, 256), nn.AvgPool2d(2, 2)
        )
        self.stage4 = nn.Sequential(
            *conv_bn_if(256, 512), *conv_bn_if(512, 512), *conv_bn_if(512, 512), nn.AvgPool2d(2, 2)
        )
        self.stage5 = nn.Sequential(
            *conv_bn_if(512, 512), *conv_bn_if(512, 512), *conv_bn_if(512, 512), nn.AvgPool2d(2, 2)
        )

    def forward_from_stage2(self, x):
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.stage5(x)


class PetVGGBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = 0
        self.merge = MergeTemporalDim(0)
        self.expand = ExpandTemporalDim(0)
        self.spike_schedule = "normal"
        self.first_layer_input_noise_sigma = 0.0
        self.first_layer_input_noise_type = "gaussian"
        self.first_layer_input_noise_position = "post_input_if"
        self._mne_layer_map = "legacy"
        self.encoder = VGG16BNEncoder()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def set_spike_schedule(self, mode):
        if mode not in SPIKE_SCHEDULE_MODES:
            raise ValueError(f"spike_schedule must be {sorted(SPIKE_SCHEDULE_MODES)}, got {mode}")
        self.spike_schedule = mode

    def set_T(self, T):
        self.T = int(T)
        for module in self.modules():
            if isinstance(module, (IF, ExpandTemporalDim, MergeTemporalDim)):
                module.T = self.T

    def set_L(self, L):
        for module in self.modules():
            if isinstance(module, IF):
                module.L = int(L)

    def set_mode(self, mode="normal"):
        for module in self.modules():
            if isinstance(module, IF):
                module.mode = mode

    def set_first_layer_input_noise_sigma(self, sigma=0.0):
        self.first_layer_input_noise_sigma = max(0.0, float(sigma))

    def set_first_layer_input_noise_type(self, noise_type="gaussian"):
        nt = str(noise_type).strip().lower()
        if nt not in ("gaussian", "pink"):
            raise ValueError(f"noise_type must be gaussian or pink, got {noise_type}")
        self.first_layer_input_noise_type = nt

    def set_first_layer_input_noise_position(self, position="post_input_if"):
        pos = str(position).strip().lower()
        if pos not in NOISE_POSITIONS:
            raise ValueError(f"noise position must be {list(NOISE_POSITIONS)}, got {position}")
        self.first_layer_input_noise_position = pos

    def _time_mean(self, tensor):
        if self.T <= 0:
            return tensor
        batch = tensor.shape[0] // self.T
        return tensor.view(self.T, batch, *tensor.shape[1:]).mean(0)

    def encode(self, images):
        if self.T > 0:
            x = add_dimention(images, self.T)
            x = self.merge(x)
            x = _forward_sequential_first_if_spike_schedule(
                self.encoder.stage1,
                x,
                self.T,
                self.spike_schedule,
                self.first_layer_input_noise_sigma,
                self.first_layer_input_noise_type,
                self.first_layer_input_noise_position,
            )
        else:
            x = _forward_sequential_first_if_no_schedule(
                self.encoder.stage1,
                images,
                self.first_layer_input_noise_sigma,
                self.first_layer_input_noise_type,
                self.first_layer_input_noise_position,
            )
        return self.encoder.forward_from_stage2(x)


class PetVGGClassifier(PetVGGBase):
    def __init__(self, num_classes: int = NUM_CLS, head_if: bool = False):
        super().__init__()
        self.num_classes = int(num_classes)
        self.head_if = bool(head_if)
        self.pool = nn.AdaptiveAvgPool2d(1)
        if self.head_if:
            self.classifier = nn.Sequential(nn.Linear(512, self.num_classes), IF())
        else:
            self.classifier = nn.Linear(512, self.num_classes)
        self._init_weights()

    def forward(self, images):
        x = self.encode(images)
        x = self.pool(x).flatten(1)
        return self._time_mean(self.classifier(x))


class PetVGGSegmentor(PetVGGBase):
    def __init__(self, num_classes: int = NUM_SEG, head_if: bool = False):
        super().__init__()
        self.num_classes = int(num_classes)
        self.head_if = bool(head_if)
        blocks = []
        cin = 512
        for cout in DECODER_CHANNELS:
            blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    *conv_bn_if(cin, cout),
                )
            )
            cin = cout
        self.decoder = nn.Sequential(*blocks)
        if self.head_if:
            self.classifier = nn.Sequential(*conv_bn_if(cin, self.num_classes, k=1, padding=0))
        else:
            self.classifier = nn.Conv2d(cin, self.num_classes, 1, bias=True)
        self._init_weights()

    def forward(self, images):
        height, width = images.shape[-2:]
        x = self.encode(images)
        x = self.decoder(x)
        logits = self._time_mean(self.classifier(x))
        if logits.shape[-2:] != (height, width):
            logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        return logits


def load_vgg16_bn_into_encoder(encoder: VGG16BNEncoder) -> int:
    src = _load_vgg16_bn_features()
    dst_convs, dst_bns = [], []
    for stage in (encoder.stage1, encoder.stage2, encoder.stage3, encoder.stage4, encoder.stage5):
        modules = list(stage)
        for i, module in enumerate(modules):
            if not (isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3)):
                continue
            if i + 1 >= len(modules) or not isinstance(modules[i + 1], nn.BatchNorm2d):
                continue
            dst_convs.append(module)
            dst_bns.append(modules[i + 1])
            if len(dst_convs) == 13:
                break
        if len(dst_convs) == 13:
            break
    src_convs = [m for m in src if isinstance(m, nn.Conv2d)]
    src_bns = [m for m in src if isinstance(m, nn.BatchNorm2d)]
    n = min(13, len(dst_convs), len(src_convs))
    with torch.no_grad():
        for dst, src_m in zip(dst_convs[:n], src_convs[:n]):
            if dst.weight.shape == src_m.weight.shape:
                dst.weight.copy_(src_m.weight)
        for dst, src_m in zip(dst_bns[:n], src_bns[:n]):
            if dst.weight.shape == src_m.weight.shape:
                dst.weight.copy_(src_m.weight)
                dst.bias.copy_(src_m.bias)
                dst.running_mean.copy_(src_m.running_mean)
                dst.running_var.copy_(src_m.running_var)
    return n


def count_maxpool2d(root: nn.Module) -> int:
    return sum(isinstance(module, nn.MaxPool2d) for module in root.modules())


def count_if(root: nn.Module) -> int:
    return sum(isinstance(module, IF) for module in root.modules())
