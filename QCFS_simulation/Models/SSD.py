"""QCFS SSD300 with a VGG-16 BN-IF backbone.

Every backbone/extra Conv is Conv-BN-IF so MNE-L2 can fold BN and match IF
thresholds. Detection heads are unmatched Conv (same role as a classifier).
Train T=0; eval T=L with rate_uniform and post-input-IF noise on stage1.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from Models.layer import ExpandTemporalDim, IF, MergeTemporalDim, add_dimention
from Models.VGG import (
    NOISE_POSITIONS,
    SPIKE_SCHEDULE_MODES,
    _forward_sequential_first_if_no_schedule,
    _forward_sequential_first_if_spike_schedule,
)

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
NUM_CLASSES = 1 + len(VOC_CLASSES)

SOURCE_SHAPES = (38, 19, 10, 5, 3, 1)
ASPECT_RATIOS = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
MIN_SIZES = (30, 60, 111, 162, 213, 264)
MAX_SIZES = (60, 111, 162, 213, 264, 315)
STEPS = (8, 16, 32, 64, 100, 300)
IMAGE_SIZE = 300


def n_boxes_per_cell(aspects) -> int:
    return 2 + 2 * len(aspects)


def _conv_bn_if(cin, cout, k=3, stride=1, padding=1, dilation=1):
    return [
        nn.Conv2d(cin, cout, k, stride=stride, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(cout),
        IF(),
    ]


class SSD300VGG16(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = int(num_classes)
        self.T = 0
        self.merge = MergeTemporalDim(0)
        self.expand = ExpandTemporalDim(0)
        self.spike_schedule = "normal"
        self.first_layer_input_noise_sigma = 0.0
        self.first_layer_input_noise_type = "gaussian"
        self.first_layer_input_noise_position = "post_input_if"
        self._mne_layer_map = "legacy"

        self.stage1 = nn.Sequential(
            *_conv_bn_if(3, 64),
            *_conv_bn_if(64, 64),
            nn.AvgPool2d(2, 2),
        )
        self.stage2 = nn.Sequential(
            *_conv_bn_if(64, 128),
            *_conv_bn_if(128, 128),
            nn.AvgPool2d(2, 2),
        )
        self.stage3 = nn.Sequential(
            *_conv_bn_if(128, 256),
            *_conv_bn_if(256, 256),
            *_conv_bn_if(256, 256),
            nn.AvgPool2d(2, 2, ceil_mode=True),
        )
        self.stage4 = nn.Sequential(
            *_conv_bn_if(256, 512),
            *_conv_bn_if(512, 512),
            *_conv_bn_if(512, 512),
        )
        self.stage4_pool = nn.AvgPool2d(2, 2)
        self.stage5 = nn.Sequential(
            *_conv_bn_if(512, 512),
            *_conv_bn_if(512, 512),
            *_conv_bn_if(512, 512),
            nn.AvgPool2d(3, stride=1, padding=1),
            *_conv_bn_if(512, 1024, k=3, padding=6, dilation=6),
            *_conv_bn_if(1024, 1024, k=1, padding=0),
        )
        extra_cfg = (
            (1024, 256, 512, 2, 1),
            (512, 128, 256, 2, 1),
            (256, 128, 256, 1, 0),
            (256, 128, 256, 1, 0),
        )
        extras = []
        for cin, mid, cout, stride, padding in extra_cfg:
            extras.append(
                nn.Sequential(
                    *_conv_bn_if(cin, mid, k=1, padding=0),
                    *_conv_bn_if(mid, cout, k=3, stride=stride, padding=padding),
                )
            )
        self.extras = nn.ModuleList(extras)

        loc_layers = []
        conf_layers = []
        in_channels = (512, 1024, 512, 256, 256, 256)
        for cin, aspects in zip(in_channels, ASPECT_RATIOS):
            n = n_boxes_per_cell(aspects)
            loc_layers.append(nn.Conv2d(cin, n * 4, kernel_size=3, padding=1))
            conf_layers.append(nn.Conv2d(cin, n * self.num_classes, kernel_size=3, padding=1))
        self.loc_heads = nn.ModuleList(loc_layers)
        self.conf_heads = nn.ModuleList(conf_layers)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        self.register_buffer("priors", ssd300_priors(), persistent=False)

    def set_spike_schedule(self, mode):
        if mode not in SPIKE_SCHEDULE_MODES:
            raise ValueError(f"spike_schedule must be {sorted(SPIKE_SCHEDULE_MODES)}, got {mode}")
        self.spike_schedule = mode

    def set_T(self, T):
        self.T = int(T)
        for module in self.modules():
            if isinstance(module, (IF, ExpandTemporalDim)):
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

    def _sources(self, x):
        if self.T > 0:
            x = add_dimention(x, self.T)
            x = self.merge(x)
            x = _forward_sequential_first_if_spike_schedule(
                self.stage1,
                x,
                self.T,
                self.spike_schedule,
                self.first_layer_input_noise_sigma,
                self.first_layer_input_noise_type,
                self.first_layer_input_noise_position,
            )
        else:
            x = _forward_sequential_first_if_no_schedule(
                self.stage1,
                x,
                self.first_layer_input_noise_sigma,
                self.first_layer_input_noise_type,
                self.first_layer_input_noise_position,
            )
        x = self.stage2(x)
        x = self.stage3(x)
        conv4_3 = self.stage4(x)
        x = self.stage4_pool(conv4_3)
        conv7 = self.stage5(x)
        sources = [conv4_3, conv7]
        x = conv7
        for extra in self.extras:
            x = extra(x)
            sources.append(x)
        return sources

    def forward(self, images):
        sources = self._sources(images)
        loc = []
        conf = []
        for feature, loc_head, conf_head in zip(sources, self.loc_heads, self.conf_heads):
            loc.append(
                self._time_mean(loc_head(feature)).permute(0, 2, 3, 1).contiguous().flatten(1)
            )
            conf.append(
                self._time_mean(conf_head(feature)).permute(0, 2, 3, 1).contiguous().flatten(1)
            )
        loc = torch.cat(loc, dim=1).view(images.shape[0], -1, 4)
        conf = torch.cat(conf, dim=1).view(images.shape[0], -1, self.num_classes)
        return loc, conf


def ssd300_priors() -> torch.Tensor:
    priors = []
    for k, (f_k, min_size, max_size, aspects) in enumerate(
        zip(SOURCE_SHAPES, MIN_SIZES, MAX_SIZES, ASPECT_RATIOS)
    ):
        step = STEPS[k] / float(IMAGE_SIZE)
        min_s = min_size / float(IMAGE_SIZE)
        max_s = max_size / float(IMAGE_SIZE)
        for i in range(f_k):
            for j in range(f_k):
                cx = (j + 0.5) * step
                cy = (i + 0.5) * step
                priors.append((cx, cy, min_s, min_s))
                priors.append((cx, cy, max_s, max_s))
                for ratio in aspects:
                    r = float(ratio) ** 0.5
                    priors.append((cx, cy, min_s * r, min_s / r))
                    priors.append((cx, cy, min_s / r, min_s * r))
    boxes = torch.tensor(priors, dtype=torch.float32)
    boxes[:, :2].clamp_(min=0.0, max=1.0)
    return boxes


def _load_vgg16_bn_features():
    """Load VGG-16 BN features, offline-first for compute nodes without internet."""
    import os
    from torchvision.models import vgg16_bn

    torch_home = os.environ.get(
        "TORCH_HOME",
        os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "torch"),
    )
    hub_dir = os.path.join(torch_home, "hub", "checkpoints")
    candidates = [
        os.path.join(hub_dir, "vgg16_bn-6c64b313.pth"),
        os.path.join(torch_home, "checkpoints", "vgg16_bn-6c64b313.pth"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            print(f"[SSD] loading VGG-16 BN from cache: {path}", flush=True)
            state = torch.load(path, map_location="cpu")
            model = vgg16_bn()
            model.load_state_dict(state)
            return model.features
    try:
        from torchvision.models import VGG16_BN_Weights

        return vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1).features
    except Exception:
        return vgg16_bn(pretrained=True).features


def load_vgg16_bn_into_ssd(model: SSD300VGG16) -> int:
    """Copy ImageNet VGG-16 BN conv/BN weights into the serial backbone."""
    src = _load_vgg16_bn_features()
    dst_convs = []
    dst_bns = []
    for stage in (model.stage1, model.stage2, model.stage3, model.stage4, model.stage5):
        modules = list(stage)
        for i, module in enumerate(modules):
            if not (
                isinstance(module, nn.Conv2d)
                and module.kernel_size == (3, 3)
                and module.dilation == (1, 1)
            ):
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
                if dst.bias is not None and src_m.bias is not None:
                    dst.bias.copy_(src_m.bias)
        for dst, src_m in zip(dst_bns[:n], src_bns[:n]):
            if dst.weight.shape == src_m.weight.shape:
                dst.weight.copy_(src_m.weight)
                dst.bias.copy_(src_m.bias)
                dst.running_mean.copy_(src_m.running_mean)
                dst.running_var.copy_(src_m.running_var)
    return n


def decode_boxes(loc, priors, variances=(0.1, 0.2)):
    cx = loc[..., 0] * variances[0] * priors[:, 2] + priors[:, 0]
    cy = loc[..., 1] * variances[0] * priors[:, 3] + priors[:, 1]
    w = torch.exp(loc[..., 2] * variances[1]) * priors[:, 2]
    h = torch.exp(loc[..., 3] * variances[1]) * priors[:, 3]
    boxes = torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)
    return boxes.clamp(min=0.0, max=1.0)


def encode_boxes(matched, priors, variances=(0.1, 0.2)):
    g_cx = ((matched[:, 0] + matched[:, 2]) / 2 - priors[:, 0]) / (variances[0] * priors[:, 2])
    g_cy = ((matched[:, 1] + matched[:, 3]) / 2 - priors[:, 1]) / (variances[0] * priors[:, 3])
    g_w = torch.log((matched[:, 2] - matched[:, 0]).clamp(min=1e-6) / priors[:, 2]) / variances[1]
    g_h = torch.log((matched[:, 3] - matched[:, 1]).clamp(min=1e-6) / priors[:, 3]) / variances[1]
    return torch.stack((g_cx, g_cy, g_w, g_h), dim=1)


def nms(boxes, scores, iou_thresh=0.45, top_k=200):
    keep = []
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)
    x1, y1, x2, y2 = boxes.unbind(1)
    area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    _, order = scores.sort(descending=True)
    order = order[:top_k]
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (area[i] + area[rest] - inter + 1e-6)
        order = rest[iou <= iou_thresh]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)
