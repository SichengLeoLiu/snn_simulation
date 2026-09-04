"""FCN-32s VGG-16 BN-IF for VOC semantic segmentation.

Every learned Conv is Conv-BN-IF (encoder, decoder, 21-class head) so MNE-L2
covers the whole network. Train T=0; eval T=L with rate_uniform and
post-input-IF noise on stage1. Spike-count logits are bilinear-upsampled,
then per-pixel argmax.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.layer import ExpandTemporalDim, IF, MergeTemporalDim, add_dimention
from Models.VGG import (
    NOISE_POSITIONS,
    SPIKE_SCHEDULE_MODES,
    _forward_sequential_first_if_no_schedule,
    _forward_sequential_first_if_spike_schedule,
)

VOC_SEG_CLASSES = (
    "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
NUM_CLASSES = len(VOC_SEG_CLASSES)
IGNORE_INDEX = 255
STRIDE = 32


def _conv_bn_if(cin, cout, k=3, stride=1, padding=1):
    return [
        nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(cout),
        IF(),
    ]


class FCNVGG16(nn.Module):
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
            nn.AvgPool2d(2, 2),
        )
        self.stage4 = nn.Sequential(
            *_conv_bn_if(256, 512),
            *_conv_bn_if(512, 512),
            *_conv_bn_if(512, 512),
            nn.AvgPool2d(2, 2),
        )
        self.stage5 = nn.Sequential(
            *_conv_bn_if(512, 512),
            *_conv_bn_if(512, 512),
            *_conv_bn_if(512, 512),
            nn.AvgPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            *_conv_bn_if(512, 256),
            *_conv_bn_if(256, self.num_classes, k=1, padding=0),
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

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

    def forward(self, images):
        height, width = images.shape[-2:]
        if self.T > 0:
            x = add_dimention(images, self.T)
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
                images,
                self.first_layer_input_noise_sigma,
                self.first_layer_input_noise_type,
                self.first_layer_input_noise_position,
            )
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        logits = self._time_mean(self.decoder(x))
        return F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)


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
            print(f"[FCN] loading VGG-16 BN from cache: {path}", flush=True)
            state = torch.load(path, map_location="cpu")
            model = vgg16_bn()
            model.load_state_dict(state)
            return model.features
    try:
        from torchvision.models import VGG16_BN_Weights

        return vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1).features
    except Exception:
        return vgg16_bn(pretrained=True).features


def load_vgg16_bn_into_fcn(model: FCNVGG16) -> int:
    """Copy ImageNet VGG-16 BN conv/BN weights into the FCN encoder."""
    src = _load_vgg16_bn_features()
    dst_convs, dst_bns = [], []
    for stage in (model.stage1, model.stage2, model.stage3, model.stage4, model.stage5):
        modules = list(stage)
        for i, module in enumerate(modules):
            if not (
                isinstance(module, nn.Conv2d)
                and module.kernel_size == (3, 3)
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
        for dst, src_m in zip(dst_bns[:n], src_bns[:n]):
            if dst.weight.shape == src_m.weight.shape:
                dst.weight.copy_(src_m.weight)
                dst.bias.copy_(src_m.bias)
                dst.running_mean.copy_(src_m.running_mean)
                dst.running_var.copy_(src_m.running_var)
    return n


def _conv_if(cin, cout, k=3, stride=1, padding=1):
    return [
        nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=True),
        IF(),
    ]


class FCN32sVGG16(nn.Module):
    """Original FCN-32s VGG-16: no BN, MaxPool, fc6/fc7, linear score_fr.

    Hidden ReLU → QCFS/IF (15 IFs). score_fr has no IF (signed class logits).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
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

        self.stage1 = nn.Sequential(*_conv_if(3, 64), *_conv_if(64, 64), nn.MaxPool2d(2, 2))
        self.stage2 = nn.Sequential(*_conv_if(64, 128), *_conv_if(128, 128), nn.MaxPool2d(2, 2))
        self.stage3 = nn.Sequential(
            *_conv_if(128, 256), *_conv_if(256, 256), *_conv_if(256, 256), nn.MaxPool2d(2, 2)
        )
        self.stage4 = nn.Sequential(
            *_conv_if(256, 512), *_conv_if(512, 512), *_conv_if(512, 512), nn.MaxPool2d(2, 2)
        )
        self.stage5 = nn.Sequential(
            *_conv_if(512, 512), *_conv_if(512, 512), *_conv_if(512, 512), nn.MaxPool2d(2, 2)
        )
        self.fc6 = nn.Sequential(nn.Conv2d(512, 4096, 7, padding=3, bias=True), IF(), nn.Dropout(dropout))
        self.fc7 = nn.Sequential(nn.Conv2d(4096, 4096, 1, bias=True), IF(), nn.Dropout(dropout))
        self.classifier = nn.Conv2d(4096, self.num_classes, 1, bias=True)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
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

    def forward(self, images):
        height, width = images.shape[-2:]
        if self.T > 0:
            x = add_dimention(images, self.T)
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
                images,
                self.first_layer_input_noise_sigma,
                self.first_layer_input_noise_type,
                self.first_layer_input_noise_position,
            )
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        logits = self._time_mean(self.classifier(x))
        return F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)


def _load_vgg16():
    """Load VGG-16 (no BN), offline-first for compute nodes without internet."""
    import os
    from torchvision.models import vgg16

    torch_home = os.environ.get(
        "TORCH_HOME",
        os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "torch"),
    )
    hub_dir = os.path.join(torch_home, "hub", "checkpoints")
    candidates = [
        os.path.join(hub_dir, "vgg16-397923af.pth"),
        os.path.join(hub_dir, "vgg16-397923af.pth.tar"),
        os.path.join(torch_home, "checkpoints", "vgg16-397923af.pth"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            print(f"[FCN32s] loading VGG-16 from cache: {path}", flush=True)
            state = torch.load(path, map_location="cpu")
            model = vgg16()
            model.load_state_dict(state)
            return model
    try:
        from torchvision.models import VGG16_Weights

        print("[FCN32s] loading VGG-16 from torchvision weights", flush=True)
        return vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    except Exception:
        return vgg16(pretrained=True)


def load_vgg16_into_fcn32s(model: FCN32sVGG16) -> dict:
    """Copy ImageNet VGG-16 conv + fc6/fc7 weights. score_fr stays random."""
    src = _load_vgg16()
    dst_convs = []
    for stage in (model.stage1, model.stage2, model.stage3, model.stage4, model.stage5):
        dst_convs.extend(m for m in stage if isinstance(m, nn.Conv2d))
    src_convs = [m for m in src.features if isinstance(m, nn.Conv2d)]
    n_feat = min(13, len(dst_convs), len(src_convs))
    with torch.no_grad():
        for dst, src_m in zip(dst_convs[:n_feat], src_convs[:n_feat]):
            if dst.weight.shape == src_m.weight.shape:
                dst.weight.copy_(src_m.weight)
            if dst.bias is not None and src_m.bias is not None:
                dst.bias.copy_(src_m.bias)
        fc6 = next(m for m in model.fc6 if isinstance(m, nn.Conv2d))
        fc7 = next(m for m in model.fc7 if isinstance(m, nn.Conv2d))
        w6 = src.classifier[0].weight.view(4096, 512, 7, 7)
        fc6.weight.copy_(w6)
        fc6.bias.copy_(src.classifier[0].bias)
        w7 = src.classifier[3].weight.view(4096, 4096, 1, 1)
        fc7.weight.copy_(w7)
        fc7.bias.copy_(src.classifier[3].bias)
    return {"n_features": n_feat, "fc6": True, "fc7": True, "score_fr": False}


@torch.no_grad()
def init_if_thresholds_percentile(model, loader, device, q: float = 99.9, max_images: int = 64):
    """Set each IF thresh to the q-th percentile of ReLU(conv) activations."""
    model.eval()
    model.set_T(0)
    ifs = [m for m in model.modules() if isinstance(m, IF)]
    samples = {id(module): [] for module in ifs}

    def _hook(module, inputs, _output):
        values = F.relu(inputs[0].detach()).flatten()
        if values.numel() > 200_000:
            values = values[torch.randint(0, values.numel(), (200_000,), device=values.device)]
        samples[id(module)].append(values.float().cpu())

    handles = [module.register_forward_hook(_hook) for module in ifs]
    n = 0
    try:
        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            if not torch.is_tensor(images):
                images = images[0]
            images = images.to(device, non_blocking=True)
            if images.dim() == 3:
                images = images.unsqueeze(0)
            model(images)
            n += images.shape[0]
            if n >= max_images:
                break
    finally:
        for handle in handles:
            handle.remove()
    report = []
    for index, module in enumerate(ifs):
        chunks = samples[id(module)]
        if not chunks:
            report.append({"if": index, "thresh": float(module.thresh.detach()), "n": 0})
            continue
        values = torch.cat(chunks)
        thresh = float(torch.quantile(values, q / 100.0).clamp(min=1e-3))
        module.thresh.copy_(torch.tensor([thresh], dtype=module.thresh.dtype, device=module.thresh.device))
        report.append({"if": index, "thresh": thresh, "n": int(values.numel())})
    print(f"[IF-INIT] q={q} images={n} " + ", ".join(f"{r['if']}:{r['thresh']:.3g}" for r in report), flush=True)
    return report
