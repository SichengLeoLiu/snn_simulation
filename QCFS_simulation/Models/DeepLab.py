"""DeepLabV3-ResNet50 with unique QCFS/IF neurons.

TorchVision Bottleneck reuses one nn.ReLU three times. Replacing that module
in-place would share one IF (and one λ) across pre-activations and the
post-add residual. Each use is therefore a separate IF: if1, if2, if3.

Remaining ReLUs (stem, ASPP, DeepLab head) become IF in place. The final
1x1 classifier has no IF and is excluded from MNE, same as FCN-32s score_fr.

This is QCFS-style replacement plus optional short fine-tuning, not a
bit-exact copy of Bu et al. CVPR 2025 training-free conversion.
"""
from __future__ import annotations

import os
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.layer import ExpandTemporalDim, IF, MergeTemporalDim, add_dimention
from Models.VGG import NOISE_POSITIONS, _inject_noise_tensor

NUM_CLASSES = 21
COCO_WEIGHT_NAME = "deeplabv3_resnet50_coco-cd0a2569.pth"


def _is_tv_bottleneck(module: nn.Module) -> bool:
    return all(
        hasattr(module, name)
        for name in ("conv1", "bn1", "conv2", "bn2", "conv3", "bn3", "relu")
    ) and not hasattr(module, "residual_function")


def convert_bottleneck_relu_to_if(block: nn.Module) -> None:
    if hasattr(block, "if1") and isinstance(block.if1, IF):
        return
    block.if1 = IF()
    block.if2 = IF()
    block.if3 = IF()

    def forward(self, x):
        identity = x
        out = self.if1(self.bn1(self.conv1(x)))
        out = self.if2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.if3(out + identity)

    block.forward = types.MethodType(forward, block)
    block.relu = nn.Identity()


def _replace_relus(module: nn.Module) -> None:
    if isinstance(module, nn.ModuleDict):
        for key in list(module.keys()):
            child = module[key]
            if isinstance(child, nn.ReLU):
                module[key] = IF()
            else:
                _replace_relus(child)
        return
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ReLU):
            setattr(module, name, IF())
        else:
            _replace_relus(child)


def convert_deeplab_relu_to_if(root: nn.Module) -> None:
    for module in root.modules():
        if _is_tv_bottleneck(module):
            convert_bottleneck_relu_to_if(module)
    _replace_relus(root)


def _hub_dir() -> Path:
    torch_home = os.environ.get(
        "TORCH_HOME",
        os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "torch"),
    )
    return Path(torch_home) / "hub" / "checkpoints"


def cached_deeplab_weight_path() -> Path | None:
    primary = _hub_dir() / COCO_WEIGHT_NAME
    if primary.is_file():
        return primary
    matches = sorted(_hub_dir().glob("deeplabv3_resnet50_coco-*.pth"))
    return matches[0] if matches else None


def _build_tv_deeplab(load_coco: bool) -> nn.Module:
    from torchvision.models.segmentation import deeplabv3_resnet50

    if not load_coco:
        try:
            return deeplabv3_resnet50(
                weights=None, weights_backbone=None, aux_loss=False, num_classes=NUM_CLASSES
            )
        except TypeError:
            return deeplabv3_resnet50(pretrained=False, pretrained_backbone=False, num_classes=NUM_CLASSES)

    cache = cached_deeplab_weight_path()
    if cache is not None:
        print(f"[DeepLabV3] loading COCO-VOC weights from cache: {cache}", flush=True)
        try:
            model = deeplabv3_resnet50(
                weights=None, weights_backbone=None, aux_loss=True, num_classes=NUM_CLASSES
            )
        except TypeError:
            model = deeplabv3_resnet50(pretrained=False, num_classes=NUM_CLASSES)
        state = torch.load(cache, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        model.aux_classifier = None
        return model

    try:
        from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

        print("[DeepLabV3] loading COCO-VOC weights from torchvision", flush=True)
        model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
    except Exception:
        print("[DeepLabV3] loading COCO-VOC weights via pretrained=True", flush=True)
        model = deeplabv3_resnet50(pretrained=True)
    model.aux_classifier = None
    return model


class DeepLabV3ResNet50IF(nn.Module):
    def __init__(self, tv_model: nn.Module):
        super().__init__()
        tv_model.aux_classifier = None
        convert_deeplab_relu_to_if(tv_model)
        self.backbone = tv_model.backbone
        self.classifier = tv_model.classifier
        self.num_classes = NUM_CLASSES
        self.T = 0
        self.merge = MergeTemporalDim(0)
        self.expand = ExpandTemporalDim(0)
        self.first_layer_input_noise_sigma = 0.0
        self.first_layer_input_noise_type = "gaussian"
        self.first_layer_input_noise_position = "post_input_if"
        self._mne_layer_map = "resnet"

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

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

    def _inject(self, x):
        return _inject_noise_tensor(
            x,
            self.first_layer_input_noise_sigma,
            self.first_layer_input_noise_type,
            self.T,
        )

    def _time_mean(self, tensor):
        if self.T <= 0:
            return tensor
        batch = tensor.shape[0] // self.T
        return tensor.view(self.T, batch, *tensor.shape[1:]).mean(0)

    def _forward_backbone(self, x):
        pos = self.first_layer_input_noise_position
        out = None
        return_layers = getattr(self.backbone, "return_layers", {})
        for name, module in self.backbone.items():
            if name == "conv1" and pos == "pre_first_conv":
                x = self._inject(x)
            if name == "relu" and pos == "pre_input_if":
                x = self._inject(x)
            x = module(x)
            if name == "relu" and pos == "post_input_if":
                x = self._inject(x)
            if name in return_layers:
                out = x
        return x if out is None else out

    def forward(self, images):
        height, width = images.shape[-2:]
        x = images
        if self.T > 0:
            x = add_dimention(images, self.T)
            x = self.merge(x)
        logits = self.classifier(self._forward_backbone(x))
        logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        return self._time_mean(logits)


def build_deeplabv3_resnet50_if(load_coco: bool = True) -> DeepLabV3ResNet50IF:
    return DeepLabV3ResNet50IF(_build_tv_deeplab(load_coco=load_coco))
