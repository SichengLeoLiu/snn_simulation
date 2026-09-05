import re

from .layer import *
from .cnn_mnist import (
    cnn2_mnist,
    cnn4_mnist,
    cnn6_mnist,
    cnn8_mnist,
    cnn10_mnist,
    cnn6_vgg_mnist,
    cnn6_narrow_staged_mnist,
    cnn6_wide_early_mnist,
)
from .fc_mnist import fc2_mnist, fc3_mnist, fc3rev_mnist
from .fc_cifar import fc2_cifar, fc3_cifar, fc5_cifar
from .toy_diff1d import toy_diff1d
from .VGG import vgg11, vgg13, vgg16, vgg19, vgg16_wobn, vgg16_inputif
from .ResNet import resnet18, resnet18_imagenet, resnet34, resnet34_imagenet
from .SSD import SSD300VGG16


def _parse_mnist_cnn2_variant(model_name: str):
    m = model_name.lower()
    if m in ("cnn2", "cnn2_mnist"):
        return 2, 4, "max"
    if m in ("cnn2_avg", "cnn2_avgpool", "cnn2_mnist_avg", "cnn2_mnist_avgpool"):
        return 2, 4, "avg"
    match = re.fullmatch(r"cnn2(?:_mnist)?_avg(?:pool)?_c(\d+)_c(\d+)", m)
    if match:
        return int(match.group(1)), int(match.group(2)), "avg"
    match = re.fullmatch(r"cnn2(?:_mnist)?_c(\d+)_c(\d+)", m)
    if match:
        return int(match.group(1)), int(match.group(2)), "max"
    return None


def _parse_mnist_deep_cnn_variant(model_name: str, depth: int):
    m = model_name.lower()
    if depth not in (4, 6, 8, 10):
        raise ValueError(f"unsupported deep CNN depth: {depth}")
    defaults = (2,) + (4,) * (depth - 1)

    if m in (f"cnn{depth}", f"cnn{depth}_mnist"):
        return defaults
    channel_pattern = "".join(r"_c(\d+)" for _ in range(depth))
    match = re.fullmatch(rf"cnn{depth}(?:_mnist)?{channel_pattern}", m)
    if match:
        return tuple(int(value) for value in match.groups())
    return None


def _parse_mnist_fc2_variant(model_name: str):
    m = model_name.lower()
    if m in ("fc2", "fc2_mnist", "mlp2", "mlp2_mnist"):
        return 256
    match = re.fullmatch(r"(?:fc2|mlp2)(?:_mnist)?_h(\d+)", m)
    if match:
        return int(match.group(1))
    return None


def _parse_mnist_fc3_variant(model_name: str):
    m = model_name.lower()
    if m in ("fc3", "fc3_mnist", "mlp3", "mlp3_mnist"):
        return 64
    match = re.fullmatch(r"(?:fc3|mlp3)(?:_mnist)?_h(\d+)", m)
    if match:
        return int(match.group(1))
    return None


def _parse_mnist_fc3rev_variant(model_name: str):
    m = model_name.lower()
    if m in ("fc3rev", "fc3rev_mnist", "mlp3rev", "mlp3rev_mnist"):
        return 64
    match = re.fullmatch(r"(?:fc3rev|mlp3rev)(?:_mnist)?_h(\d+)", m)
    if match:
        return int(match.group(1))
    return None


def _parse_cifar_fc2_variant(model_name: str):
    m = model_name.lower()
    if m in ("fc2_cifar", "mlp2_cifar", "fc2", "mlp2"):
        return 512
    match = re.fullmatch(r"(?:fc2|mlp2)(?:_cifar)?_h(\d+)", m)
    if match:
        return int(match.group(1))
    return None


def _parse_cifar_fc3_variant(model_name: str):
    m = model_name.lower()
    if m in ("fc3_cifar", "mlp3_cifar", "fc3", "mlp3"):
        return 512
    match = re.fullmatch(r"(?:fc3|mlp3)(?:_cifar)?_h(\d+)", m)
    if match:
        return int(match.group(1))
    return None


def _parse_cifar_fc5_variant(model_name: str):
    """
    Deep tapering BN-free CIFAR MLP.
    Names: fc5_cifar / mlp_cifar / fc5_cifar_w{mult}
    width_mult scales 1024-512-256-128 (default 1.0).
    """
    m = model_name.lower()
    if m in ("fc5_cifar", "mlp_cifar", "mlp5_cifar", "fc5"):
        return 1.0
    match = re.fullmatch(r"(?:fc5|mlp5|mlp)(?:_cifar)?_w(\d+(?:\.\d+)?)", m)
    if match:
        return float(match.group(1))
    return None


def modelpool(model_name, dataset_name="mnist"):
    m = model_name.lower()
    d = dataset_name.lower().replace("-", "").replace("_", "")

    if d in ("diff1d", "toydiff1d"):
        return toy_diff1d()

    if d in ("mnist", "fashionmnist"):
        channels = _parse_mnist_cnn2_variant(model_name)
        if channels is not None:
            c1, c2, pool = channels
            return cnn2_mnist(num_classes=10, c1=c1, c2=c2, pool=pool)
        if m in ("cnn6_vgg", "cnn6_vgg_mnist"):
            return cnn6_vgg_mnist(num_classes=10)
        if m in ("cnn6_narrow_staged", "cnn6_narrow_staged_mnist"):
            return cnn6_narrow_staged_mnist(num_classes=10)
        if m in ("cnn6_wide_early", "cnn6_wide_early_mnist"):
            return cnn6_wide_early_mnist(num_classes=10)
        deep_factories = {
            4: cnn4_mnist,
            6: cnn6_mnist,
            8: cnn8_mnist,
            10: cnn10_mnist,
        }
        for depth, factory in deep_factories.items():
            channels = _parse_mnist_deep_cnn_variant(model_name, depth=depth)
            if channels is not None:
                return factory(num_classes=10, channels=channels)
        hidden_dim3rev = _parse_mnist_fc3rev_variant(model_name)
        if hidden_dim3rev is not None:
            return fc3rev_mnist(num_classes=10, hidden_dim=hidden_dim3rev)
        hidden_dim3 = _parse_mnist_fc3_variant(model_name)
        if hidden_dim3 is not None:
            return fc3_mnist(num_classes=10, hidden_dim=hidden_dim3)
        hidden_dim = _parse_mnist_fc2_variant(model_name)
        if hidden_dim is not None:
            return fc2_mnist(num_classes=10, hidden_dim=hidden_dim)
        raise ValueError(
            "MNIST 当前仅支持模型: cnn2/cnn2_avg/cnn4/cnn6/cnn6_vgg/cnn6_narrow_staged/cnn6_wide_early/cnn8/cnn10（可用 _c{n} 指定每层通道）/fc2/fc3/fc3rev"
        )

    if d in ("voc", "voc2007", "voc0712", "pascalvoc", "voc20072012", "voc2012"):
        from .FCN import FCN32sVGG16, FCNVGG16

        if m in ("ssd300", "ssd300_vgg16", "ssd300vgg16"):
            return SSD300VGG16()
        if m in ("fcn32s", "fcn32s_vgg16", "fcn32s_nobn"):
            return FCN32sVGG16()
        if m in ("fcn", "fcn_vgg16", "fcn32", "vgg16"):
            return FCNVGG16()
        if m in ("deeplabv3", "deeplab", "deeplabv3_resnet50"):
            from .DeepLab import build_deeplabv3_resnet50_if

            return build_deeplabv3_resnet50_if(load_coco=False)
        raise ValueError(
            "VOC 当前仅支持模型: ssd300 / fcn / fcn_vgg16 / fcn32s / deeplabv3_resnet50，收到: %s"
            % (model_name,)
        )

    if d in ("cifar10", "cifa10"):
        num_classes = 10
    elif d == "cifar100":
        num_classes = 100
    elif d in ("imagenet", "imagenet1k"):
        num_classes = 1000
    else:
        raise ValueError("未知数据集: %s" % (dataset_name,))

    if d in ("cifar10", "cifa10", "cifar100"):
        # Prefer deep tapering MLP first (recommended default: fc5_cifar).
        width_mult = _parse_cifar_fc5_variant(model_name)
        if width_mult is not None:
            return fc5_cifar(num_classes=num_classes, width_mult=width_mult)
        hidden3 = _parse_cifar_fc3_variant(model_name)
        if hidden3 is not None:
            return fc3_cifar(num_classes=num_classes, hidden_dim=hidden3)
        hidden2 = _parse_cifar_fc2_variant(model_name)
        if hidden2 is not None:
            return fc2_cifar(num_classes=num_classes, hidden_dim=hidden2)

    if m in ("cnn2", "cnn2_mnist"):
        raise ValueError(
            "数据集 %s 请使用 VGG（-arch vgg16 / vgg19 / vgg16_wobn）或 BN-free MLP（fc5_cifar / fc3_cifar / fc2_cifar），勿用 cnn2"
            % (dataset_name,)
        )
    dropout = 0.5 if d in ("cifar10", "cifar100") else 0.0
    if m in ("resnet18", "resnet18_imagenet"):
        if d in ("imagenet", "imagenet1k"):
            return resnet18_imagenet(num_classes=num_classes)
        return resnet18(num_classes=num_classes)
    if m in ("resnet34", "resnet34_imagenet"):
        if d in ("imagenet", "imagenet1k"):
            return resnet34_imagenet(num_classes=num_classes)
        return resnet34(num_classes=num_classes)
    if m == "vgg11":
        return vgg11(num_classes=num_classes, dropout=dropout)
    if m == "vgg13":
        return vgg13(num_classes=num_classes, dropout=dropout)
    if m == "vgg16":
        return vgg16(num_classes=num_classes, dropout=dropout)
    if m in ("vgg16_inputif", "vgg16_preconv_if"):
        return vgg16_inputif(num_classes=num_classes, dropout=dropout)
    if m == "vgg16_wobn":
        return vgg16_wobn(num_classes=num_classes, dropout=0.1)
    if m == "vgg19":
        return vgg19(num_classes=num_classes, dropout=dropout)
    raise ValueError(
        "数据集 %s 下支持的模型: vgg11 | vgg13 | vgg16 | vgg16_inputif | vgg16_wobn | vgg19 | resnet18 | resnet34 | "
        "fc5_cifar[_wMULT] | fc3_cifar[_hN] | fc2_cifar[_hN]，收到: %s"
        % (dataset_name, model_name)
    )
