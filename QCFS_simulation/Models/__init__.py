import re

from .layer import *
from .cnn_mnist import cnn2_mnist
from .fc_mnist import fc2_mnist, fc3_mnist, fc3rev_mnist
from .toy_diff1d import toy_diff1d
from .VGG import vgg16, vgg19, vgg16_wobn
from .ResNet import resnet18, resnet18_imagenet, resnet34, resnet34_imagenet


def _parse_mnist_cnn2_variant(model_name: str):
    m = model_name.lower()
    if m in ("cnn2", "cnn2_mnist"):
        return 2, 4
    match = re.fullmatch(r"cnn2(?:_mnist)?_c(\d+)_c(\d+)", m)
    if match:
        return int(match.group(1)), int(match.group(2))
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


def modelpool(model_name, dataset_name="mnist"):
    m = model_name.lower()
    d = dataset_name.lower().replace("-", "").replace("_", "")

    if d in ("diff1d", "toydiff1d"):
        return toy_diff1d()

    if d in ("mnist", "fashionmnist"):
        channels = _parse_mnist_cnn2_variant(model_name)
        if channels is not None:
            c1, c2 = channels
            return cnn2_mnist(num_classes=10, c1=c1, c2=c2)
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
            "MNIST 当前仅支持模型: cnn2/cnn2_c{c1}_c{c2}/fc2/fc2_h{dim}/fc3/fc3_h{dim}/fc3rev/fc3rev_h{dim}"
        )

    if d in ("cifar10", "cifa10"):
        num_classes = 10
    elif d == "cifar100":
        num_classes = 100
    elif d in ("imagenet", "imagenet1k"):
        num_classes = 1000
    else:
        raise ValueError("未知数据集: %s" % (dataset_name,))

    if m in ("cnn2", "cnn2_mnist"):
        raise ValueError(
            "数据集 %s 请使用 VGG（-arch vgg16 / vgg19 / vgg16_wobn），勿用 cnn2"
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
    if m == "vgg16":
        return vgg16(num_classes=num_classes, dropout=dropout)
    if m == "vgg16_wobn":
        return vgg16_wobn(num_classes=num_classes, dropout=0.1)
    if m == "vgg19":
        return vgg19(num_classes=num_classes, dropout=dropout)
    raise ValueError(
        "数据集 %s 下支持的模型: vgg16 | vgg16_wobn | vgg19 | resnet18 | resnet34，收到: %s"
        % (dataset_name, model_name)
    )
