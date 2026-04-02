from .layer import *
from .cnn_mnist import cnn2_mnist
from .VGG import vgg16, vgg19, vgg16_wobn


def modelpool(model_name, dataset_name="mnist"):
    m = model_name.lower()
    d = dataset_name.lower().replace("-", "")

    if d == "mnist":
        if m in ("cnn2", "cnn2_mnist"):
            return cnn2_mnist(num_classes=10)
        raise ValueError("MNIST 当前仅支持模型: cnn2")

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
    if m == "vgg16":
        return vgg16(num_classes=num_classes, dropout=dropout)
    if m == "vgg16_wobn":
        return vgg16_wobn(num_classes=num_classes, dropout=0.1)
    if m == "vgg19":
        return vgg19(num_classes=num_classes, dropout=dropout)
    raise ValueError(
        "数据集 %s 下支持的模型: vgg16 | vgg16_wobn | vgg19，收到: %s"
        % (dataset_name, model_name)
    )
