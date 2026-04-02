import torch

from .getdataloader import GetMNIST, GetCifar10, GetCifar100, GetImageNet


def datapool(
    dataset_name,
    batch_size=128,
    dist_sample=False,
    num_workers=4,
    pin_memory=None,
):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    name = dataset_name.lower().replace("-", "")
    if name == "mnist":
        return GetMNIST(
            batch_size, num_workers=num_workers, pin_memory=pin_memory
        )
    if name in ("cifar10", "cifa10"):
        return GetCifar10(
            batch_size, num_workers=num_workers, pin_memory=pin_memory
        )
    if name == "cifar100":
        return GetCifar100(
            batch_size, num_workers=num_workers, pin_memory=pin_memory
        )
    if name in ("imagenet", "imagenet1k"):
        return GetImageNet(
            train_batch_size=batch_size,
            val_batch_size=min(batch_size, 64),
            workers=num_workers,
            dist_sample=dist_sample,
        )
    raise ValueError(
        "datapool 支持: mnist | cifar10 | cifar100 | imagenet，收到: %s"
        % (dataset_name,)
    )
