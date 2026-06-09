import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from Preprocess.augment import Cutout

MNIST_ROOT = os.path.expanduser(os.environ.get("MNIST_ROOT", "~/datasets"))
CIFAR_ROOT = os.path.expanduser(os.environ.get("CIFAR_ROOT", "~/datasets"))


def _cifar_pil_autoaugment():
    try:
        from torchvision.transforms import AutoAugment, AutoAugmentPolicy

        return AutoAugment(AutoAugmentPolicy.CIFAR10)
    except (ImportError, AttributeError):
        from Preprocess.augment import CIFAR10Policy

        return CIFAR10Policy()


def GetMNIST(batch_size, num_workers=4, pin_memory=None):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_data = datasets.MNIST(
        MNIST_ROOT, train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        MNIST_ROOT, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def GetCifar10(batchsize, num_workers=8, pin_memory=True, attack=False):
    aa = _cifar_pil_autoaugment()
    trans_t = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            aa,
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
            Cutout(n_holes=1, length=16),
        ]
    )
    if attack:
        trans = transforms.Compose([transforms.ToTensor()])
    else:
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    root = os.path.expanduser(CIFAR_ROOT)
    train_data = datasets.CIFAR10(
        root, train=True, transform=trans_t, download=True
    )
    test_data = datasets.CIFAR10(
        root, train=False, transform=trans, download=True
    )
    train_dataloader = DataLoader(
        train_data,
        batch_size=batchsize,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batchsize,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_dataloader, test_dataloader


def GetCifar100(batchsize, num_workers=8, pin_memory=True):
    aa = _cifar_pil_autoaugment()
    trans_t = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            aa,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[n / 255.0 for n in [129.3, 124.1, 112.4]],
                std=[n / 255.0 for n in [68.2, 65.4, 70.4]],
            ),
            Cutout(n_holes=1, length=16),
        ]
    )
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[n / 255.0 for n in [129.3, 124.1, 112.4]],
                std=[n / 255.0 for n in [68.2, 65.4, 70.4]],
            ),
        ]
    )
    root = os.path.expanduser(CIFAR_ROOT)
    train_data = datasets.CIFAR100(
        root, train=True, transform=trans_t, download=True
    )
    test_data = datasets.CIFAR100(
        root, train=False, transform=trans, download=True
    )
    train_dataloader = DataLoader(
        train_data,
        batch_size=batchsize,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batchsize,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=pin_memory,
    )
    return train_dataloader, test_dataloader


class Diff1DDataset(Dataset):
    """
    玩具回归：输入 [1, 2]（单通道、长度 2，无高度维）。
    采样满足 ``low <= x2 <= x1 <= high``（默认 ``[0,1]`` 上 ``x1 > x2`` 几乎处处成立），
    标签 ``y = x1 - x2``，∈ ``[0, high-low]``（在 ``[0,1]`` 上为 ``[0,1]``）。
    """

    def __init__(self, n_samples, low=0.0, high=1.0, seed=0):
        g = torch.Generator().manual_seed(seed)
        span = high - low
        x1 = torch.rand(n_samples, generator=g) * span + low
        u = torch.rand(n_samples, generator=g)
        x2 = low + u * (x1 - low)
        self.x = torch.stack([x1, x2], dim=-1).unsqueeze(1).float()
        self.y = (x1 - x2).unsqueeze(-1).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def GetDiff1D(
    batch_size,
    train_n=50000,
    test_n=5000,
    seed=0,
    low=0.0,
    high=1.0,
    num_workers=0,
    pin_memory=False,
):
    train_ds = Diff1DDataset(train_n, low=low, high=high, seed=seed)
    test_ds = Diff1DDataset(test_n, low=low, high=high, seed=seed + 100000)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


normalize_imagenet = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


class TransformedDataset(Dataset):
    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"].convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, item["label"]


def GetImageNet(
    input_size: int = 224,
    train_batch_size: int = 12,
    val_batch_size: int = 4,
    workers: int = 12,
    dist_sample: bool = False,
):
    from Preprocess.imagenet_hf_env import (
        configure_imagenet_hf_env,
        load_imagenet_dataset,
        resolve_imagenet_datasets_cache,
    )
    configure_imagenet_hf_env(verbose=False)
    cache_dir = str(resolve_imagenet_datasets_cache())

    ds = load_imagenet_dataset(cache_dir=cache_dir)
    train_ds, val_ds = ds["train"], ds["validation"]
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_imagenet,
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize_imagenet,
        ]
    )
    train_dataset = TransformedDataset(train_ds, train_transforms)
    val_dataset = TransformedDataset(val_ds, val_transforms)
    if dist_sample:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False
        )
    else:
        train_sampler = val_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=(train_sampler is None),
        num_workers=workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        sampler=val_sampler,
    )
    return train_loader, val_loader
