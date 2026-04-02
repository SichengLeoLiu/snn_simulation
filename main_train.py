import argparse
import os
import torch
import torch.nn as nn
import torch.optim
from Models import modelpool
from Models.spike_temporal_adjust import SPIKE_SCHEDULE_MODES
from Preprocess import datapool
from utils import train, val, seed_all, get_logger, get_torch_device

DATASET_CHOICES = ["mnist", "cifar10", "cifar100"]

parser = argparse.ArgumentParser(
    description="训练（MNIST: CNN2；CIFAR: VGG 等）"
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="数据加载进程数",
)
parser.add_argument(
    "-b", "--batch_size", default=128, type=int, metavar="N", help="batch 大小"
)
parser.add_argument("--seed", default=42, type=int, help="随机种子")
parser.add_argument("-suffix", "--suffix", default="", type=str, help="日志/权重后缀")
parser.add_argument("-T", "--time", default=0, type=int, help="SNN 时间步 T，0 为 ANN 模式")
parser.add_argument(
    "-data",
    "--dataset",
    default="mnist",
    type=str,
    choices=DATASET_CHOICES,
    help="数据集",
)
parser.add_argument(
    "-arch",
    "--model",
    default="cnn2",
    type=str,
    help="mnist: cnn2；cifar: vgg16 / vgg19 / vgg16_wobn",
)
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N", help="训练轮数（CIFAR 常用 300）"
)
parser.add_argument(
    "-lr",
    "--lr",
    default=0.01,
    type=float,
    metavar="LR",
    help="初始学习率（CIFAR+SGD 常用 0.1）",
)
parser.add_argument(
    "-wd",
    "--weight_decay",
    default=0.0,
    type=float,
    help="权重衰减（CIFAR 常用 5e-4）",
)
parser.add_argument("-L", "--L", default=8, type=int, help="量化步数 L")
parser.add_argument(
    "-dev",
    "--device",
    default="auto",
    type=str,
    help="计算设备: auto（cuda>mps>cpu）| mps | cuda | cpu",
)
parser.add_argument(
    "--spike_schedule",
    default="normal",
    type=str,
    choices=sorted(SPIKE_SCHEDULE_MODES),
    help="仅 CNN2 等有 set_spike_schedule 的模型在训练中记录；VGG 无此项",
)

args = parser.parse_args()


def _resolved_model_name(dataset, model):
    m = model.lower()
    if dataset.lower() != "mnist" and m in ("cnn2", "cnn2_mnist"):
        return "vgg16"
    return model


def main():
    global args
    device = get_torch_device(args.device)
    print("device: %s" % (device,))
    seed_all(args.seed)

    ds = args.dataset.lower()
    train_loader, test_loader = datapool(
        args.dataset,
        args.batch_size,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    arch = _resolved_model_name(args.dataset, args.model)
    if arch != args.model:
        print("提示: CIFAR 不使用 cnn2，已改用 %s 构建与保存" % (arch,))
    model = modelpool(arch, args.dataset)
    model.set_L(args.L)
    model.set_T(args.time)
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule(args.spike_schedule)

    log_dir = "%s-checkpoints" % ds
    os.makedirs(log_dir, exist_ok=True)

    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    if ds == "mnist":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    best_acc = 0.0

    identifier = arch
    identifier += "_L[%d]" % (args.L,)
    if args.time > 0:
        identifier += "_T[%d]" % (args.time,)
    if args.suffix:
        identifier += "_%s" % (args.suffix,)

    logger = get_logger(os.path.join(log_dir, "%s.log" % (identifier,)))
    logger.info(
        "start training dataset=%s arch=%s T=%d" % (args.dataset, arch, args.time)
    )
    if ds != "mnist":
        logger.info(
            "CIFAR 建议: -lr 0.1 -wd 5e-4 --epochs 300 -b 128"
        )

    for epoch in range(args.epochs):
        loss, acc = train(
            model, device, train_loader, criterion, optimizer, args.time
        )
        logger.info(
            "Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}".format(
                epoch, args.epochs, loss, acc
            )
        )
        scheduler.step()
        tmp = val(model, test_loader, T=args.time, device=device)
        logger.info(
            "Epoch:[{}/{}]\t Test acc={:.3f}\n".format(
                epoch, args.epochs, tmp
            )
        )

        if best_acc < tmp:
            best_acc = tmp
            filename = os.path.join(log_dir, "%s.pth" % (identifier,))
            print("Saving model to %s" % (filename,))
            torch.save(model.state_dict(), filename)

    logger.info("Best Test acc={:.3f}".format(best_acc))


if __name__ == "__main__":
    main()
