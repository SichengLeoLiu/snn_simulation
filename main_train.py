import argparse
import os
import torch
import torch.nn as nn
import torch.optim
from Models import modelpool
from Models.spike_temporal_adjust import SPIKE_SCHEDULE_MODES
from Preprocess import datapool
from utils import (
    train,
    val,
    train_reg,
    val_reg,
    seed_all,
    get_logger,
    get_torch_device,
    compute_mne_l2_regularization,
)

DATASET_CHOICES = ["mnist", "cifar10", "cifar100", "diff1d"]

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
    help="mnist: cnn2；cifar: vgg16/…；diff1d: x1>=x2∈[0,1]、y=x1-x2",
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
parser.add_argument(
    "--regularizer",
    default="weight_decay",
    type=str,
    choices=["weight_decay", "resolution_aware", "mne_l2"],
    help="正则方式：weight_decay（默认）| resolution_aware | mne_l2",
)
parser.add_argument(
    "--reg_coeff",
    default=1.0,
    type=float,
    help="--regularizer=resolution_aware 或 mne_l2 时的全局系数 beta",
)
parser.add_argument(
    "--mne_eps",
    default=1e-6,
    type=float,
    help="--regularizer=mne_l2 时的 eps（用于 BN-fold 与 lambda 分母）",
)
parser.add_argument(
    "--mne_use_max",
    action="store_true",
    help="--regularizer=mne_l2 时使用保守版 M_eff=max_o ||W_tilde_o||^2",
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
    help="CNN2/VGG/diff1d：T>0 时第一层 IF 后脉冲时间重排模式（与 main_test 一致）",
)

args = parser.parse_args()


def _resolved_model_name(dataset, model):
    m = model.lower()
    d = dataset.lower().replace("-", "").replace("_", "")
    if d in ("diff1d", "toydiff1d"):
        return "diff1d"
    if d != "mnist" and m in ("cnn2", "cnn2_mnist"):
        return "vgg16"
    return model


def _log_diff1d_param_values(model, logger):
    """训练结束后仅输出 diff1d 模型各层 weight 的具体数值（写入日志并打印到控制台）。"""
    logger.info("训练结束，diff1d 模型 weight 数值（detach 至 CPU）:")
    for name, p in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        logger.info("  %s  shape=%s\n%s", name, tuple(p.shape), p.detach().cpu())


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
        print("提示: 已用 arch=%s 构建与保存（与 -arch 输入不同）" % (arch,))
    model = modelpool(arch, args.dataset)
    model.set_L(args.L)
    model.set_T(args.time)
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule(args.spike_schedule)

    log_ds = "diff1d" if ds.replace("_", "") in ("diff1d", "toydiff1d") else ds
    log_dir = "%s-checkpoints" % log_ds
    os.makedirs(log_dir, exist_ok=True)

    model.to(device)

    reg_loss_fn = None

    is_diff1d = log_ds == "diff1d"
    if is_diff1d:
        criterion = nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=(args.weight_decay if args.regularizer == "weight_decay" else 0.0),
        )
    else:
        criterion = nn.CrossEntropyLoss().to(device)
        if ds == "mnist":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                weight_decay=(args.weight_decay if args.regularizer == "weight_decay" else 0.0),
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=0.9,
                weight_decay=(args.weight_decay if args.regularizer == "weight_decay" else 0.0),
            )

    if args.regularizer == "resolution_aware":
        if hasattr(model, "resolution_aware_noise_regularization"):
            reg_loss_fn = lambda m, t: m.resolution_aware_noise_regularization(T=t)
        else:
            raise ValueError(
                "模型 %s 不支持 resolution_aware 正则（缺少 resolution_aware_noise_regularization）"
                % (arch,)
            )
    elif args.regularizer == "mne_l2":
        reg_loss_fn = lambda m, t: compute_mne_l2_regularization(
            m,
            quant_level=args.L,
            eps=args.mne_eps,
            use_max=args.mne_use_max,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    best_acc = 0.0
    best_rmse = float("inf")

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
    logger.info(
        "regularizer=%s, weight_decay=%.6g, reg_coeff=%.6g"
        % (args.regularizer, args.weight_decay, args.reg_coeff)
    )
    if args.regularizer == "mne_l2":
        logger.info(
            "mne_l2: L=%d, eps=%.3e, use_max=%s"
            % (args.L, args.mne_eps, str(bool(args.mne_use_max)))
        )
    if ds not in ("mnist", "diff1d", "toy_diff1d", "diff_1d"):
        logger.info(
            "CIFAR 建议: -lr 0.1 -wd 5e-4 --epochs 300 -b 128"
        )
    if is_diff1d:
        logger.info(
            "diff1d：回归 y=x1-x2（数据上 x1>=x2）；Linear 无 bias、写死差分；指标为 RMSE"
        )

    for epoch in range(args.epochs):
        if is_diff1d:
            loss, mae = train_reg(
                model,
                device,
                train_loader,
                criterion,
                optimizer,
                args.time,
                reg_loss_fn=reg_loss_fn,
                reg_coeff=args.reg_coeff,
            )
            logger.info(
                "Epoch:[{}/{}]\t loss(sum)={:.5f}\t train_MAE={:.6f}".format(
                    epoch, args.epochs, loss, mae
                )
            )
            scheduler.step()
            tmp = val_reg(
                model, test_loader, T=args.time, device=device
            )
            logger.info(
                "Epoch:[{}/{}]\t Test RMSE={:.6f}\n".format(
                    epoch, args.epochs, tmp
                )
            )
            if tmp < best_rmse:
                best_rmse = tmp
                filename = os.path.join(log_dir, "%s.pth" % (identifier,))
                print("Saving model to %s" % (filename,))
                torch.save(model.state_dict(), filename)
        else:
            loss, acc = train(
                model,
                device,
                train_loader,
                criterion,
                optimizer,
                args.time,
                reg_loss_fn=reg_loss_fn,
                reg_coeff=args.reg_coeff,
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

    if is_diff1d:
        logger.info("Best Test RMSE={:.6f}".format(best_rmse))
        _log_diff1d_param_values(model, logger)
    else:
        logger.info("Best Test acc={:.3f}".format(best_acc))


if __name__ == "__main__":
    main()
