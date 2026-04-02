import argparse
import os
import torch

from Models import modelpool
from Models.cnn_mnist import remap_legacy_cnn2_state_dict
from Models.VGG import remap_legacy_vgg_state_dict
from Models.spike_temporal_adjust import SPIKE_SCHEDULE_MODES
from Preprocess import datapool
from utils import val, seed_all, get_logger, calibrate_thresholds, get_torch_device

SPIKE_SCHEDULE_CHOICES = sorted(SPIKE_SCHEDULE_MODES) + ["all"]
DATASET_CHOICES = ["mnist", "cifar10", "cifar100"]

parser = argparse.ArgumentParser(
    description="测试（MNIST: CNN2；CIFAR: VGG 等）"
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="数据加载进程数",
)
parser.add_argument("-b", "--batch_size", default=128, type=int, help="batch 大小")
parser.add_argument("--seed", default=44, type=int, help="随机种子")
parser.add_argument("-suffix", "--suffix", default="", type=str, help="后缀")
parser.add_argument(
    "-data",
    "--dataset",
    default="mnist",
    type=str,
    choices=DATASET_CHOICES,
    help="数据集",
)
parser.add_argument(
    "-arch", "--model", default="cnn2", type=str, help="与训练时一致"
)
parser.add_argument(
    "-dev",
    "--device",
    default="auto",
    type=str,
    help="计算设备: auto | mps | cuda | cpu",
)
parser.add_argument("-T", "--time", default=0, type=int, help="SNN 时间步 T")
parser.add_argument("-L", "--L", default=8, type=int, help="量化步数 L")
parser.add_argument(
    "--scaling_factor", default=1.0, type=float, help="IF 缩放因子"
)
parser.add_argument("--mode", default="normal", type=str, help="IF 模式")
parser.add_argument("--calibrate", action="store_true", help="测试前阈值校准")
parser.add_argument("--calib_epochs", default=5, type=int, help="校准轮数")
parser.add_argument("--calib_lr", default=0.01, type=float, help="校准学习率")
parser.add_argument(
    "--calib_batch_size",
    default=None,
    type=int,
    help="校准 batch（默认与 -b 相同）",
)
parser.add_argument(
    "--calib_samples",
    default=None,
    type=int,
    help="校准样本上限",
)
parser.add_argument(
    "--calib_data",
    default=None,
    type=str,
    choices=DATASET_CHOICES,
    help="校准数据集（默认同 --dataset）",
)
parser.add_argument(
    "-w",
    "--weights",
    default=None,
    type=str,
    help="权重 .pth（默认 {dataset}-checkpoints 下按规则查找）",
)
parser.add_argument(
    "--spike_schedule",
    default="all",
    type=str,
    choices=SPIKE_SCHEDULE_CHOICES,
    help="T>0 且模型支持时多模式；否则只测一轮",
)
parser.add_argument(
    "--viz",
    action="store_true",
    help="仅 MNIST CNN2：IF 特征图",
)
parser.add_argument(
    "--viz_out_dir",
    type=str,
    default=None,
    help="默认可为 {dataset}-test-viz",
)
parser.add_argument(
    "--viz_batch_idx",
    type=int,
    default=0,
    metavar="K",
    help="--viz 用第 K 个 test batch（0 起）",
)
parser.add_argument(
    "--viz_num_show",
    type=int,
    default=6,
    help="--viz 每 batch 展示张数",
)
parser.add_argument(
    "--viz_diff_abs_max",
    type=float,
    default=None,
)
parser.add_argument("--viz_feat_vmin", type=float, default=None)
parser.add_argument("--viz_feat_vmax", type=float, default=None)

args = parser.parse_args()


def _resolved_model_name(dataset, model):
    m = model.lower()
    if dataset.lower() != "mnist" and m in ("cnn2", "cnn2_mnist"):
        return "vgg16"
    return model


def _checkpoint_candidates(model_dir, model, L, time_steps, suffix):
    base = "%s_L[%d]" % (model, L)

    def pth(name):
        return os.path.join(model_dir, name + ".pth")

    out = []
    if time_steps > 0:
        if suffix:
            out.append(pth(base + "_T[%d]_%s" % (time_steps, suffix)))
        out.append(pth(base + "_T[%d]" % (time_steps,)))
        if suffix:
            out.append(pth(base + "_%s" % (suffix,)))
        out.append(pth(base))
    else:
        if suffix:
            out.append(pth(base + "_%s" % (suffix,)))
        out.append(pth(base))
    seen, unique = set(), []
    for path in out:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def resolve_weight_path(args, model_dir, model_name):
    if args.weights:
        path = os.path.expanduser(args.weights)
        if not os.path.isfile(path):
            raise FileNotFoundError("未找到权重: %s" % (path,))
        return path
    for p in _checkpoint_candidates(
        model_dir, model_name, args.L, args.time, args.suffix
    ):
        if os.path.isfile(p):
            print("加载权重: %s" % (p,))
            return p
    tried = _checkpoint_candidates(
        model_dir, model_name, args.L, args.time, args.suffix
    )
    raise FileNotFoundError("未找到权重，已尝试:\n  " + "\n  ".join(tried))


def _schedules_to_run(args, model):
    if args.time == 0:
        return ["normal"]
    if not hasattr(model, "set_spike_schedule"):
        if args.spike_schedule != "all":
            print("注意: 模型无 set_spike_schedule，仅单次推理")
        else:
            print("注意: 忽略 --spike_schedule all，仅单次推理")
        return ["normal"]
    if args.spike_schedule == "all":
        return sorted(SPIKE_SCHEDULE_MODES)
    return [args.spike_schedule]


def main():
    global args
    print(args)

    ds = args.dataset.lower()
    log_dir = "%s-test-accuracy" % ds
    os.makedirs(log_dir, exist_ok=True)
    model_dir = "%s-checkpoints" % ds
    viz_out = args.viz_out_dir or ("%s-test-viz" % ds)

    arch = _resolved_model_name(args.dataset, args.model)
    if arch != args.model:
        print("提示: 权重检索使用 arch=%s" % (arch,))
    identifier = arch
    identifier += "_L[%d]" % (args.L,)
    if args.time > 0:
        identifier += "_T[%d]" % (args.time,)
    save_acc_filename = "%s_%s_L%s_T%s" % (
        ds,
        identifier.replace("[", "").replace("]", ""),
        args.L,
        args.time,
    )
    if args.suffix:
        save_acc_filename += "_%s" % (args.suffix,)
    if args.spike_schedule == "all":
        save_acc_filename += "_sch_all"
    elif args.spike_schedule != "normal":
        save_acc_filename += "_sch_%s" % (args.spike_schedule,)
    logger = get_logger(os.path.join(log_dir, "%s.log" % (save_acc_filename,)))

    device = get_torch_device(args.device)
    print("device: %s" % (device,))
    seed_all(args.seed)

    train_loader, test_loader = datapool(
        args.dataset,
        args.batch_size,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    model = modelpool(arch, args.dataset)

    weight_path = resolve_weight_path(args, model_dir, arch)
    logger.info("checkpoint: %s" % (weight_path,))
    state_dict = torch.load(weight_path, map_location="cpu")
    if ds == "mnist":
        state_dict, legacy = remap_legacy_cnn2_state_dict(state_dict)
        if legacy:
            logger.info("已兼容旧版 CNN2 checkpoint")
    else:
        state_dict = remap_legacy_vgg_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.set_T(args.time)
    model.set_L(args.L)
    if hasattr(model, "set_scaling_factor"):
        model.set_scaling_factor(args.scaling_factor)
    model.set_mode(args.mode)

    schedules = _schedules_to_run(args, model)
    if args.time == 0 and args.spike_schedule == "all":
        logger.info("T=0：spike_schedule 不参与，仅评测一次")
    elif hasattr(model, "set_spike_schedule"):
        logger.info(
            "将评测 spike_schedule: %d 种 — %s"
            % (len(schedules), ", ".join(schedules))
        )

    if args.calibrate:
        if args.time == 0:
            logger.warning("校准需 T>0，跳过")
        else:
            calib_ds = args.calib_data or args.dataset
            calib_bs = (
                args.calib_batch_size
                if args.calib_batch_size is not None
                else args.batch_size
            )
            calib_train_loader, _ = datapool(
                calib_ds,
                calib_bs,
                num_workers=args.workers,
                pin_memory=(device.type == "cuda"),
            )
            calib_dataset = calib_train_loader.dataset
            if args.calib_samples is not None:
                from torch.utils.data import Subset

                n = min(args.calib_samples, len(calib_dataset))
                idx = torch.randperm(len(calib_dataset))[:n]
                calib_dataset = Subset(calib_dataset, idx.tolist())
            calib_loader = torch.utils.data.DataLoader(
                calib_dataset,
                batch_size=calib_bs,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=(device.type == "cuda"),
            )
            model = calibrate_thresholds(
                model=model,
                calib_loader=calib_loader,
                device=device,
                epochs=args.calib_epochs,
                lr=args.calib_lr,
                verbose=True,
            )

    val_verbose = len(schedules) <= 1
    results = []
    for mode in schedules:
        if hasattr(model, "set_spike_schedule"):
            model.set_spike_schedule(mode)
        logger.info("spike_schedule=%s" % (mode,))
        acc = val(model, test_loader, args.time, device, verbose=val_verbose)
        results.append((mode, acc))
        print("spike_schedule=%s  Test acc = %.3f" % (mode, acc))
        logger.info("spike_schedule=%s  Test acc = %.3f" % (mode, acc))

    print("\n--- 汇总 dataset=%s T=%d ---" % (args.dataset, args.time))
    for mode, acc in results:
        print("  %-26s  %.3f" % (mode, acc))
    logger.info(
        "汇总: %s"
        % (", ".join("%s=%.3f" % (m, a) for m, a in results),)
    )

    if args.viz:
        if ds != "mnist":
            logger.warning("--viz 仅支持 dataset=mnist 的 CNN2")
        elif args.time <= 0:
            logger.warning("viz 需 -T>0")
        elif not hasattr(model, "forward_with_if_features"):
            logger.warning("viz 需 forward_with_if_features")
        else:
            from viz_cnn_mnist import save_cnn_mnist_feature_maps

            n_batches = len(test_loader)
            k = args.viz_batch_idx
            if k < 0 or k >= n_batches:
                logger.warning(
                    "viz_batch_idx=%d 无效，改用最后一 batch" % (k,)
                )
                k = n_batches - 1
            it = iter(test_loader)
            for _ in range(k):
                next(it)
            im, lb = next(it)
            nshow = min(args.viz_num_show, im.shape[0])
            ftag = identifier.replace("[", "").replace("]", "")
            os.makedirs(viz_out, exist_ok=True)
            save_cnn_mnist_feature_maps(
                model,
                im[:nshow],
                lb[:nshow],
                args.time,
                args.L,
                viz_out,
                file_tag=ftag,
                logger=logger,
                viz_diff_abs_max=args.viz_diff_abs_max,
                viz_feat_vmin=args.viz_feat_vmin,
                viz_feat_vmax=args.viz_feat_vmax,
            )
            if schedules and hasattr(model, "set_spike_schedule"):
                model.set_spike_schedule(schedules[-1])

    return results


if __name__ == "__main__":
    main()
