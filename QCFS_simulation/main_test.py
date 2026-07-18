import argparse
import csv
import os
import torch

from Models import modelpool
from Models.toy_diff1d import format_diff1d_trace
from Models.cnn_mnist import remap_legacy_cnn2_state_dict
from Models.VGG import remap_legacy_vgg_state_dict
from Models.spike_temporal_adjust import SPIKE_SCHEDULE_MODES
from Preprocess import datapool
from utils import val, val_reg, seed_all, get_logger, calibrate_thresholds, get_torch_device

SPIKE_SCHEDULE_CHOICES = sorted(SPIKE_SCHEDULE_MODES) + ["all"]
DATASET_CHOICES = ["mnist", "fashion_mnist", "cifar10", "cifar100", "imagenet", "diff1d"]

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
    help="校准数据集（默认同 --dataset；diff1d 一般不校准）",
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
    default="normal",
    type=str,
    choices=SPIKE_SCHEDULE_CHOICES,
    help="默认 normal（与训练一致）；可设 all 在 T>0 时评测多模式",
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
parser.add_argument(
    "--diff1d_trace_samples",
    type=int,
    default=0,
    metavar="N",
    help="diff1d：取 test 第一个 batch 的前 N 个样本，打印中间计算（需模型有 forward_trace_dict）",
)
parser.add_argument(
    "--first_layer_noise_sigma",
    type=float,
    default=0.0,
    help="输入噪声标准差（默认第一层 input_if 附近；position=input_image 时直接加到输入图像）",
)
parser.add_argument(
    "--first_layer_noise_type",
    type=str,
    default="gaussian",
    choices=["gaussian", "pink"],
    help="输入噪声类型（input_image 位置当前按 gaussian 处理）",
)
parser.add_argument(
    "--first_layer_noise_position",
    type=str,
    default="post_input_if",
    choices=["post_input_if", "pre_input_if", "input_image"],
    help="输入噪声注入位置：input_if 后（默认）/ input_if 前 / 直接输入图像",
)
parser.add_argument(
    "--noise_sweep",
    action="store_true",
    help="扫描 sigma，寻找准确率首次降到目标阈值的点（仅分类任务）",
)
parser.add_argument(
    "--noise_target_acc",
    type=float,
    default=90.0,
    help="--noise_sweep 目标准确率阈值（百分比）",
)
parser.add_argument(
    "--noise_sigma_start",
    type=float,
    default=0.0,
    help="--noise_sweep 起始 sigma",
)
parser.add_argument(
    "--noise_sigma_end",
    type=float,
    default=1.0,
    help="--noise_sweep 结束 sigma（含）",
)
parser.add_argument(
    "--noise_sigma_step",
    type=float,
    default=0.05,
    help="--noise_sweep sigma 步长",
)
parser.add_argument(
    "--noise_output_dir",
    type=str,
    default="Noise_exp",
    help="--noise_sweep 结果 CSV 输出目录",
)

args = parser.parse_args()


def _resolved_model_name(dataset, model):
    m = model.lower()
    d = dataset.lower().replace("-", "").replace("_", "")
    if d in ("diff1d", "toydiff1d"):
        return "diff1d"
    if d not in ("mnist", "fashionmnist") and m in ("cnn2", "cnn2_mnist"):
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


def _build_sigma_values(start, end, step):
    if step <= 0:
        raise ValueError("noise_sigma_step 必须 > 0")
    if end < start:
        raise ValueError("noise_sigma_end 必须 >= noise_sigma_start")
    vals = []
    cur = float(start)
    eps = 1e-12
    while cur <= end + eps:
        vals.append(round(cur, 10))
        cur += step
    return vals


def _inject_input_image_noise(inputs, sigma, noise_type):
    sigma = float(sigma)
    if sigma <= 0:
        return inputs
    # input_image 位置当前统一采用高斯噪声；pink 先回退到高斯，避免改动数据管线。
    _ = noise_type
    return inputs + torch.randn_like(inputs) * sigma


def _val_with_input_image_noise(
    model,
    test_loader,
    T,
    device,
    sigma,
    noise_type="gaussian",
    sample_iter=None,
    verbose=True,
):
    if sample_iter is None:
        sample_iter = len(test_loader)
        if verbose:
            print(f"sample_iter of the whole test data loader: {sample_iter}")

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            batch_size = inputs.size(0)
            noisy_inputs = _inject_input_image_noise(
                inputs.to(device), sigma=sigma, noise_type=noise_type
            )
            targets = targets.to(device)
            outputs = model(noisy_inputs)
            if outputs.dim() == 3:
                outputs = outputs.mean(0)
            _, predicted = outputs.max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            if verbose and (batch_idx + 1) % 20 == 0:
                current_acc = 100 * correct / total
                print(
                    f"batch idx={batch_idx + 1}, batch size={batch_size}: current accuracy: {current_acc:.3f}%"
                )
            if batch_idx == sample_iter:
                break
    return 100 * correct / total


def _sigma_key(sigma):
    return ("%.6f" % float(sigma)).rstrip("0").rstrip(".") or "0"


def _write_noise_matrix_csv(matrix_csv_path, l_value, sigma_to_acc):
    """
    写入/更新矩阵表：
    行 = L；列 = sigma；值 = acc
    """
    matrix = {}
    sigma_cols = set()

    if os.path.exists(matrix_csv_path):
        with open(matrix_csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            old_cols = [c for c in (reader.fieldnames or []) if c != "L"]
            sigma_cols.update(old_cols)
            for row in reader:
                l_raw = row.get("L", "").strip()
                if not l_raw:
                    continue
                try:
                    l_key = int(float(l_raw))
                except ValueError:
                    continue
                matrix[l_key] = {}
                for c in old_cols:
                    val = row.get(c, "")
                    if val is None or str(val).strip() == "":
                        continue
                    matrix[l_key][c] = val

    l_int = int(l_value)
    if l_int not in matrix:
        matrix[l_int] = {}
    for sigma, acc in sigma_to_acc.items():
        s_key = _sigma_key(sigma)
        sigma_cols.add(s_key)
        matrix[l_int][s_key] = "%.6f" % float(acc)

    def _sigma_sort_key(x):
        try:
            return float(x)
        except ValueError:
            return float("inf")

    ordered_sigmas = sorted(sigma_cols, key=_sigma_sort_key)
    fieldnames = ["L"] + ordered_sigmas

    with open(matrix_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for l_key in sorted(matrix.keys()):
            row = {"L": l_key}
            row.update(matrix[l_key])
            writer.writerow(row)


def _upsert_noise_combined_csv(combined_csv_path, l_value, t_value, sigma_to_acc):
    """
    维护总表：
    行 = (L, T)；列 = sigma；值 = acc
    若 (L,T) 已存在则覆盖对应 sigma 值；否则新增行。
    """
    table = {}
    sigma_cols = set()

    if os.path.exists(combined_csv_path):
        with open(combined_csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            old_cols = [c for c in (reader.fieldnames or []) if c not in ("L", "T")]
            sigma_cols.update(old_cols)
            for row in reader:
                l_raw = str(row.get("L", "")).strip()
                t_raw = str(row.get("T", "")).strip()
                if not l_raw or not t_raw:
                    continue
                try:
                    key = (int(float(l_raw)), int(float(t_raw)))
                except ValueError:
                    continue
                table[key] = {}
                for c in old_cols:
                    v = row.get(c, "")
                    if v is None or str(v).strip() == "":
                        continue
                    table[key][c] = v

    key = (int(l_value), int(t_value))
    if key not in table:
        table[key] = {}
    for sigma, acc in sigma_to_acc.items():
        s_key = _sigma_key(sigma)
        sigma_cols.add(s_key)
        table[key][s_key] = "%.6f" % float(acc)

    def _sigma_sort_key(x):
        try:
            return float(x)
        except ValueError:
            return float("inf")

    ordered_sigmas = sorted(sigma_cols, key=_sigma_sort_key)
    fieldnames = ["L", "T"] + ordered_sigmas

    with open(combined_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (l_key, t_key) in sorted(table.keys(), key=lambda x: (x[0], x[1])):
            row = {"L": l_key, "T": t_key}
            row.update(table[(l_key, t_key)])
            writer.writerow(row)


def main():
    global args
    print(args)

    ds = args.dataset.lower()
    log_ds = "diff1d" if ds.replace("_", "").replace("-", "") in (
        "diff1d",
        "toydiff1d",
    ) else ds
    log_dir = "%s-test-accuracy" % log_ds
    os.makedirs(log_dir, exist_ok=True)
    model_dir = "%s-checkpoints" % log_ds
    viz_out = args.viz_out_dir or ("%s-test-viz" % log_ds)

    arch = _resolved_model_name(args.dataset, args.model)
    if arch != args.model:
        print("提示: 权重检索使用 arch=%s" % (arch,))
    identifier = arch
    identifier += "_L[%d]" % (args.L,)
    if args.time > 0:
        identifier += "_T[%d]" % (args.time,)
    save_acc_filename = "%s_%s_L%s_T%s" % (
        log_ds,
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
    elif ds in ("diff1d", "toy_diff1d", "diff_1d"):
        pass
    else:
        state_dict = remap_legacy_vgg_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.set_T(args.time)
    model.set_L(args.L)
    if hasattr(model, "set_scaling_factor"):
        model.set_scaling_factor(args.scaling_factor)
    model.set_mode(args.mode)
    use_model_side_noise = args.first_layer_noise_position in (
        "post_input_if",
        "pre_input_if",
    )
    if use_model_side_noise and hasattr(model, "set_first_layer_input_noise_sigma"):
        model.set_first_layer_input_noise_sigma(args.first_layer_noise_sigma)
        if hasattr(model, "set_first_layer_input_noise_type"):
            model.set_first_layer_input_noise_type(args.first_layer_noise_type)
        if hasattr(model, "set_first_layer_input_noise_position"):
            model.set_first_layer_input_noise_position(args.first_layer_noise_position)
        logger.info(
            "第一层输入噪声 type=%s sigma=%.6f position=%s"
            % (
                args.first_layer_noise_type,
                args.first_layer_noise_sigma,
                args.first_layer_noise_position,
            )
        )
    elif use_model_side_noise and args.first_layer_noise_sigma > 0:
        logger.warning("当前模型不支持第一层输入噪声注入，忽略 --first_layer_noise_sigma")
    elif use_model_side_noise and args.first_layer_noise_type != "gaussian":
        logger.warning("当前模型不支持第一层噪声类型设置，忽略 --first_layer_noise_type")
    elif use_model_side_noise and args.first_layer_noise_position != "post_input_if":
        logger.warning(
            "当前模型不支持第一层噪声注入位置设置，忽略 --first_layer_noise_position"
        )
    else:
        # input_image：由测试阶段直接对输入图像加噪，关闭模型内部 first-layer 注入。
        if hasattr(model, "set_first_layer_input_noise_sigma"):
            model.set_first_layer_input_noise_sigma(0.0)
        if hasattr(model, "set_first_layer_input_noise_position"):
            model.set_first_layer_input_noise_position("post_input_if")
        if args.first_layer_noise_type != "gaussian":
            logger.warning(
                "position=input_image 当前按 gaussian 处理，忽略 noise_type=%s"
                % (args.first_layer_noise_type,)
            )
        logger.info(
            "输入图像噪声 type=gaussian sigma=%.6f position=input_image"
            % (args.first_layer_noise_sigma,)
        )

    schedules = _schedules_to_run(args, model)
    if args.time == 0 and args.spike_schedule == "all":
        logger.info("T=0：spike_schedule 不参与，仅评测一次")
    elif hasattr(model, "set_spike_schedule"):
        logger.info(
            "将评测 spike_schedule: %d 种 — %s"
            % (len(schedules), ", ".join(schedules))
        )

    is_diff1d = ds in ("diff1d", "toy_diff1d", "diff_1d")

    if args.calibrate:
        if is_diff1d:
            logger.warning("diff1d 回归任务不支持当前 CE 校准流程，跳过")
        elif args.time == 0:
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

    if args.noise_sweep:
        if is_diff1d:
            logger.warning("--noise_sweep 仅支持分类任务，当前为 diff1d 回归，跳过")
        elif not hasattr(model, "set_first_layer_input_noise_sigma"):
            logger.warning("当前模型不支持第一层输入噪声注入，无法执行 --noise_sweep")
        else:
            sigma_values = _build_sigma_values(
                args.noise_sigma_start, args.noise_sigma_end, args.noise_sigma_step
            )
            logger.info(
                "开始噪声扫描: noise_type=%s, target_acc<=%.3f%%, sigma 从 %.6f 到 %.6f, step=%.6f"
                % (
                    args.first_layer_noise_type,
                    args.noise_target_acc,
                    args.noise_sigma_start,
                    args.noise_sigma_end,
                    args.noise_sigma_step,
                )
            )
            os.makedirs(args.noise_output_dir, exist_ok=True)
            sweep_summary = []
            for mode in schedules:
                if hasattr(model, "set_spike_schedule"):
                    model.set_spike_schedule(mode)
                hit = None
                sigma_to_acc = {}
                for sigma in sigma_values:
                    if use_model_side_noise:
                        model.set_first_layer_input_noise_sigma(sigma)
                        acc = val(model, test_loader, args.time, device, verbose=False)
                    else:
                        acc = _val_with_input_image_noise(
                            model,
                            test_loader,
                            args.time,
                            device,
                            sigma=sigma,
                            noise_type=args.first_layer_noise_type,
                            verbose=False,
                        )
                    print(
                        "noise_sweep mode=%s sigma=%.6f acc=%.3f"
                        % (mode, sigma, acc)
                    )
                    logger.info(
                        "noise_sweep mode=%s sigma=%.6f acc=%.3f"
                        % (mode, sigma, acc)
                    )
                    sigma_to_acc[sigma] = acc
                    if hit is None and acc <= args.noise_target_acc:
                        hit = (sigma, acc)
                matrix_csv_path = os.path.join(
                    args.noise_output_dir,
                    "noise_sweep_matrix_%s_%s_T%d_mode_%s_schedule_%s_seed_%d.csv"
                    % (ds, arch, args.time, args.mode, mode, args.seed),
                )
                _write_noise_matrix_csv(matrix_csv_path, args.L, sigma_to_acc)
                combined_csv_path = os.path.join(
                    args.noise_output_dir, "noise_sweep_combined_L_T.csv"
                )
                _upsert_noise_combined_csv(
                    combined_csv_path, args.L, args.time, sigma_to_acc
                )
                if hit is None:
                    msg = (
                        "mode=%s 在 sigma<=%.6f 范围内未降到 %.3f%%"
                        % (mode, args.noise_sigma_end, args.noise_target_acc)
                    )
                    print(msg)
                    logger.info(msg)
                    sweep_summary.append((mode, None, None))
                else:
                    sigma_hit, acc_hit = hit
                    msg = (
                        "mode=%s 首次达到 acc<=%.3f%%: sigma=%.6f, acc=%.3f"
                        % (mode, args.noise_target_acc, sigma_hit, acc_hit)
                    )
                    print(msg)
                    logger.info(msg)
                    sweep_summary.append((mode, sigma_hit, acc_hit))
                logger.info("noise_sweep 矩阵结果已更新: %s" % (matrix_csv_path,))
                print("noise_sweep 矩阵结果已更新: %s" % (matrix_csv_path,))
                logger.info("noise_sweep 总表已更新: %s" % (combined_csv_path,))
                print("noise_sweep 总表已更新: %s" % (combined_csv_path,))
            if use_model_side_noise:
                model.set_first_layer_input_noise_sigma(args.first_layer_noise_sigma)
            if schedules and hasattr(model, "set_spike_schedule"):
                model.set_spike_schedule(schedules[-1])
            return sweep_summary

    if (
        is_diff1d
        and args.diff1d_trace_samples > 0
        and hasattr(model, "forward_trace_dict")
    ):
        first_mode = schedules[0]
        if hasattr(model, "set_spike_schedule"):
            model.set_spike_schedule(first_mode)
        it0 = iter(test_loader)
        inputs, targets = next(it0)
        n = min(args.diff1d_trace_samples, inputs.size(0))
        inputs = inputs[:n].to(device)
        targets = targets[:n].to(device, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            steps = model.forward_trace_dict(inputs)
        trace_txt = format_diff1d_trace(
            steps, n, args.time, y_true=targets.view(-1)
        )
        hdr = (
            "\n--- diff1d 样本计算过程 (spike_schedule=%s, T=%d, n=%d) ---\n%s\n"
            % (first_mode, args.time, n, trace_txt)
        )
        # 仅 logger：避免 print + StreamHandler 在终端重复打印同一段
        logger.info(hdr)

    results = []
    for mode in schedules:
        if hasattr(model, "set_spike_schedule"):
            model.set_spike_schedule(mode)
        logger.info("spike_schedule=%s" % (mode,))
        if is_diff1d:
            acc = val_reg(
                model, test_loader, args.time, device, verbose=val_verbose
            )
            results.append((mode, acc))
            print("spike_schedule=%s  Test RMSE = %.6f" % (mode, acc))
            logger.info(
                "spike_schedule=%s  Test RMSE = %.6f" % (mode, acc)
            )
        else:
            if use_model_side_noise:
                acc = val(
                    model, test_loader, args.time, device, verbose=val_verbose
                )
            else:
                acc = _val_with_input_image_noise(
                    model,
                    test_loader,
                    args.time,
                    device,
                    sigma=args.first_layer_noise_sigma,
                    noise_type=args.first_layer_noise_type,
                    verbose=val_verbose,
                )
            results.append((mode, acc))
            print("spike_schedule=%s  Test acc = %.3f" % (mode, acc))
            logger.info(
                "spike_schedule=%s  Test acc = %.3f" % (mode, acc)
            )

    print("\n--- 汇总 dataset=%s T=%d ---" % (args.dataset, args.time))
    for mode, acc in results:
        if is_diff1d:
            print("  %-26s  RMSE %.6f" % (mode, acc))
        else:
            print("  %-26s  %.3f" % (mode, acc))
    if is_diff1d:
        summ = ", ".join("%s=%.6f" % (m, a) for m, a in results)
    else:
        summ = ", ".join("%s=%.3f" % (m, a) for m, a in results)
    logger.info("汇总: %s" % (summ,))

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
