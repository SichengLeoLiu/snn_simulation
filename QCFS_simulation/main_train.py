import argparse
import csv
import math
import os
import torch
import torch.nn as nn
import torch.optim
from Models import modelpool
from Models.VGG import remap_legacy_vgg_state_dict
from Models.spike_temporal_adjust import SPIKE_SCHEDULE_MODES
from Preprocess import datapool
from grad_probe import (
    append_probe_csv,
    parse_probe_epochs,
    probe_ce_vs_reg_grads,
    stage_name_for_epoch,
    summarize_probe_rows,
)
from utils import (
    train,
    val,
    train_reg,
    val_reg,
    seed_all,
    get_logger,
    get_torch_device,
    configure_cuda_fast,
    compute_mne_l2_regularization,
    compute_mne_l2_all_regularization,
    compute_l2_calibrated_mne_regularization,
    compute_stable_mne_l2_regularization,
    compute_hinge_mne_regularization,
    compute_pc_mne_regularization,
    compute_margin_mne_regularization,
    compute_conv_mne_l2_regularization,
    dump_mne_mapping_report,
    ce_vs_reg_grad_ratio,
    compute_l1_regularization,
    compute_l1_all_regularization,
    compute_manual_l2_regularization,
    compute_manual_l2_all_regularization,
    compute_selective_l2_regularization,
    compute_elastic_net_all_regularization,
    compute_scale_l2_regularization,
    compute_group_lasso_regularization,
    compute_spectral_norm_regularization,
    compute_orthogonal_regularization,
    compute_spectral_mne_regularization,
    compute_effective_l2_regularization,
    compute_threshold_normalized_l2_regularization,
    compute_l2_sp_regularization,
)

DATASET_CHOICES = ["mnist", "fashion_mnist", "cifar10", "cifar100", "imagenet", "diff1d"]

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
    choices=[
        "weight_decay",
        "resolution_aware",
        "mne_l2",
        "mne_l2_all",
        "calibrated_mne_l2",
        "stable_mne_l2",
        "hinge_mne",
        "pc_mne",
        "margin_mne",
        "spectral_mne",
        "conv_mne_l2",
        "l1",
        "l1_all",
        "manual_l2",
        "manual_l2_all",
        "manual_l2_w_bn",
        "manual_l2_w_bn_gamma",
        "manual_l2_w_bn_beta",
        "manual_l2_w_if",
        "manual_l2_w_bn_if",
        "elastic_net_all",
        "scale_l2",
        "weight_decay_weights_only",
        "group_lasso",
        "spectral_norm",
        "orthogonal",
        "effective_l2",
        "threshold_l2",
        "l2_sp",
    ],
    help="正则方式：... | hinge_mne | pc_mne | margin_mne | spectral_mne | ...",
)
parser.add_argument(
    "--reg_coeff",
    default=1.0,
    type=float,
    help="显式正则项（MNE/L1/group_lasso/spectral_norm/orthogonal 等）的全局系数 beta",
)
parser.add_argument(
    "--reg_warmup_epochs",
    default=0,
    type=int,
    help="正则系数 warmup 轮数；0 表示不 warmup",
)
parser.add_argument(
    "--reg_warmup_schedule",
    default="linear",
    choices=["linear", "cosine"],
    help="--reg_warmup_epochs>0 时 β1 从 0 升到目标值的形状",
)
parser.add_argument(
    "--init-from",
    default="",
    type=str,
    help="用已有 checkpoint 初始化后再训练（两阶段 / 续训）",
)
parser.add_argument(
    "--save-epochs",
    default="",
    type=str,
    help="额外保存的 0-based epoch，逗号分隔，例如 29,59",
)
parser.add_argument(
    "--lr-cosine-t-max",
    default=None,
    type=int,
    help="覆盖 CosineAnnealingLR 的 T_max；默认等于 --epochs",
)
parser.add_argument(
    "--lr-cosine-last-epoch",
    default=-1,
    type=int,
    help="CosineAnnealingLR last_epoch，用于续上一段 300-epoch 余弦",
)
parser.add_argument(
    "--elastic_l1_ratio",
    default=0.04,
    type=float,
    help="--regularizer=elastic_net_all 时，显式 L1 系数与全局 reg_coeff 的比值",
)
parser.add_argument(
    "--group_lasso_eps",
    default=1e-12,
    type=float,
    help="--regularizer=group_lasso 时 filter L2 norm 的数值稳定项",
)
parser.add_argument(
    "--spectral_power_iters",
    default=3,
    type=int,
    help="--regularizer=spectral_norm 时估计最大奇异值的 power iteration 次数",
)
parser.add_argument(
    "--spectral_mne_power_iters",
    default=3,
    type=int,
    help="--regularizer=spectral_mne 时估计最大奇异值的 power iteration 次数",
)
parser.add_argument(
    "--spectral_mne_layer_reduce",
    default="sum",
    type=str,
    choices=["sum", "mean"],
    help="--regularizer=spectral_mne 时跨层聚合方式",
)
parser.add_argument(
    "--spectral_mne_no_detach_bn_stats",
    action="store_true",
    help="--regularizer=spectral_mne 时不 detach BN running_var",
)
parser.add_argument(
    "--spectral_mne_no_detach_bn_affine",
    action="store_true",
    help="--regularizer=spectral_mne 时不 detach BN affine gamma",
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
parser.add_argument(
    "--mne_detach_lambda",
    action="store_true",
    help="--regularizer=mne_l2 时对 IF 阈值 lambda 使用 stop-gradient",
)
parser.add_argument(
    "--mne_no_detach_bn_stats",
    action="store_true",
    help="--regularizer=mne_l2 时允许正则梯度回传到 BN running_var/gamma",
)
parser.add_argument(
    "--mne_no_bn_fold",
    action="store_true",
    help="--regularizer=mne_l2 时分子只用生权重 W，不把 BN γ/var fold 进去",
)
parser.add_argument(
    "--mne_frobenius",
    action="store_true",
    help="--regularizer=mne_l2 时 M_eff=||W||_F^2（逐层完整平方和，对齐 weights-only L2）；默认是按输出通道取均值",
)
parser.add_argument(
    "--calibrated_mne_alpha",
    default=0.25,
    type=float,
    help="--regularizer=calibrated_mne_l2 时 MNE 风险重加权比例；0 严格退化为 weights-only L2",
)
parser.add_argument(
    "--calibrated_mne_risk_min",
    default=0.5,
    type=float,
    help="calibrated MNE 的归一化风险下界",
)
parser.add_argument(
    "--calibrated_mne_risk_max",
    default=2.0,
    type=float,
    help="calibrated MNE 的归一化风险上界",
)
parser.add_argument(
    "--calibrated_mne_alpha_start_epoch",
    default=0,
    type=int,
    help="开始从 alpha=0 向目标 calibrated MNE alpha 过渡的 epoch",
)
parser.add_argument(
    "--calibrated_mne_alpha_warmup_epochs",
    default=0,
    type=int,
    help="calibrated MNE alpha 的线性 warmup 长度；基础 L2 始终开启",
)
parser.add_argument(
    "--calibrated_mne_no_bn_fold",
    action="store_true",
    help="calibrated MNE 风险系数仅使用 L^2/lambda^2，不使用 BN gamma^2/var",
)
parser.add_argument(
    "--calibrated_mne_normalization",
    default="global",
    choices=("global", "layerwise"),
    help="calibrated MNE 风险系数使用全模型或逐层 mean-one 归一化",
)
parser.add_argument(
    "--calibrated_mne_onesided",
    action="store_true",
    help="calibrated MNE 只加强高风险通道: q=1+α max(r̂-τ,0)，不减弱低风险通道",
)
parser.add_argument(
    "--calibrated_mne_tau",
    default=1.0,
    type=float,
    help="--calibrated_mne_onesided 时的风险阈值 τ（mean-one 风险，默认 1）",
)
parser.add_argument(
    "--calibrated_mne_q_assignment",
    default="risk",
    choices=("risk", "identity", "strength", "shuffle", "layer_mean"),
    help="onesided q 如何接到通道: risk=真实风险, identity=q=1, "
    "strength=全体用 onesided 的均值 q, shuffle=打乱通道对应, "
    "layer_mean=层内通道共用该层 q_OS 均值",
)
parser.add_argument(
    "--calibrated_mne_q_max",
    default=0.0,
    type=float,
    help="onesided q 的上限；0 表示不封顶。q=min(q_max, 1+α[r̂-τ]_+)",
)
parser.add_argument(
    "--mne_layer_map",
    default="legacy",
    choices=("legacy", "resnet"),
    help="Conv/IF matching: legacy=same-Sequential Conv-BN-IF; "
    "resnet=also map residual-terminal and shortcut convs to BasicBlock.act",
)
parser.add_argument(
    "--mapping_diag_dir",
    default="",
    type=str,
    help="若非空，训练开始时写出 mapping_summary.json / layer_q.csv",
)
parser.add_argument(
    "--epoch_log_csv",
    default="",
    type=str,
    help="每个 epoch 追加 q/P(r>τ)/正则梯度/ANN acc；空则写到 checkpoint 目录",
)
parser.add_argument(
    "--stable_mne_l_ref",
    default=16.0,
    type=float,
    help="--regularizer=stable_mne_l2 时使用 (L/L_ref)^2 缩放",
)
parser.add_argument(
    "--stable_mne_no_fan_in_norm",
    action="store_true",
    help="--regularizer=stable_mne_l2 时关闭 fan-in 归一化",
)
parser.add_argument(
    "--stable_mne_layer_reduce",
    default="mean",
    type=str,
    choices=["sum", "mean"],
    help="--regularizer=stable_mne_l2 时跨层聚合方式",
)
parser.add_argument(
    "--stable_mne_detach_bn_affine",
    action="store_true",
    help="--regularizer=stable_mne_l2 时也 detach BN affine gamma",
)
parser.add_argument(
    "--stable_mne_no_detach_bn_running_stats",
    action="store_true",
    help="--regularizer=stable_mne_l2 时不 detach BN running_var",
)
parser.add_argument(
    "--hinge_mne_tau",
    default=1.0,
    type=float,
    help="--regularizer=hinge_mne 时的 gain 阈值 tau",
)
parser.add_argument(
    "--hinge_mne_linear",
    action="store_true",
    help="--regularizer=hinge_mne 时使用 relu(gain-tau)^2；默认使用 log-hinge",
)
parser.add_argument(
    "--hinge_mne_layer_reduce",
    default="mean",
    type=str,
    choices=["sum", "mean"],
    help="--regularizer=hinge_mne 时跨层聚合方式",
)
parser.add_argument(
    "--hinge_mne_normalize_by_fan_in",
    action="store_true",
    help="--regularizer=hinge_mne 时对 M_eff 做 fan-in 归一化",
)
parser.add_argument(
    "--hinge_mne_no_detach_bn_stats",
    action="store_true",
    help="--regularizer=hinge_mne 时不 detach BN running_var",
)
parser.add_argument(
    "--hinge_mne_no_detach_bn_affine",
    action="store_true",
    help="--regularizer=hinge_mne 时不 detach BN affine gamma",
)
parser.add_argument(
    "--pc_mne_sigma",
    type=float,
    default=1.0,
    help="--regularizer=pc_mne/margin_mne 时理论噪声强度 σ（默认 1.0）",
)
parser.add_argument(
    "--pc_mne_protocol",
    type=str,
    default="snn_indep",
    choices=["snn_indep", "ann_qcfs", "snn_shared", "pre_first_conv"],
    help="噪声尺度 a：snn_indep/pre_first_conv→√T；ann_qcfs→L；snn_shared→T",
)
parser.add_argument(
    "--pc_mne_eval_T",
    type=int,
    default=16,
    help="PC/Margin-MNE 预测 SNN 测试时间步 T（默认 16）",
)
parser.add_argument(
    "--pc_mne_detach_lambda",
    action="store_true",
    help="PC/Margin-MNE 对 IF λ 使用 stop-gradient（默认允许回传到 λ）",
)
parser.add_argument(
    "--pc_mne_lambda_log_coeff",
    type=float,
    default=0.0,
    help="可选 (logλ-logλ_ref)^2 系数，默认 0",
)
parser.add_argument(
    "--pc_mne_lambda_ref",
    type=float,
    default=1.0,
    help="λ log-penalty 参考值",
)
parser.add_argument(
    "--margin_mne_tau",
    type=float,
    default=2.0,
    help="--regularizer=margin_mne 的最小 ρ=d/s 目标（默认 2）",
)
parser.add_argument(
    "--conv_mne_no_detach_lambda",
    action="store_true",
    help="--regularizer=conv_mne_l2 时关闭 stop-gradient（默认开启 detach）",
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
parser.add_argument(
    "--ckpt-save-mode",
    default="best",
    type=str,
    choices=["best", "last"],
    help="checkpoint 保存策略：best=验证集最优（默认），last=仅保存最后一个 epoch",
)
parser.add_argument(
    "--ckpt-dir",
    default="",
    type=str,
    help="checkpoint/日志目录；空则用 <dataset>-checkpoints。Gadi 请指到 scratch",
)
parser.add_argument(
    "--grad_probe",
    action="store_true",
    help=(
        "在指定 epoch 结束后，用 1 个 train batch 分解 CE vs 显式正则对 "
        "IF.lambda / BN.gamma 的梯度（方向 cosine + L2 范数）；"
        "需要显式正则（如 manual_l2 / mne_l2），不适用于纯 optimizer weight_decay"
    ),
)
parser.add_argument(
    "--grad_probe_epochs",
    default="auto",
    type=str,
    help="grad probe 的 0-based epoch 列表，逗号分隔；auto=初期/中期/末期各一次",
)
parser.add_argument(
    "--grad_probe_csv",
    default="",
    type=str,
    help="grad probe CSV 路径；默认写到 checkpoint 目录 <identifier>_grad_probe.csv",
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
    configure_cuda_fast(device)

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
    model._mne_layer_map = args.mne_layer_map
    model.set_L(args.L)
    model.set_T(args.time)
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule(args.spike_schedule)

    log_ds = "diff1d" if ds.replace("_", "") in ("diff1d", "toydiff1d") else ds
    log_dir = args.ckpt_dir.strip() or ("%s-checkpoints" % log_ds)
    os.makedirs(log_dir, exist_ok=True)

    model.to(device)
    if args.init_from:
        init_path = os.path.abspath(args.init_from)
        if not os.path.isfile(init_path):
            raise FileNotFoundError("--init-from missing: %s" % (init_path,))
        state = torch.load(init_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(remap_legacy_vgg_state_dict(state), strict=True)
        print("init-from: %s" % (init_path,))

    reg_loss_fn = None
    calibrated_mne_state = {
        "alpha": float(args.calibrated_mne_alpha),
        "shuffle_counter": 0,
    }
    l2_sp_reference = None
    if args.regularizer == "l2_sp":
        l2_sp_reference = {
            f"{name}.weight": module.weight.detach().clone()
            for name, module in model.named_modules()
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear))
            and getattr(module, "weight", None) is not None
        }

    is_diff1d = log_ds == "diff1d"

    def _optimizer_weight_decay(regularizer: str, weight_decay: float) -> float:
        if regularizer in ("weight_decay", "weight_decay_weights_only"):
            return weight_decay
        if regularizer in ("mne_l2", "stable_mne_l2", "hinge_mne", "spectral_mne", "conv_mne_l2") and weight_decay > 0:
            return weight_decay
        # Other regularizers are explicit losses and do not use optimizer WD.
        return 0.0

    optimizer_parameters = model.parameters()
    if args.regularizer == "weight_decay_weights_only":
        decay_ids = {
            id(module.weight)
            for module in model.modules()
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear))
            and getattr(module, "weight", None) is not None
        }
        decay_parameters = []
        no_decay_parameters = []
        for parameter in model.parameters():
            target = decay_parameters if id(parameter) in decay_ids else no_decay_parameters
            target.append(parameter)
        optimizer_parameters = [
            {"params": decay_parameters},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ]

    if is_diff1d:
        criterion = nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(
            optimizer_parameters,
            lr=args.lr,
            weight_decay=_optimizer_weight_decay(args.regularizer, args.weight_decay),
        )
    else:
        criterion = nn.CrossEntropyLoss().to(device)
        if ds == "mnist":
            optimizer = torch.optim.Adam(
                optimizer_parameters,
                lr=args.lr,
                weight_decay=_optimizer_weight_decay(args.regularizer, args.weight_decay),
            )
        else:
            optimizer = torch.optim.SGD(
                optimizer_parameters,
                lr=args.lr,
                momentum=0.9,
                weight_decay=_optimizer_weight_decay(args.regularizer, args.weight_decay),
            )

    if args.regularizer == "resolution_aware":
        if hasattr(model, "resolution_aware_noise_regularization"):
            reg_loss_fn = (
                lambda m, t, q: m.resolution_aware_noise_regularization(T=t)
            )
        else:
            raise ValueError(
                "模型 %s 不支持 resolution_aware 正则（缺少 resolution_aware_noise_regularization）"
                % (arch,)
            )
    elif args.regularizer == "mne_l2":
        reg_loss_fn = lambda m, t, q: compute_mne_l2_regularization(
            m,
            quant_level=(args.L if q is None else q),
            eps=args.mne_eps,
            use_max=args.mne_use_max,
            detach_lambda=args.mne_detach_lambda,
            detach_bn_stats=(not args.mne_no_detach_bn_stats),
            fold_bn=(not args.mne_no_bn_fold),
            full_frobenius=args.mne_frobenius,
        )
    elif args.regularizer == "mne_l2_all":
        reg_loss_fn = lambda m, t, q: compute_mne_l2_all_regularization(
            m,
            quant_level=(args.L if q is None else q),
            eps=args.mne_eps,
            use_max=args.mne_use_max,
            detach_lambda=args.mne_detach_lambda,
            detach_bn_stats=(not args.mne_no_detach_bn_stats),
            fold_bn=(not args.mne_no_bn_fold),
            full_frobenius=args.mne_frobenius,
        )
    elif args.regularizer == "calibrated_mne_l2":
        def _calibrated_mne_reg(m, t, q):
            calibrated_mne_state["shuffle_counter"] += 1
            return compute_l2_calibrated_mne_regularization(
                m,
                quant_level=(args.L if q is None else q),
                alpha=calibrated_mne_state["alpha"],
                eps=args.mne_eps,
                risk_min=args.calibrated_mne_risk_min,
                risk_max=args.calibrated_mne_risk_max,
                fold_bn=(not args.calibrated_mne_no_bn_fold),
                normalization=args.calibrated_mne_normalization,
                onesided=args.calibrated_mne_onesided,
                tau=args.calibrated_mne_tau,
                q_assignment=args.calibrated_mne_q_assignment,
                shuffle_seed=int(args.seed) * 1_000_003
                + int(calibrated_mne_state["shuffle_counter"]),
                q_max=args.calibrated_mne_q_max,
            )

        reg_loss_fn = _calibrated_mne_reg
    elif args.regularizer == "stable_mne_l2":
        reg_loss_fn = lambda m, t, q: compute_stable_mne_l2_regularization(
            m,
            quant_level=(args.L if q is None else q),
            eps=args.mne_eps,
            use_max=args.mne_use_max,
            detach_lambda=args.mne_detach_lambda,
            detach_bn_running_stats=(not args.stable_mne_no_detach_bn_running_stats),
            detach_bn_affine=args.stable_mne_detach_bn_affine,
            normalize_by_fan_in=(not args.stable_mne_no_fan_in_norm),
            layer_reduction=args.stable_mne_layer_reduce,
            l_ref=args.stable_mne_l_ref,
        )
    elif args.regularizer == "hinge_mne":
        reg_loss_fn = lambda m, t, q: compute_hinge_mne_regularization(
            m,
            quant_level=(args.L if q is None else q),
            eps=args.mne_eps,
            use_max=args.mne_use_max,
            detach_lambda=args.mne_detach_lambda,
            detach_bn_stats=(not args.hinge_mne_no_detach_bn_stats),
            detach_bn_affine=(not args.hinge_mne_no_detach_bn_affine),
            tau=args.hinge_mne_tau,
            use_log=(not args.hinge_mne_linear),
            normalize_by_fan_in=args.hinge_mne_normalize_by_fan_in,
            layer_reduction=args.hinge_mne_layer_reduce,
        )
    elif args.regularizer == "pc_mne":
        reg_loss_fn = lambda m, t, q: compute_pc_mne_regularization(
            m,
            quant_level=(args.L if q is None else q),
            noise_sigma=args.pc_mne_sigma,
            protocol=args.pc_mne_protocol,
            eval_T=args.pc_mne_eval_T,
            eps=args.mne_eps,
            detach_lambda=args.pc_mne_detach_lambda,
            lambda_log_coeff=args.pc_mne_lambda_log_coeff,
            lambda_ref=args.pc_mne_lambda_ref,
            first_layer_only=True,
        )
    elif args.regularizer == "margin_mne":
        reg_loss_fn = lambda m, t, q: compute_margin_mne_regularization(
            m,
            quant_level=(args.L if q is None else q),
            noise_sigma=args.pc_mne_sigma,
            protocol=args.pc_mne_protocol,
            eval_T=args.pc_mne_eval_T,
            tau=args.margin_mne_tau,
            eps=args.mne_eps,
            detach_lambda=args.pc_mne_detach_lambda,
            lambda_log_coeff=args.pc_mne_lambda_log_coeff,
            lambda_ref=args.pc_mne_lambda_ref,
            first_layer_only=True,
        )
    elif args.regularizer == "spectral_mne":
        reg_loss_fn = lambda m, t, q: compute_spectral_mne_regularization(
            m,
            quant_level=(args.L if q is None else q),
            eps=args.mne_eps,
            power_iters=args.spectral_mne_power_iters,
            detach_lambda=args.mne_detach_lambda,
            detach_bn_stats=(not args.spectral_mne_no_detach_bn_stats),
            detach_bn_affine=(not args.spectral_mne_no_detach_bn_affine),
            layer_reduction=args.spectral_mne_layer_reduce,
        )
    elif args.regularizer == "conv_mne_l2":
        reg_loss_fn = lambda m, t, q: compute_conv_mne_l2_regularization(
            m,
            quant_level=(args.L if q is None else q),
            eps=args.mne_eps,
            use_max=args.mne_use_max,
            detach_lambda=(not args.conv_mne_no_detach_lambda),
        )
    elif args.regularizer == "l1":
        reg_loss_fn = lambda m, t, q: compute_l1_regularization(m, T=t, quant_level=q)
    elif args.regularizer == "l1_all":
        reg_loss_fn = lambda m, t, q: compute_l1_all_regularization(
            m, T=t, quant_level=q
        )
    elif args.regularizer == "manual_l2":
        reg_loss_fn = lambda m, t, q: compute_manual_l2_regularization(
            m, T=t, quant_level=q
        )
    elif args.regularizer == "manual_l2_all":
        reg_loss_fn = lambda m, t, q: compute_manual_l2_all_regularization(
            m, T=t, quant_level=q
        )
    elif args.regularizer == "manual_l2_w_bn":
        reg_loss_fn = lambda m, t, q: compute_selective_l2_regularization(
            m, T=t, quant_level=q, include_bn=True
        )
    elif args.regularizer == "manual_l2_w_bn_gamma":
        reg_loss_fn = lambda m, t, q: compute_selective_l2_regularization(
            m, T=t, quant_level=q, include_bn_weight=True
        )
    elif args.regularizer == "manual_l2_w_bn_beta":
        reg_loss_fn = lambda m, t, q: compute_selective_l2_regularization(
            m, T=t, quant_level=q, include_bn_bias=True
        )
    elif args.regularizer == "manual_l2_w_if":
        reg_loss_fn = lambda m, t, q: compute_selective_l2_regularization(
            m, T=t, quant_level=q, include_if=True
        )
    elif args.regularizer == "manual_l2_w_bn_if":
        reg_loss_fn = lambda m, t, q: compute_selective_l2_regularization(
            m, T=t, quant_level=q, include_bn=True, include_if=True
        )
    elif args.regularizer == "elastic_net_all":
        reg_loss_fn = lambda m, t, q: compute_elastic_net_all_regularization(
            m,
            T=t,
            quant_level=q,
            l1_ratio=args.elastic_l1_ratio,
        )
    elif args.regularizer == "scale_l2":
        reg_loss_fn = lambda m, t, q: compute_scale_l2_regularization(
            m, T=t, quant_level=q
        )
    elif args.regularizer == "group_lasso":
        reg_loss_fn = lambda m, t, q: compute_group_lasso_regularization(
            m,
            T=t,
            quant_level=q,
            eps=args.group_lasso_eps,
        )
    elif args.regularizer == "spectral_norm":
        reg_loss_fn = lambda m, t, q: compute_spectral_norm_regularization(
            m,
            T=t,
            quant_level=q,
            power_iters=args.spectral_power_iters,
        )
    elif args.regularizer == "orthogonal":
        reg_loss_fn = lambda m, t, q: compute_orthogonal_regularization(
            m,
            T=t,
            quant_level=q,
        )
    elif args.regularizer == "effective_l2":
        reg_loss_fn = lambda m, t, q: compute_effective_l2_regularization(
            m,
            T=t,
            quant_level=q,
        )
    elif args.regularizer == "threshold_l2":
        reg_loss_fn = lambda m, t, q: compute_threshold_normalized_l2_regularization(
            m,
            T=t,
            quant_level=q,
            eps=args.mne_eps,
            use_max=args.mne_use_max,
            detach_lambda=True,
        )
    elif args.regularizer == "l2_sp":
        reg_loss_fn = lambda m, t, q: compute_l2_sp_regularization(
            m,
            reference_weights=l2_sp_reference,
            T=t,
            quant_level=q,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(args.lr_cosine_t_max if args.lr_cosine_t_max else args.epochs),
        last_epoch=int(args.lr_cosine_last_epoch),
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
    epoch_log_csv = args.epoch_log_csv.strip() or os.path.join(
        log_dir, "%s_epoch_log.csv" % (identifier,)
    )
    os.makedirs(os.path.dirname(os.path.abspath(epoch_log_csv)) or ".", exist_ok=True)
    epoch_log_fields = [
        "epoch",
        "alpha",
        "q_assignment",
        "q_mean",
        "q_std",
        "q_min",
        "q_max",
        "q_os_mean",
        "q_os_std",
        "p_gt_tau",
        "p_at_qmax",
        "q_cap",
        "reg_grad_norm",
        "reg_coeff_grad_norm",
        "ce_grad_norm",
        "reg_ce_ratio",
        "ann_train_acc",
        "ann_test_acc",
        "train_loss",
    ]

    def _append_epoch_log(row: dict) -> None:
        write_header = not os.path.exists(epoch_log_csv)
        with open(epoch_log_csv, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=epoch_log_fields)
            if write_header:
                writer.writeheader()
            out = {}
            for key in epoch_log_fields:
                value = row.get(key, "")
                if hasattr(value, "detach"):
                    value = float(value.detach().cpu().item())
                elif isinstance(value, (int, float)) and key != "q_assignment":
                    value = float(value)
                out[key] = value
            writer.writerow(out)

    def _reg_grad_norms(epoch_reg_coeff: float) -> tuple[float, float]:
        if reg_loss_fn is None:
            return 0.0, 0.0
        model.zero_grad(set_to_none=True)
        reg = reg_loss_fn(model, args.time, args.L)
        if not torch.is_tensor(reg) or not bool(reg.requires_grad):
            model.zero_grad(set_to_none=True)
            return 0.0, 0.0
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(reg, params, allow_unused=True, retain_graph=False)
        total = 0.0
        for grad in grads:
            if grad is not None:
                total += float(grad.detach().pow(2).sum().item())
        model.zero_grad(set_to_none=True)
        raw = total ** 0.5
        return raw, raw * float(epoch_reg_coeff)

    def _ce_reg_ratio(epoch_reg_coeff: float) -> dict:
        if reg_loss_fn is None:
            return {
                "ce_grad_norm": 0.0,
                "reg_grad_norm": 0.0,
                "reg_coeff_grad_norm": 0.0,
                "reg_ce_ratio": float("nan"),
            }
        try:
            images, labels = next(iter(train_loader))
            return ce_vs_reg_grad_ratio(
                model,
                images,
                labels,
                criterion,
                reg_loss_fn,
                args.time,
                args.L,
                epoch_reg_coeff,
            )
        except Exception as exc:
            logger.info("ce/reg grad ratio failed: %s", exc)
            return {
                "ce_grad_norm": 0.0,
                "reg_grad_norm": 0.0,
                "reg_coeff_grad_norm": 0.0,
                "reg_ce_ratio": float("nan"),
            }
    logger.info(
        "start training dataset=%s arch=%s T=%d" % (args.dataset, arch, args.time)
    )
    logger.info(
        "regularizer=%s, weight_decay=%.6g, reg_coeff=%.6g"
        % (args.regularizer, args.weight_decay, args.reg_coeff)
    )
    if args.reg_warmup_epochs > 0:
        logger.info(
            "reg_warmup_epochs=%d schedule=%s",
            args.reg_warmup_epochs,
            args.reg_warmup_schedule,
        )
    if args.init_from:
        logger.info("init_from=%s", args.init_from)
    logger.info("ckpt_save_mode=%s", args.ckpt_save_mode)
    if args.regularizer == "mne_l2":
        logger.info(
            "mne_l2: L=%d, eps=%.3e, use_max=%s, frobenius=%s, detach_lambda=%s, detach_bn_stats=%s, fold_bn=%s, layer_map=%s"
            % (
                args.L,
                args.mne_eps,
                str(bool(args.mne_use_max)),
                str(bool(args.mne_frobenius)),
                str(bool(args.mne_detach_lambda)),
                str(bool(not args.mne_no_detach_bn_stats)),
                str(bool(not args.mne_no_bn_fold)),
                args.mne_layer_map,
            )
        )
    if args.regularizer == "mne_l2_all":
        logger.info(
            "mne_l2_all: MNE on matched Conv/Linear weights + L2 on remaining trainable params; "
            "L=%d, eps=%.3e, use_max=%s, detach_lambda=%s, detach_bn_stats=%s, fold_bn=%s, optimizer_wd=0"
            % (
                args.L,
                args.mne_eps,
                str(bool(args.mne_use_max)),
                str(bool(args.mne_detach_lambda)),
                str(bool(not args.mne_no_detach_bn_stats)),
                str(bool(not args.mne_no_bn_fold)),
            )
        )
    if args.regularizer == "calibrated_mne_l2":
        logger.info(
            "calibrated_mne_l2: base=0.5*sum(q*W^2), target_alpha=%.4g, "
            "alpha_start=%d, alpha_warmup=%d, risk_clip=[%.4g, %.4g], "
            "fold_bn=%s, normalization=%s, onesided=%s, tau=%.4g, "
            "q_assignment=%s, q_max=%s, layer_map=%s, risk_detached=True, unmatched_head_q=1, optimizer_wd=0"
            % (
                args.calibrated_mne_alpha,
                args.calibrated_mne_alpha_start_epoch,
                args.calibrated_mne_alpha_warmup_epochs,
                args.calibrated_mne_risk_min,
                args.calibrated_mne_risk_max,
                str(bool(not args.calibrated_mne_no_bn_fold)),
                args.calibrated_mne_normalization,
                str(bool(args.calibrated_mne_onesided)),
                args.calibrated_mne_tau,
                args.calibrated_mne_q_assignment,
                (
                    "none"
                    if float(args.calibrated_mne_q_max) <= 0
                    else ("%.4g" % args.calibrated_mne_q_max)
                ),
                args.mne_layer_map,
            )
        )
    if args.regularizer == "stable_mne_l2":
        logger.info(
            "stable_mne_l2: L=%d, L_ref=%.6g, eps=%.3e, use_max=%s, detach_lambda=%s, detach_bn_running_stats=%s, detach_bn_affine=%s, fan_in_norm=%s, layer_reduce=%s"
            % (
                args.L,
                args.stable_mne_l_ref,
                args.mne_eps,
                str(bool(args.mne_use_max)),
                str(bool(args.mne_detach_lambda)),
                str(bool(not args.stable_mne_no_detach_bn_running_stats)),
                str(bool(args.stable_mne_detach_bn_affine)),
                str(bool(not args.stable_mne_no_fan_in_norm)),
                args.stable_mne_layer_reduce,
            )
        )
    if args.regularizer == "hinge_mne":
        logger.info(
            "hinge_mne: L=%d, tau=%.6g, eps=%.3e, use_log=%s, use_max=%s, detach_lambda=%s, detach_bn_stats=%s, detach_bn_affine=%s, fan_in_norm=%s, layer_reduce=%s"
            % (
                args.L,
                args.hinge_mne_tau,
                args.mne_eps,
                str(bool(not args.hinge_mne_linear)),
                str(bool(args.mne_use_max)),
                str(bool(args.mne_detach_lambda)),
                str(bool(not args.hinge_mne_no_detach_bn_stats)),
                str(bool(not args.hinge_mne_no_detach_bn_affine)),
                str(bool(args.hinge_mne_normalize_by_fan_in)),
                args.hinge_mne_layer_reduce,
            )
        )
    if args.regularizer in ("pc_mne", "margin_mne"):
        logger.info(
            "%s: first_layer_only=True, sigma=%.6g, protocol=%s, eval_T=%d, "
            "detach_lambda=%s, lambda_log_coeff=%.3g, lambda_ref=%.3g, "
            "warmup=%d, margin_tau=%.3g, optimizer_wd=0 (no BN/IF L2)"
            % (
                args.regularizer,
                args.pc_mne_sigma,
                args.pc_mne_protocol,
                args.pc_mne_eval_T,
                str(bool(args.pc_mne_detach_lambda)),
                args.pc_mne_lambda_log_coeff,
                args.pc_mne_lambda_ref,
                args.reg_warmup_epochs,
                args.margin_mne_tau,
            )
        )
    if args.regularizer == "spectral_mne":
        logger.info(
            "spectral_mne: L=%d, eps=%.3e, power_iters=%d, detach_lambda=%s, detach_bn_stats=%s, detach_bn_affine=%s, layer_reduce=%s"
            % (
                args.L,
                args.mne_eps,
                args.spectral_mne_power_iters,
                str(bool(args.mne_detach_lambda)),
                str(bool(not args.spectral_mne_no_detach_bn_stats)),
                str(bool(not args.spectral_mne_no_detach_bn_affine)),
                args.spectral_mne_layer_reduce,
            )
        )
    if args.regularizer == "conv_mne_l2":
        logger.info(
            "conv_mne_l2: L=%d, eps=%.3e, use_max=%s, detach_lambda=%s"
            % (
                args.L,
                args.mne_eps,
                str(bool(args.mne_use_max)),
                str(bool(not args.conv_mne_no_detach_lambda)),
            )
        )
    if args.regularizer == "group_lasso":
        logger.info("group_lasso: conv_filters_only=True, eps=%.3e", args.group_lasso_eps)
    if args.regularizer == "manual_l2":
        logger.info(
            "manual_l2: penalty=sum(W^2), conv_linear_weights=True, bias=False, optimizer_wd=0"
        )
    if args.regularizer == "manual_l2_all":
        logger.info(
            "manual_l2_all: penalty=sum(p^2), all_trainable_parameters=True, optimizer_wd=0"
        )
    if args.regularizer in (
        "manual_l2_w_bn",
        "manual_l2_w_bn_gamma",
        "manual_l2_w_bn_beta",
        "manual_l2_w_if",
        "manual_l2_w_bn_if",
    ):
        logger.info(
            "%s: conv_linear_weights=True, bn_gamma=%s, bn_beta=%s, if_thresholds=%s, optimizer_wd=0"
            % (
                args.regularizer,
                args.regularizer
                in ("manual_l2_w_bn", "manual_l2_w_bn_gamma", "manual_l2_w_bn_if"),
                args.regularizer
                in ("manual_l2_w_bn", "manual_l2_w_bn_beta", "manual_l2_w_bn_if"),
                args.regularizer in ("manual_l2_w_if", "manual_l2_w_bn_if"),
            )
        )
    if args.regularizer == "l1_all":
        logger.info("l1_all: penalty=sum(abs(p)), all_trainable_parameters=True")
    if args.regularizer == "elastic_net_all":
        logger.info(
            "elastic_net_all: penalty=sum(p^2)+%.6g*sum(abs(p)), all_trainable_parameters=True",
            args.elastic_l1_ratio,
        )
    if args.regularizer == "scale_l2":
        logger.info("scale_l2: bn_affine=True, if_thresholds=True, diagnostic_only=True")
    if args.regularizer == "weight_decay_weights_only":
        logger.info(
            "weight_decay_weights_only: conv_linear_weights=True, bias_bn_if=False"
        )
    if args.regularizer == "spectral_norm":
        logger.info(
            "spectral_norm: conv_linear=True, power_iters=%d",
            args.spectral_power_iters,
        )
    if args.regularizer == "orthogonal":
        logger.info("orthogonal: conv_linear=True, penalty=||gram-I||_F^2")
    if args.regularizer == "effective_l2":
        logger.info(
            "effective_l2: bn_folded=True, fan_in_norm=True, layer_reduce=mean"
        )
    if args.regularizer == "threshold_l2":
        logger.info(
            "threshold_l2: raw_weight_energy/lambda^2, detach_lambda=True, fan_in_norm=True, layer_reduce=mean"
        )
    if args.regularizer == "l2_sp":
        logger.info("l2_sp: reference=initial_weights, conv_linear=True, layer_reduce=mean")
    if ds not in ("mnist", "diff1d", "toy_diff1d", "diff_1d"):
        if ds in ("imagenet", "imagenet1k"):
            logger.info(
                "ImageNet 建议: -arch resnet18 -lr 0.05 -wd 1e-4 --epochs 90 -b 128"
            )
        else:
            logger.info(
                "CIFAR 建议: -lr 0.1 -wd 5e-4 --epochs 300 -b 128"
            )
    if is_diff1d:
        logger.info(
            "diff1d：回归 y=x1-x2（数据上 x1>=x2）；Linear 无 bias、写死差分；指标为 RMSE"
        )

    if args.mapping_diag_dir.strip():
        map_dir = os.path.abspath(args.mapping_diag_dir.strip())
        os.makedirs(map_dir, exist_ok=True)
        report = dump_mne_mapping_report(
            model,
            map_dir,
            layer_map=args.mne_layer_map,
            quant_level=args.L,
            alpha=float(args.calibrated_mne_alpha),
            tau=float(args.calibrated_mne_tau),
            risk_min=float(args.calibrated_mne_risk_min),
            risk_max=float(args.calibrated_mne_risk_max),
            q_assignment=args.calibrated_mne_q_assignment,
            onesided=bool(args.calibrated_mne_onesided),
        )
        logger.info(
            "mapping_diag layer_map=%s matched=%s body_param_ratio=%.4f "
            "identity_vs_l2wo_relerr=%.3e p_gt_tau=%.4f"
            % (
                args.mne_layer_map,
                report["matched_layers_over_total"],
                float(report["body_param_ratio"]),
                float(report["identity_vs_l2wo_relerr"]),
                float(report.get("p_gt_tau", float("nan"))),
            )
        )

    extra_save = set()
    if args.save_epochs.strip():
        extra_save = {int(x) for x in args.save_epochs.split(",") if x.strip()}

    def _epoch_reg_coeff(epoch: int) -> float:
        if reg_loss_fn is None or args.reg_warmup_epochs <= 0:
            return args.reg_coeff
        t = min(1.0, float(epoch + 1) / float(args.reg_warmup_epochs))
        if args.reg_warmup_schedule == "cosine":
            warmup_scale = 0.5 * (1.0 - math.cos(math.pi * t))
        else:
            warmup_scale = t
        return args.reg_coeff * warmup_scale

    def _epoch_calibrated_mne_alpha(epoch: int) -> float:
        target = float(args.calibrated_mne_alpha)
        start = int(args.calibrated_mne_alpha_start_epoch)
        warmup = int(args.calibrated_mne_alpha_warmup_epochs)
        if epoch < start:
            return 0.0
        if warmup <= 0:
            return target
        progress = min(1.0, float(epoch - start + 1) / float(warmup))
        return target * progress

    probe_epochs = set()
    probe_csv_path = None
    probe_batch = None
    if args.grad_probe:
        if reg_loss_fn is None:
            raise ValueError(
                "--grad_probe 需要显式正则（例如 --regularizer manual_l2 或 mne_l2）；"
                "weight_decay / weight_decay_weights_only 没有可分解的 reg loss"
            )
        probe_epochs = set(parse_probe_epochs(args.grad_probe_epochs, args.epochs))
        probe_csv_path = (
            args.grad_probe_csv
            if args.grad_probe_csv
            else os.path.join(log_dir, "%s_grad_probe.csv" % (identifier,))
        )
        probe_images, probe_labels = next(iter(train_loader))
        probe_batch = (probe_images, probe_labels)
        logger.info(
            "grad_probe enabled: epochs=%s csv=%s"
            % (sorted(probe_epochs), probe_csv_path)
        )

    def _maybe_run_grad_probe(epoch: int, epoch_reg_coeff: float) -> None:
        if epoch not in probe_epochs or probe_batch is None:
            return
        stage = stage_name_for_epoch(epoch, sorted(probe_epochs))
        images, labels = probe_batch
        rows = probe_ce_vs_reg_grads(
            model,
            images,
            labels,
            criterion,
            reg_loss_fn=reg_loss_fn,
            reg_coeff=epoch_reg_coeff,
            T=args.time,
            quant_level=args.L,
            stage=stage,
            epoch=epoch,
            epochs=args.epochs,
            regularizer=args.regularizer,
        )
        append_probe_csv(probe_csv_path, rows)
        logger.info("grad_probe stage=%s epoch=%d" % (stage, epoch))
        for line in summarize_probe_rows(rows):
            logger.info(line)

    for epoch in range(args.epochs):
        if args.regularizer == "calibrated_mne_l2":
            calibrated_mne_state["alpha"] = _epoch_calibrated_mne_alpha(epoch)
        epoch_reg_coeff = _epoch_reg_coeff(epoch)
        if is_diff1d:
            loss, mae = train_reg(
                model,
                device,
                train_loader,
                criterion,
                optimizer,
                args.time,
                quant_level=args.L,
                reg_loss_fn=reg_loss_fn,
                reg_coeff=epoch_reg_coeff,
            )
            logger.info(
                "Epoch:[{}/{}]\t loss(sum)={:.5f}\t train_MAE={:.6f}".format(
                    epoch, args.epochs, loss, mae
                )
            )
            _maybe_run_grad_probe(epoch, epoch_reg_coeff)
            scheduler.step()
            tmp = val_reg(
                model, test_loader, T=args.time, device=device
            )
            logger.info(
                "Epoch:[{}/{}]\t Test RMSE={:.6f}\n".format(
                    epoch, args.epochs, tmp
                )
            )
            is_better = tmp < best_rmse
            if is_better:
                best_rmse = tmp
            should_save = (
                (args.ckpt_save_mode == "best" and is_better)
                or (args.ckpt_save_mode == "last" and epoch == args.epochs - 1)
            )
            if should_save:
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
                quant_level=args.L,
                reg_loss_fn=reg_loss_fn,
                reg_coeff=epoch_reg_coeff,
            )
            logger.info(
                "Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}".format(
                    epoch, args.epochs, loss, acc
                )
            )
            if args.regularizer == "calibrated_mne_l2" and hasattr(
                model, "_calibrated_mne_stats"
            ):
                stats = getattr(model, "_calibrated_mne_epoch_stats", None) or model._calibrated_mne_stats
                logger.info(
                    "  calibrated_mne: assign=%s q_max=%s alpha=%.4f q_mean=%.6f "
                    "q_std=%.6f q_range=[%.4g, %.4g] p_gt_tau=%.4f p_at_qmax=%.4f"
                    % (
                        args.calibrated_mne_q_assignment,
                        (
                            "none"
                            if float(args.calibrated_mne_q_max) <= 0
                            else ("%.4g" % args.calibrated_mne_q_max)
                        ),
                        float(getattr(model, "_calibrated_mne_stats", {}).get("alpha", calibrated_mne_state["alpha"])),
                        float(stats.get("q_mean", 1.0)),
                        float(stats.get("q_std", 0.0)),
                        float(stats.get("q_min", 1.0)),
                        float(stats.get("q_max", 1.0)),
                        float(stats.get("p_gt_tau", 0.0)),
                        float(stats.get("p_at_qmax", 0.0)),
                    )
                )
            _maybe_run_grad_probe(epoch, epoch_reg_coeff)
            if args.regularizer in ("pc_mne", "margin_mne") and hasattr(model, "_pc_mne_stats"):
                st = model._pc_mne_stats
                logger.info(
                    "  pc_mne_stats: zero_rate={:.4f} sat_rate={:.4f} mean_s={:.4g} "
                    "mean_d={:.4g} mean_rho={} lambda={:.4g}".format(
                        float(st.get("zero_rate", float("nan"))),
                        float(st.get("sat_rate", float("nan"))),
                        float(st.get("mean_s", float("nan"))),
                        float(st.get("mean_d", float("nan"))),
                        ("%.4g" % st["mean_rho"]) if "mean_rho" in st else "n/a",
                        float(st.get("lambda", float("nan"))),
                    )
                )
            scheduler.step()
            tmp = val(model, test_loader, T=args.time, device=device)
            logger.info(
                "Epoch:[{}/{}]\t Test acc={:.3f}\n".format(
                    epoch, args.epochs, tmp
                )
            )
            if args.regularizer in ("calibrated_mne_l2", "mne_l2") and (
                args.regularizer == "calibrated_mne_l2" or args.epoch_log_csv.strip()
            ):
                epoch_stats = getattr(model, "_calibrated_mne_epoch_stats", {})
                last_stats = getattr(model, "_calibrated_mne_stats", {})
                merged = dict(last_stats)
                merged.update(epoch_stats)
                grads = _ce_reg_ratio(epoch_reg_coeff)
                _append_epoch_log(
                    {
                        "epoch": epoch,
                        "alpha": float(calibrated_mne_state["alpha"]),
                        "q_assignment": args.calibrated_mne_q_assignment,
                        "q_mean": float(merged.get("q_mean", 1.0)),
                        "q_std": float(merged.get("q_std", 0.0)),
                        "q_min": float(merged.get("q_min", 1.0)),
                        "q_max": float(merged.get("q_max", 1.0)),
                        "q_os_mean": float(merged.get("q_os_mean", 1.0)),
                        "q_os_std": float(merged.get("q_os_std", 0.0)),
                        "p_gt_tau": float(merged.get("p_gt_tau", 0.0)),
                        "p_at_qmax": float(merged.get("p_at_qmax", 0.0)),
                        "q_cap": float(args.calibrated_mne_q_max),
                        "reg_grad_norm": grads["reg_grad_norm"],
                        "reg_coeff_grad_norm": grads["reg_coeff_grad_norm"],
                        "ce_grad_norm": grads["ce_grad_norm"],
                        "reg_ce_ratio": grads["reg_ce_ratio"],
                        "ann_train_acc": acc,
                        "ann_test_acc": tmp,
                        "train_loss": loss,
                    }
                )

            is_better = best_acc < tmp
            if is_better:
                best_acc = tmp
            should_save = (
                (args.ckpt_save_mode == "best" and is_better)
                or (args.ckpt_save_mode == "last" and epoch == args.epochs - 1)
            )
            if should_save:
                filename = os.path.join(log_dir, "%s.pth" % (identifier,))
                print("Saving model to %s" % (filename,))
                torch.save(model.state_dict(), filename)
            if epoch in extra_save:
                snap = os.path.join(log_dir, "%s_ep%d.pth" % (identifier, epoch))
                print("Saving snapshot to %s" % (snap,))
                torch.save(model.state_dict(), snap)

    if is_diff1d:
        logger.info("Best Test RMSE={:.6f}".format(best_rmse))
        _log_diff1d_param_values(model, logger)
    else:
        logger.info("Best Test acc={:.3f}".format(best_acc))


if __name__ == "__main__":
    main()
