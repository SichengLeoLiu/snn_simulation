from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Models import IF, modelpool  # noqa: E402
from Preprocess import datapool  # noqa: E402
from utils import (  # noqa: E402
    compute_l2_calibrated_mne_regularization,
    get_torch_device,
    seed_all,
    train,
    val,
)


BRANCHES = (
    {
        "key": "A_l2_wo",
        "label": "A: weights-only L2",
        "normalization": None,
        "freeze_scale": False,
    },
    {
        "key": "B_global_cmne",
        "label": "B: global Calibrated MNE",
        "normalization": "global",
        "freeze_scale": False,
    },
    {
        "key": "C_layerwise_cmne",
        "label": "C: layerwise Calibrated MNE",
        "normalization": "layerwise",
        "freeze_scale": False,
    },
    {
        "key": "D_global_cmne_frozen",
        "label": "D: global MNE, BN/lambda frozen",
        "normalization": "global",
        "freeze_scale": True,
    },
)


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _make_model(args, device: torch.device) -> nn.Module:
    model = modelpool(args.model, "fashion_mnist")
    model.set_L(args.L)
    model.set_T(0)
    model.set_mode("normal")
    return model.to(device)


def _make_optimizer(model: nn.Module, lr: float, weight_decay: float):
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
    return torch.optim.SGD(
        [
            {"params": decay_parameters, "weight_decay": weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=lr,
        momentum=0.9,
    )


def _set_optimizer_weight_decay(optimizer, value: float) -> None:
    for index, group in enumerate(optimizer.param_groups):
        group["weight_decay"] = float(value) if index == 0 else 0.0


def _capture_rng_state(device: torch.device) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        state["mps"] = torch.mps.get_rng_state()
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _freeze_bn_affine_and_lambda(model: nn.Module) -> dict[str, int]:
    bn_parameters = 0
    thresholds = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            for parameter in (module.weight, module.bias):
                if parameter is not None:
                    parameter.requires_grad_(False)
                    bn_parameters += parameter.numel()
        if isinstance(module, IF):
            module.thresh.requires_grad_(False)
            thresholds += module.thresh.numel()
    return {"bn_affine_parameters": bn_parameters, "lambda_parameters": thresholds}


def _alpha_for_epoch(epoch: int, args) -> float:
    progress = epoch - args.warmup_epochs + 1
    if progress <= 0:
        return 0.0
    if args.alpha_ramp_epochs <= 0:
        return float(args.alpha)
    return float(args.alpha) * min(1.0, progress / float(args.alpha_ramp_epochs))


def _parameter_stats(model: nn.Module) -> dict[str, float]:
    lambdas = torch.cat(
        [module.thresh.detach().cpu().reshape(-1) for module in model.modules() if isinstance(module, IF)]
    )
    gammas = torch.cat(
        [
            module.weight.detach().cpu().reshape(-1)
            for module in model.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
            and module.weight is not None
        ]
    )
    return {
        "mean_lambda": float(lambdas.mean()),
        "min_lambda": float(lambdas.min()),
        "max_lambda": float(lambdas.max()),
        "mean_abs_gamma": float(gammas.abs().mean()),
        "gamma_rms": float(gammas.square().mean().sqrt()),
        "min_gamma": float(gammas.min()),
        "max_gamma": float(gammas.max()),
    }


def _read_noise_matrix(path: Path, level: int) -> dict[float, float]:
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(float(row["L"])) != int(level):
                continue
            return {
                float(key): float(value)
                for key, value in row.items()
                if key not in ("L", "T") and value not in (None, "")
            }
    raise ValueError(f"No L={level} row in {path}")


def _noise_sweep(checkpoint: Path, branch_key: str, args) -> dict[float, float]:
    output_dir = args.out_root / branch_key / "noise"
    matrix = output_dir / (
        f"noise_sweep_matrix_fashion_mnist_{args.model}_T{args.T}"
        f"_mode_rate_uniform_schedule_normal_seed_{args.seed}.csv"
    )
    command = [
        sys.executable,
        str(ROOT / "main_test.py"),
        "-data",
        "fashion_mnist",
        "-arch",
        args.model,
        "-L",
        str(args.L),
        "-T",
        str(args.T),
        "--mode",
        "rate_uniform",
        "--spike_schedule",
        "normal",
        "--noise_sweep",
        "--noise_sigma_start",
        "0",
        "--noise_sigma_end",
        "1",
        "--noise_sigma_step",
        str(args.noise_step),
        "--noise_output_dir",
        str(output_dir),
        "--first_layer_noise_position",
        "post_input_if",
        "--first_layer_noise_type",
        "gaussian",
        "-w",
        str(checkpoint),
        "-suffix",
        branch_key,
        "-b",
        str(args.batch_size),
        "-j",
        str(args.workers),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return _read_noise_matrix(matrix, args.L)


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_training(history: list[dict], args) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    warmup = [row for row in history if row["branch"] == "shared_warmup"]
    ax.plot(
        [row["epoch"] for row in warmup],
        [row["val_acc"] for row in warmup],
        color="0.35",
        marker="o",
        label="Shared weights-only L2 warm-up",
    )
    for branch in BRANCHES:
        rows = [row for row in history if row["branch"] == branch["key"]]
        ax.plot(
            [row["epoch"] for row in rows],
            [row["val_acc"] for row in rows],
            marker="o",
            markersize=3,
            label=branch["label"],
        )
    ax.axvline(args.warmup_epochs, color="0.25", linestyle="--", linewidth=1)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("ANN validation accuracy (%)")
    ax.set_title(f"Fashion-MNIST {args.model}: shared-checkpoint fork")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_root / "training_validation_accuracy.png", dpi=220)
    plt.close(fig)


def _plot_noise(noise_rows: list[dict], args) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    baseline = {
        float(row["sigma"]): float(row["accuracy"])
        for row in noise_rows
        if row["branch"] == "A_l2_wo"
    }
    for branch in BRANCHES:
        rows = [row for row in noise_rows if row["branch"] == branch["key"]]
        sigmas = [float(row["sigma"]) for row in rows]
        accuracy = [float(row["accuracy"]) for row in rows]
        axes[0].plot(sigmas, accuracy, marker="o", label=branch["label"])
        axes[1].plot(
            sigmas,
            [acc - baseline[sigma] for sigma, acc in zip(sigmas, accuracy)],
            marker="o",
            label=branch["label"],
        )
    axes[0].set_ylabel("SNN accuracy (%)")
    axes[1].set_ylabel("Accuracy difference vs A (points)")
    axes[1].axhline(0.0, color="0.25", linewidth=1)
    for ax in axes:
        ax.set_xlabel("Absolute Gaussian noise sigma (post-input-IF)")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Fashion-MNIST {args.model}: calibrated MNE fork")
    fig.tight_layout()
    fig.savefig(args.out_root / "snn_noise_curves_and_delta.png", dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Four-way Fashion-MNIST c16-c32 fork from one epoch-5 checkpoint."
    )
    parser.add_argument("--model", default="cnn2_c16_c32")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--warmup-epochs", default=5, type=int)
    parser.add_argument("--alpha-ramp-epochs", default=10, type=int)
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--risk-min", default=0.5, type=float)
    parser.add_argument("--risk-max", default=2.0, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--L", default=16, type=int)
    parser.add_argument("--T", default=16, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--noise-step", default=0.1, type=float)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=(
            ROOT.parent
            / "important_results"
            / "fashion_calibrated_mne_shared_fork_c16c32_seed42_ep30"
        ),
    )
    args = parser.parse_args()
    if not 0 < args.warmup_epochs < args.epochs:
        parser.error("Expected 0 < warmup-epochs < epochs.")
    args.out_root = args.out_root.resolve()
    return args


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    device = get_torch_device(args.device)
    print(f"[DEVICE] {device}", flush=True)
    seed_all(args.seed)
    train_loader, test_loader = datapool(
        "fashion_mnist",
        args.batch_size,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    criterion = nn.CrossEntropyLoss().to(device)

    warm_model = _make_model(args, device)
    warm_optimizer = _make_optimizer(
        warm_model, lr=args.lr, weight_decay=args.weight_decay
    )
    warm_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        warm_optimizer, T_max=args.epochs
    )
    history = []
    warm_best_acc = float("-inf")
    warm_best_state = None
    warm_best_epoch = 0

    for epoch in range(args.warmup_epochs):
        loss, train_acc = train(
            warm_model,
            device,
            train_loader,
            criterion,
            warm_optimizer,
            T=0,
            quant_level=args.L,
        )
        warm_scheduler.step()
        val_acc = float(val(warm_model, test_loader, T=0, device=device, verbose=False))
        if val_acc > warm_best_acc:
            warm_best_acc = val_acc
            warm_best_state = _cpu_state_dict(warm_model)
            warm_best_epoch = epoch + 1
        row = {
            "branch": "shared_warmup",
            "epoch": epoch + 1,
            "alpha": 0.0,
            "lr": warm_optimizer.param_groups[0]["lr"],
            "loss_sum": loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        history.append(row)
        print(f"[WARMUP] {row}", flush=True)

    fork_model_state = _cpu_state_dict(warm_model)
    fork_optimizer_state = copy.deepcopy(warm_optimizer.state_dict())
    fork_scheduler_state = copy.deepcopy(warm_scheduler.state_dict())
    fork_rng_state = _capture_rng_state(device)
    torch.save(
        {
            "epoch": args.warmup_epochs,
            "model": fork_model_state,
            "optimizer": fork_optimizer_state,
            "scheduler": fork_scheduler_state,
            "warmup_best_acc": warm_best_acc,
            "warmup_best_epoch": warm_best_epoch,
        },
        args.out_root / f"shared_epoch{args.warmup_epochs}_training_checkpoint.pth",
    )

    branch_records = []
    for branch in BRANCHES:
        model = _make_model(args, device)
        model.load_state_dict(fork_model_state, strict=True)
        optimizer = _make_optimizer(
            model, lr=args.lr, weight_decay=args.weight_decay
        )
        optimizer.load_state_dict(fork_optimizer_state)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        scheduler.load_state_dict(fork_scheduler_state)

        if branch["normalization"] is None:
            _set_optimizer_weight_decay(optimizer, args.weight_decay)
        else:
            _set_optimizer_weight_decay(optimizer, 0.0)

        frozen = {"bn_affine_parameters": 0, "lambda_parameters": 0}
        if branch["freeze_scale"]:
            frozen = _freeze_bn_affine_and_lambda(model)
        _restore_rng_state(fork_rng_state, device)

        best_acc = warm_best_acc
        best_state = copy.deepcopy(warm_best_state)
        best_epoch = warm_best_epoch
        final_acc = warm_best_acc

        for epoch in range(args.warmup_epochs, args.epochs):
            alpha = (
                0.0
                if branch["normalization"] is None
                else _alpha_for_epoch(epoch, args)
            )
            reg_loss_fn = None
            if branch["normalization"] is not None:
                normalization = branch["normalization"]

                def reg_loss_fn(m, _t, q, alpha=alpha, normalization=normalization):
                    return compute_l2_calibrated_mne_regularization(
                        m,
                        quant_level=args.L if q is None else q,
                        alpha=alpha,
                        risk_min=args.risk_min,
                        risk_max=args.risk_max,
                        fold_bn=True,
                        normalization=normalization,
                    )

            loss, train_acc = train(
                model,
                device,
                train_loader,
                criterion,
                optimizer,
                T=0,
                quant_level=args.L,
                reg_loss_fn=reg_loss_fn,
                reg_coeff=args.weight_decay,
            )
            scheduler.step()
            final_acc = float(val(model, test_loader, T=0, device=device, verbose=False))
            if final_acc > best_acc:
                best_acc = final_acc
                best_state = _cpu_state_dict(model)
                best_epoch = epoch + 1
            row = {
                "branch": branch["key"],
                "epoch": epoch + 1,
                "alpha": alpha,
                "lr": optimizer.param_groups[0]["lr"],
                "loss_sum": loss,
                "train_acc": train_acc,
                "val_acc": final_acc,
            }
            history.append(row)
            print(f"[TRAIN] {row}", flush=True)

        checkpoint = args.out_root / f"{branch['key']}_best.pth"
        last_checkpoint = args.out_root / f"{branch['key']}_last.pth"
        torch.save(best_state, checkpoint)
        torch.save(_cpu_state_dict(model), last_checkpoint)
        model.load_state_dict(best_state, strict=True)
        stats = _parameter_stats(model)
        if branch["normalization"] is None:
            compute_l2_calibrated_mne_regularization(
                model, quant_level=args.L, alpha=0.0
            )
        else:
            compute_l2_calibrated_mne_regularization(
                model,
                quant_level=args.L,
                alpha=args.alpha,
                risk_min=args.risk_min,
                risk_max=args.risk_max,
                fold_bn=True,
                normalization=branch["normalization"],
            )
        q_stats = model._calibrated_mne_stats
        branch_records.append(
            {
                "branch": branch["key"],
                "label": branch["label"],
                "normalization": branch["normalization"] or "l2",
                "freeze_bn_lambda": branch["freeze_scale"],
                "frozen_bn_affine_parameters": frozen["bn_affine_parameters"],
                "frozen_lambda_parameters": frozen["lambda_parameters"],
                "best_epoch": best_epoch,
                "best_ann_acc": best_acc,
                "final_ann_acc": final_acc,
                **stats,
                "q_mean": float(q_stats["q_mean"]),
                "q_min": float(q_stats["q_min"]),
                "q_max": float(q_stats["q_max"]),
                "layer_q_mean_min": float(q_stats["layer_q_mean_min"]),
                "layer_q_mean_max": float(q_stats["layer_q_mean_max"]),
                "checkpoint": str(checkpoint),
            }
        )
        del model, optimizer, scheduler
        if device.type == "mps":
            torch.mps.empty_cache()

    noise_rows = []
    summaries = []
    for record in branch_records:
        curve = _noise_sweep(Path(record["checkpoint"]), record["branch"], args)
        for sigma, accuracy in sorted(curve.items()):
            noise_rows.append(
                {
                    "branch": record["branch"],
                    "label": record["label"],
                    "sigma": sigma,
                    "accuracy": accuracy,
                }
            )
        clean = curve[0.0]
        noisy = curve[1.0]
        summaries.append(
            {
                **record,
                "snn_sigma0": clean,
                "snn_sigma1": noisy,
                "snn_drop": clean - noisy,
                "mean_noise_accuracy": float(np.mean(list(curve.values()))),
            }
        )

    _write_rows(args.out_root / "training_history.csv", history)
    _write_rows(args.out_root / "noise_sweep.csv", noise_rows)
    _write_rows(args.out_root / "summary.csv", summaries)
    _plot_training(history, args)
    _plot_noise(noise_rows, args)
    with (args.out_root / "experiment_config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2, default=str)
    print(f"[DONE] {args.out_root / 'summary.csv'}", flush=True)
    for row in summaries:
        print(
            "[RESULT] {branch}: ANN={best_ann_acc:.3f}, SNN0={snn_sigma0:.3f}, "
            "SNN1={snn_sigma1:.3f}, drop={snn_drop:.3f}".format(**row),
            flush=True,
        )


if __name__ == "__main__":
    main()
