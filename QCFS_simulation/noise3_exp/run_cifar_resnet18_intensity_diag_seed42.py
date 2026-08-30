#!/usr/bin/env python3
"""CIFAR ResNet-18 seed-42 intensity-matching diagnostic.

Do not retune α/τ. Compare five conditions under the frozen onesided recipe
α=4, τ=0.5, β=5e-4, r_max=8, warmup 30/50:

    1. identity            q=1
    2. strength_norm       mean-normalized strength-only
    3. onesided_norm       mean-normalized ResNet-aware One-sided
    4. mne_legacy          Old MNE legacy map
    5. mne_aware_gmatch    Old MNE ResNet-aware, ||∇R|| matched to legacy

Mean-normalize uses
    tilde q_i = q_i / mean(q),   R = (β/2) Σ_i tilde q_i ||W_i||^2
so E[tilde q]=1. Strength-only then collapses to q=1; onesided keeps the
heterogeneous shape at L2-wo strength.

Reuse seed-42 identity / Old MNE-legacy checkpoints from the mapping
diagnostic when present. New trains: strength_norm, onesided_norm,
mne_aware_gmatch.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Models import modelpool  # noqa: E402
from run_cifar_resnet18_four_regs_5seed import ARCH  # noqa: E402
from run_cifar_vgg16_onesided_q_assignment_ablation import (  # noqa: E402
    ALPHA,
    ALPHA_START,
    ALPHA_WARMUP,
    BETA,
    EPOCHS,
    LR,
    LVAL,
    RISK_MAX,
    RISK_MIN,
    TAU,
    TEST_T,
    snn_metrics,
    sweep,
    test_loader,
    val_loader,
    write_csv,
)
from utils import (  # noqa: E402
    ce_vs_reg_grad_ratio,
    collect_weight_layer_matches,
    compute_l2_calibrated_mne_regularization,
    compute_mne_l2_regularization,
    dump_mne_mapping_report,
    get_torch_device,
    summarize_weight_layer_matches,
)

SEED = 42
MNE_RC = 1e-4
MAPPING_REUSE_DEFAULT = Path(
    "/scratch/gs14/sl9144/snn_results/cifar_resnet18_mapping_diag_seed42"
)
FOUR_REGS_DEFAULT = Path(
    "/scratch/gs14/sl9144/snn_results/cifar_resnet18_four_regs_5seed"
)

ONESIDED_FLAGS = [
    "--calibrated_mne_alpha", str(ALPHA),
    "--calibrated_mne_onesided",
    "--calibrated_mne_tau", str(TAU),
    "--calibrated_mne_risk_min", str(RISK_MIN),
    "--calibrated_mne_risk_max", str(RISK_MAX),
    "--calibrated_mne_alpha_start_epoch", str(ALPHA_START),
    "--calibrated_mne_alpha_warmup_epochs", str(ALPHA_WARMUP),
]

VARIANTS = {
    "identity": {
        "label": "q=1",
        "regularizer": "calibrated_mne_l2",
        "weight_decay": 0.0,
        "reg_coeff": BETA,
        "layer_map": "resnet",
        "q_assignment": "identity",
        "mean_normalize_q": False,
        "grad_match": "",
        "reuse": (
            ("mapping", "mapdiag_identity"),
        ),
        "extra": ONESIDED_FLAGS + ["--calibrated_mne_q_assignment", "identity"],
    },
    "strength_norm": {
        "label": "Mean-normalized strength-only",
        "regularizer": "calibrated_mne_l2",
        "weight_decay": 0.0,
        "reg_coeff": BETA,
        "layer_map": "resnet",
        "q_assignment": "strength",
        "mean_normalize_q": True,
        "grad_match": "",
        "reuse": None,
        "extra": ONESIDED_FLAGS
        + [
            "--calibrated_mne_q_assignment",
            "strength",
            "--calibrated_mne_mean_normalize_q",
        ],
    },
    "onesided_norm": {
        "label": "Mean-normalized ResNet-aware One-sided",
        "regularizer": "calibrated_mne_l2",
        "weight_decay": 0.0,
        "reg_coeff": BETA,
        "layer_map": "resnet",
        "q_assignment": "risk",
        "mean_normalize_q": True,
        "grad_match": "",
        "reuse": None,
        "extra": ONESIDED_FLAGS
        + [
            "--calibrated_mne_q_assignment",
            "risk",
            "--calibrated_mne_mean_normalize_q",
        ],
    },
    "mne_legacy": {
        "label": "Old MNE legacy",
        "regularizer": "mne_l2",
        "weight_decay": 0.0,
        "reg_coeff": MNE_RC,
        "layer_map": "legacy",
        "q_assignment": "risk",
        "mean_normalize_q": False,
        "grad_match": "",
        "reuse": (
            ("mapping", "mapdiag_mne_legacy"),
            ("four_regs", "r18_mne"),
        ),
        "extra": ["--mne_detach_lambda"],
    },
    "mne_aware_gmatch": {
        "label": "Old MNE ResNet-aware, ||∇R|| matched to legacy",
        "regularizer": "mne_l2",
        "weight_decay": 0.0,
        "reg_coeff": MNE_RC,
        "layer_map": "resnet",
        "q_assignment": "risk",
        "mean_normalize_q": False,
        "grad_match": "legacy",
        "reuse": None,
        "extra": [
            "--mne_detach_lambda",
            "--mne_grad_match_layer_map",
            "legacy",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default=None)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument(
        "--reuse-root",
        type=Path,
        default=Path(os.environ.get("REUSE_ROOT", str(MAPPING_REUSE_DEFAULT))),
    )
    parser.add_argument(
        "--four-regs-root",
        type=Path,
        default=Path(os.environ.get("FOUR_REGS_ROOT", str(FOUR_REGS_DEFAULT))),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar_resnet18_intensity_diag_seed42",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    if args.variant is None:
        parser.error("specify --variant")
    return args


def config_name(variant: str) -> str:
    return f"intdiag_{variant}"


def suffix(args) -> str:
    return f"{config_name(args.variant)}_seed{args.seed}_L{LVAL}_trainT0"


def ckpt_filename(args) -> str:
    return f"{ARCH}_L[{LVAL}]_{suffix(args)}.pth"


def cfg_dir(args) -> Path:
    return args.out_root / args.dataset / config_name(args.variant)


def ckpt_path(args) -> Path:
    return cfg_dir(args) / "checkpoints" / ckpt_filename(args)


def reuse_ckpt(args) -> Path | None:
    spec = VARIANTS[args.variant]
    if spec["reuse"] is None:
        return None
    candidates = []
    for kind, folder in spec["reuse"]:
        if kind == "mapping":
            name = f"{ARCH}_L[{LVAL}]_{folder}_seed{args.seed}_L{LVAL}_trainT0.pth"
            candidates.append(
                args.reuse_root / args.dataset / folder / "checkpoints" / name
            )
        elif kind == "four_regs":
            name = f"{ARCH}_L[{LVAL}]_{folder}_seed{args.seed}_L{LVAL}_trainT0.pth"
            candidates.append(
                args.four_regs_root
                / args.dataset
                / folder
                / f"seed{args.seed}"
                / "checkpoints"
                / name
            )
        else:
            raise ValueError(f"unknown reuse kind {kind!r}")
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_model(ckpt: Path, device, dataset: str, layer_map: str, grad_match: str):
    model = modelpool(ARCH, dataset)
    model._mne_layer_map = layer_map
    model._mne_grad_match_layer_map = grad_match or None
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.set_L(LVAL)
    model.set_T(TEST_T)
    model.set_mode("rate_uniform")
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule("normal")
    if hasattr(model, "set_first_layer_input_noise_position"):
        model.set_first_layer_input_noise_position("post_input_if")
    if hasattr(model, "set_first_layer_input_noise_type"):
        model.set_first_layer_input_noise_type("gaussian")
    return model.to(device).eval()


def make_reg_fn(spec):
    if spec["regularizer"] == "mne_l2":
        return lambda m, t, q: compute_mne_l2_regularization(
            m,
            quant_level=LVAL,
            detach_lambda=True,
            grad_match_layer_map=spec["grad_match"] or None,
        )
    return lambda m, t, q: compute_l2_calibrated_mne_regularization(
        m,
        quant_level=LVAL,
        alpha=ALPHA,
        risk_min=RISK_MIN,
        risk_max=RISK_MAX,
        onesided=True,
        tau=TAU,
        q_assignment=spec["q_assignment"],
        mean_normalize_q=spec["mean_normalize_q"],
    )


def train(args) -> Path:
    spec = VARIANTS[args.variant]
    out = cfg_dir(args)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_path(args)
    if ckpt.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {ckpt}", flush=True)
        return ckpt
    reused = None if args.retrain else reuse_ckpt(args)
    if reused is not None and not args.test_only:
        shutil.copy2(reused, ckpt)
        print(f"[REUSE CKPT] {reused} -> {ckpt}", flush=True)
        return ckpt
    if args.test_only:
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        return ckpt
    cmd = [
        sys.executable,
        str(ROOT / "main_train.py"),
        "-data", args.dataset,
        "-arch", ARCH,
        "-L", str(LVAL),
        "-T", "0",
        "--epochs", str(args.epochs),
        "-lr", str(LR),
        "-b", str(args.batch_size),
        "-j", str(args.workers),
        "--seed", str(args.seed),
        "--device", args.device,
        "--spike_schedule", "normal",
        "--ckpt-save-mode", "best",
        "--ckpt-dir", str(ckpt_dir),
        "-suffix", suffix(args),
        "--regularizer", spec["regularizer"],
        "--weight_decay", str(spec["weight_decay"]),
        "--mne_layer_map", spec["layer_map"],
        "--epoch_log_csv", str(out / "epoch_log.csv"),
        "--mapping_diag_dir", str(out / "mapping_init"),
    ]
    if spec["reg_coeff"] is not None:
        cmd += ["--reg_coeff", str(spec["reg_coeff"])]
    cmd += list(spec["extra"])
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"training finished but missing {ckpt}")
    return ckpt


def main() -> None:
    args = parse_args()
    spec = VARIANTS[args.variant]
    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] {args.dataset} ResNet-18 {spec['label']} seed={args.seed} "
        f"layer_map={spec['layer_map']} assign={spec['q_assignment']} "
        f"mean_norm={spec['mean_normalize_q']} grad_match={spec['grad_match'] or 'none'}",
        flush=True,
    )
    ckpt = train(args)
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = load_model(
        ckpt, device, args.dataset, spec["layer_map"], spec["grad_match"]
    )
    report = dump_mne_mapping_report(
        model,
        out,
        layer_map=spec["layer_map"],
        quant_level=LVAL,
        alpha=ALPHA,
        tau=TAU,
        risk_min=RISK_MIN,
        risk_max=RISK_MAX,
        q_assignment=spec["q_assignment"],
        onesided=True,
        mean_normalize_q=spec["mean_normalize_q"],
        extra={
            "legacy_match": summarize_weight_layer_matches(
                collect_weight_layer_matches(model, "legacy")
            ),
            "resnet_match": summarize_weight_layer_matches(
                collect_weight_layer_matches(model, "resnet")
            ),
        },
    )
    eval_args = argparse.Namespace(
        dataset=args.dataset,
        batch_size=args.batch_size,
        workers=0,
    )
    val_rows = sweep(model, val_loader(eval_args, pin), device, "val", args.seed)
    write_csv(out / "val_sweep.csv", val_rows)
    test_rows = sweep(model, test_loader(eval_args, pin), device, "test", args.seed)
    write_csv(out / "test_sweep.csv", test_rows)
    grads = {
        "ce_grad_norm": float("nan"),
        "reg_grad_norm": float("nan"),
        "reg_coeff_grad_norm": float("nan"),
        "reg_ce_ratio": float("nan"),
        "ce_loss": float("nan"),
        "reg_loss": float("nan"),
    }
    try:
        images, labels = next(iter(val_loader(eval_args, pin)))
        grads = ce_vs_reg_grad_ratio(
            model,
            images,
            labels,
            nn.CrossEntropyLoss(),
            make_reg_fn(spec),
            0,
            LVAL,
            spec["reg_coeff"],
        )
    except Exception as exc:
        print(f"[WARN] ce/reg grad ratio failed after sweep: {exc}", flush=True)
    match_stats = getattr(model, "_mne_grad_match_stats", {}) or {}
    card = {
        "config": config_name(args.variant),
        "label": spec["label"],
        "variant": args.variant,
        "dataset": args.dataset,
        "arch": ARCH,
        "seed": args.seed,
        "layer_map": spec["layer_map"],
        "q_assignment": spec["q_assignment"],
        "mean_normalize_q": spec["mean_normalize_q"],
        "grad_match_layer_map": spec["grad_match"] or None,
        "regularizer": spec["regularizer"],
        "checkpoint": str(ckpt),
        "matched_layers_over_total": report["matched_layers_over_total"],
        "body_param_ratio": report["body_param_ratio"],
        "identity_vs_l2wo_relerr": report["identity_vs_l2wo_relerr"],
        "p_gt_tau": report["p_gt_tau"],
        "q_mean": report["q_mean"],
        "q_mean_pre_norm": report.get("q_mean_pre_norm"),
        "mne_grad_match_scale": match_stats.get("scale"),
        "mne_ref_grad_norm": match_stats.get("ref_grad_norm"),
        "mne_matched_grad_norm": match_stats.get("matched_grad_norm"),
        **grads,
        **snn_metrics(val_rows, "val"),
        **snn_metrics(test_rows, "test"),
    }
    (out / "scorecard.json").write_text(json.dumps(card, indent=2, default=str) + "\n")
    print(json.dumps(card, indent=2, default=str), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
