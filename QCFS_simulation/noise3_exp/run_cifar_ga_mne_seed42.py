#!/usr/bin/env python3
"""CIFAR seed-42 Graph-Aware MNE-L2 screen. Do not retune α/τ.

Frozen onesided recipe: α=4, τ=0.5, β=5e-4, r_max=8, warmup 30/50.
GA replaces only the risk coefficient:
    r^GA = r^local * κ,  κ=1 on single-path (VGG) nodes.

Variants
--------
    onesided       current One-sided
    onesided_norm  current One-sided, mean-normalized q
    ga_nocov       GA-MNE, φ without branch covariance, mean-normalized q
    ga_full        GA-MNE, φ with covariance, mean-normalized q

Architectures: vgg16 (legacy map) and resnet18 (ResNet-aware map).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Models import modelpool  # noqa: E402
from Models.VGG import remap_legacy_vgg_state_dict  # noqa: E402
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
    dump_mne_mapping_report,
    estimate_ga_branch_stats,
    get_torch_device,
    set_basicblock_ga_cache,
    summarize_weight_layer_matches,
)

SEED = 42
MAPPING_REUSE_DEFAULT = Path(
    "/scratch/gs14/sl9144/snn_results/cifar_resnet18_mapping_diag_seed42"
)
INTENSITY_REUSE_DEFAULT = Path(
    "/scratch/gs14/sl9144/snn_results/cifar_resnet18_intensity_diag_seed42"
)
VGG_QABL_DEFAULT = Path(
    "/scratch/gs14/sl9144/snn_results/cifar_vgg16_onesided_q_assignment_ablation_seed42"
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
    "onesided": {
        "label": "One-sided",
        "mean_normalize_q": False,
        "ga_mode": "off",
    },
    "onesided_norm": {
        "label": "One-sided / q̄",
        "mean_normalize_q": True,
        "ga_mode": "off",
    },
    "ga_nocov": {
        "label": "GA-MNE no covariance / q̄",
        "mean_normalize_q": True,
        "ga_mode": "nocov",
    },
    "ga_full": {
        "label": "GA-MNE full / q̄",
        "mean_normalize_q": True,
        "ga_mode": "full",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("vgg16", "resnet18"), required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument(
        "--mapping-root",
        type=Path,
        default=Path(os.environ.get("MAPPING_ROOT", str(MAPPING_REUSE_DEFAULT))),
    )
    parser.add_argument(
        "--intensity-root",
        type=Path,
        default=Path(os.environ.get("INTENSITY_ROOT", str(INTENSITY_REUSE_DEFAULT))),
    )
    parser.add_argument(
        "--vgg-qabl-root",
        type=Path,
        default=Path(os.environ.get("VGG_QABL_ROOT", str(VGG_QABL_DEFAULT))),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar_ga_mne_seed42",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    return args


def layer_map_for(arch: str) -> str:
    return "resnet" if arch == "resnet18" else "legacy"


def config_name(arch: str, variant: str) -> str:
    return f"gadiag_{arch}_{variant}"


def suffix(args) -> str:
    return f"{config_name(args.arch, args.variant)}_seed{args.seed}_L{LVAL}_trainT0"


def ckpt_filename(args) -> str:
    return f"{args.arch}_L[{LVAL}]_{suffix(args)}.pth"


def cfg_dir(args) -> Path:
    return args.out_root / args.arch / args.dataset / config_name(args.arch, args.variant)


def ckpt_path(args) -> Path:
    return cfg_dir(args) / "checkpoints" / ckpt_filename(args)


def reuse_ckpt(args) -> Path | None:
    if args.arch == "resnet18" and args.variant == "onesided":
        name = f"resnet18_L[{LVAL}]_mapdiag_onesided_resnet_seed{args.seed}_L{LVAL}_trainT0.pth"
        path = (
            args.mapping_root
            / args.dataset
            / "mapdiag_onesided_resnet"
            / "checkpoints"
            / name
        )
        return path if path.is_file() else None
    if args.arch == "resnet18" and args.variant == "onesided_norm":
        name = f"resnet18_L[{LVAL}]_intdiag_onesided_norm_seed{args.seed}_L{LVAL}_trainT0.pth"
        path = (
            args.intensity_root
            / args.dataset
            / "intdiag_onesided_norm"
            / "checkpoints"
            / name
        )
        return path if path.is_file() else None
    if args.arch == "vgg16" and args.variant == "onesided":
        folder = "qabl_risk_a4_tau0.5_b0.0005_rmax8"
        name = f"vgg16_L[{LVAL}]_{folder}_seed{args.seed}_L{LVAL}_trainT0.pth"
        candidates = [
            args.vgg_qabl_root / args.dataset / folder / "checkpoints" / name,
            ROOT.parent
            / "important_results"
            / "onesided_q_assignment_ablation_seed42"
            / args.dataset
            / folder
            / "checkpoints"
            / name,
        ]
        for path in candidates:
            if path.is_file():
                return path
    return None


def extra_flags(spec) -> list[str]:
    flags = list(ONESIDED_FLAGS) + ["--calibrated_mne_q_assignment", "risk"]
    if spec["mean_normalize_q"]:
        flags.append("--calibrated_mne_mean_normalize_q")
    if spec["ga_mode"] != "off":
        flags += ["--calibrated_mne_ga", spec["ga_mode"]]
    return flags


def load_model(args, ckpt: Path, device, spec):
    model = modelpool(args.arch, args.dataset)
    model._mne_layer_map = layer_map_for(args.arch)
    model._ga_mne_mode = spec["ga_mode"]
    model._ga_mne_probe_sigma = 0.1
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if args.arch == "vgg16":
        state = remap_legacy_vgg_state_dict(state)
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
    return lambda m, t, q: compute_l2_calibrated_mne_regularization(
        m,
        quant_level=LVAL,
        alpha=ALPHA,
        risk_min=RISK_MIN,
        risk_max=RISK_MAX,
        onesided=True,
        tau=TAU,
        q_assignment="risk",
        mean_normalize_q=spec["mean_normalize_q"],
        ga_mode=spec["ga_mode"],
    )


def _spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def diagnose_ga(model, images, spec) -> dict:
    ga_on = spec["ga_mode"] in ("nocov", "full")
    set_basicblock_ga_cache(model, True)
    with torch.no_grad():
        logits_clean = model(images)
        if ga_on:
            stats = estimate_ga_branch_stats(
                model, include_cov=(spec["ga_mode"] == "full")
            )
        else:
            stats = {}
        noisy = images + torch.randn_like(images)
        logits_noisy = model(noisy)
    set_basicblock_ga_cache(model, False)
    logit_rms = float((logits_noisy - logits_clean).pow(2).mean().sqrt())
    out = {
        "logit_change_rms_sigma1": logit_rms,
        "ga_n_blocks_cached": len(stats),
    }
    if not stats:
        out["spearman_kappa_res_vs_delta_u"] = float("nan")
        out["spearman_kappa_res_vs_phi_res"] = float("nan")
        return out
    kappa_res, delta_u, phi_res = [], [], []
    for block in stats.values():
        kappa_res.append(block["kappa_res"].detach().cpu().numpy())
        delta_u.append(block["delta_u_rms"].detach().cpu().numpy())
        phi_res.append(block["phi_res"].detach().cpu().numpy())
    kappa_res = np.concatenate(kappa_res)
    delta_u = np.concatenate(delta_u)
    phi_res = np.concatenate(phi_res)
    out["spearman_kappa_res_vs_delta_u"] = _spearman(kappa_res, delta_u)
    out["spearman_kappa_res_vs_phi_res"] = _spearman(kappa_res, phi_res)
    out["kappa_res_mean"] = float(kappa_res.mean())
    out["frac_phi_neg_res"] = float((phi_res < 0).mean())
    return out


def train(args, spec) -> Path:
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
        "-arch", args.arch,
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
        "--regularizer", "calibrated_mne_l2",
        "--weight_decay", "0.0",
        "--reg_coeff", str(BETA),
        "--mne_layer_map", layer_map_for(args.arch),
        "--epoch_log_csv", str(out / "epoch_log.csv"),
        "--mapping_diag_dir", str(out / "mapping_init"),
    ] + extra_flags(spec)
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
        f"[INFO] {args.dataset} {args.arch} {spec['label']} seed={args.seed} "
        f"ga={spec['ga_mode']} mean_norm={spec['mean_normalize_q']}",
        flush=True,
    )
    ckpt = train(args, spec)
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = load_model(args, ckpt, device, spec)
    eval_args = argparse.Namespace(
        dataset=args.dataset,
        batch_size=args.batch_size,
        workers=0,
    )
    images, labels = next(iter(val_loader(eval_args, pin)))
    images = images.to(device)
    set_basicblock_ga_cache(model, spec["ga_mode"] in ("nocov", "full"))
    with torch.no_grad():
        model(images)
    report = dump_mne_mapping_report(
        model,
        out,
        layer_map=layer_map_for(args.arch),
        quant_level=LVAL,
        alpha=ALPHA,
        tau=TAU,
        risk_min=RISK_MIN,
        risk_max=RISK_MAX,
        q_assignment="risk",
        onesided=True,
        mean_normalize_q=spec["mean_normalize_q"],
        ga_mode=spec["ga_mode"],
        extra={
            "legacy_match": summarize_weight_layer_matches(
                collect_weight_layer_matches(model, "legacy")
            ),
            "resnet_match": summarize_weight_layer_matches(
                collect_weight_layer_matches(model, "resnet")
            ),
        },
    )
    set_basicblock_ga_cache(model, False)
    diag = diagnose_ga(model, images, spec)
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
        grads = ce_vs_reg_grad_ratio(
            model,
            images,
            labels,
            nn.CrossEntropyLoss(),
            make_reg_fn(spec),
            0,
            LVAL,
            BETA,
        )
    except Exception as exc:
        print(f"[WARN] ce/reg grad ratio failed after sweep: {exc}", flush=True)
    card = {
        "config": config_name(args.arch, args.variant),
        "label": spec["label"],
        "variant": args.variant,
        "dataset": args.dataset,
        "arch": args.arch,
        "seed": args.seed,
        "layer_map": layer_map_for(args.arch),
        "ga_mode": spec["ga_mode"],
        "mean_normalize_q": spec["mean_normalize_q"],
        "regularizer": "calibrated_mne_l2",
        "reg_coeff": BETA,
        "checkpoint": str(ckpt),
        "matched_layers_over_total": report["matched_layers_over_total"],
        "p_gt_tau": report["p_gt_tau"],
        "q_mean": report["q_mean"],
        **{k: report.get(k) for k in (
            "ga_n_blocks",
            "ga_kappa_res_mean",
            "ga_kappa_sc_mean",
            "ga_frac_phi_neg_res",
            "ga_cov_mean",
        )},
        **diag,
        **grads,
        **snn_metrics(val_rows, "val"),
        **snn_metrics(test_rows, "test"),
    }
    (out / "scorecard.json").write_text(json.dumps(card, indent=2, default=str) + "\n")
    print(json.dumps(card, indent=2, default=str), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
