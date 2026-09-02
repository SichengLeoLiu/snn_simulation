#!/usr/bin/env python3
"""CIFAR ResNet-18 seed-42 Graph-Margin One-sided screen.

Frozen onesided envelope: α=4, τ=0.5, β=5e-4, r_max=8, warmup 30/50.
Do not retune α/τ. Do not mean-normalize q. Test is recorded only.

Node/edge risk
--------------
    r_n = p_n D_n
    r_e = r_n m_{e→n}
    q_e = 1 + α [r̂_e − τ]_+

p is the QCFS boundary-crossing probability (BN-free, from activations).
D is detached downstream sensitivity E[||∂L/∂u||²], mean-one per IF node.
m allocates a merge-node's risk onto trainable incoming edges (ResNet
residual vs shortcut). Sequential nodes have m=1.

Variants
--------
    onesided  current BN-local One-sided (reuse mapping-diag checkpoint)
    gm_p      p only
    gm_pd     p D
    gm_pm     p m
    gm_full   p D m
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
    get_torch_device,
    set_basicblock_ga_cache,
    summarize_weight_layer_matches,
    update_graph_margin_stats,
)

ARCH = "resnet18"
SEED = 42
LAYER_MAP = "resnet"
MAPPING_REUSE_DEFAULT = Path(
    "/scratch/gs14/sl9144/snn_results/cifar_resnet18_mapping_diag_seed42"
)

ONESIDED_FLAGS = [
    "--calibrated_mne_alpha", str(ALPHA),
    "--calibrated_mne_onesided",
    "--calibrated_mne_tau", str(TAU),
    "--calibrated_mne_risk_min", str(RISK_MIN),
    "--calibrated_mne_risk_max", str(RISK_MAX),
    "--calibrated_mne_alpha_start_epoch", str(ALPHA_START),
    "--calibrated_mne_alpha_warmup_epochs", str(ALPHA_WARMUP),
    "--calibrated_mne_q_assignment", "risk",
]

VARIANTS = {
    "onesided": {
        "label": "One-sided BN risk",
        "risk_source": "bn",
        "use_p": False,
        "use_d": False,
        "use_m": False,
        "reuse": True,
    },
    "gm_p": {
        "label": "GM-OS p",
        "risk_source": "gm",
        "use_p": True,
        "use_d": False,
        "use_m": False,
        "reuse": False,
    },
    "gm_pd": {
        "label": "GM-OS p D",
        "risk_source": "gm",
        "use_p": True,
        "use_d": True,
        "use_m": False,
        "reuse": False,
    },
    "gm_pm": {
        "label": "GM-OS p m",
        "risk_source": "gm",
        "use_p": True,
        "use_d": False,
        "use_m": True,
        "reuse": False,
    },
    "gm_full": {
        "label": "GM-OS p D m",
        "risk_source": "gm",
        "use_p": True,
        "use_d": True,
        "use_m": True,
        "reuse": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar_resnet18_gm_os_seed42",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    return args


def config_name(variant: str) -> str:
    return f"gmos_{variant}"


def suffix(args) -> str:
    return f"{config_name(args.variant)}_seed{args.seed}_L{LVAL}_trainT0"


def ckpt_filename(args) -> str:
    return f"{ARCH}_L[{LVAL}]_{suffix(args)}.pth"


def cfg_dir(args) -> Path:
    return args.out_root / args.dataset / config_name(args.variant)


def ckpt_path(args) -> Path:
    return cfg_dir(args) / "checkpoints" / ckpt_filename(args)


def reuse_ckpt(args) -> Path | None:
    if args.variant != "onesided":
        return None
    name = f"{ARCH}_L[{LVAL}]_mapdiag_onesided_resnet_seed{args.seed}_L{LVAL}_trainT0.pth"
    path = (
        args.mapping_root
        / args.dataset
        / "mapdiag_onesided_resnet"
        / "checkpoints"
        / name
    )
    return path if path.is_file() else None


def extra_flags(spec) -> list[str]:
    flags = list(ONESIDED_FLAGS)
    flags += ["--calibrated_mne_risk_source", spec["risk_source"]]
    if spec["risk_source"] != "gm":
        return flags
    if not spec["use_p"]:
        flags.append("--gm_os_no_p")
    if spec["use_d"]:
        flags.append("--gm_os_use_d")
    if spec["use_m"]:
        flags.append("--gm_os_use_m")
    return flags


def apply_gm_cfg(model, spec):
    model._mne_layer_map = LAYER_MAP
    model._calibrated_mne_risk_source = spec["risk_source"]
    if spec["risk_source"] == "gm":
        model._gm_os_cfg = {
            "use_p": spec["use_p"],
            "use_d": spec["use_d"],
            "use_m": spec["use_m"],
            "ema_rho": 0.1,
            "sigma_mode": "act",
            "sigma_noise": 1.0,
            "include_cov": False,
            "probe_sigma": 0.1,
            "budget_norm": False,
            "quant_level": LVAL,
            "eps": 1e-6,
        }
    else:
        model._gm_os_cfg = None
    return model


def load_model(args, ckpt: Path, device, spec):
    model = modelpool(ARCH, args.dataset)
    apply_gm_cfg(model, spec)
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
    return lambda m, t, q: compute_l2_calibrated_mne_regularization(
        m,
        quant_level=LVAL,
        alpha=ALPHA,
        risk_min=RISK_MIN,
        risk_max=RISK_MAX,
        onesided=True,
        tau=TAU,
        q_assignment="risk",
        risk_source=spec["risk_source"],
    )


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
        "--regularizer", "calibrated_mne_l2",
        "--weight_decay", "0.0",
        "--reg_coeff", str(BETA),
        "--mne_layer_map", LAYER_MAP,
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
        f"[INFO] {args.dataset} {ARCH} {spec['label']} seed={args.seed} "
        f"p={spec['use_p']} D={spec['use_d']} m={spec['use_m']}",
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
    labels = labels.to(device)
    prev_t = int(getattr(model, "T", 0) or 0)
    model.set_T(0)
    model.train()
    set_basicblock_ga_cache(model, bool(spec["use_m"]))
    logits = model(images)
    ce = nn.CrossEntropyLoss()(logits, labels)
    gm_stats = {}
    if spec["risk_source"] == "gm":
        gm_stats = update_graph_margin_stats(model, ce)
    model.set_T(prev_t)
    model.eval()
    report = dump_mne_mapping_report(
        model,
        out,
        layer_map=LAYER_MAP,
        quant_level=LVAL,
        alpha=ALPHA,
        tau=TAU,
        risk_min=RISK_MIN,
        risk_max=RISK_MAX,
        q_assignment="risk",
        onesided=True,
        extra={
            "legacy_match": summarize_weight_layer_matches(
                collect_weight_layer_matches(model, "legacy")
            ),
            "resnet_match": summarize_weight_layer_matches(
                collect_weight_layer_matches(model, "resnet")
            ),
            "gm_os": gm_stats,
        },
    )
    set_basicblock_ga_cache(model, False)
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
        "config": config_name(args.variant),
        "label": spec["label"],
        "variant": args.variant,
        "dataset": args.dataset,
        "arch": ARCH,
        "seed": args.seed,
        "layer_map": LAYER_MAP,
        "risk_source": spec["risk_source"],
        "use_p": spec["use_p"],
        "use_d": spec["use_d"],
        "use_m": spec["use_m"],
        "regularizer": "calibrated_mne_l2",
        "reg_coeff": BETA,
        "checkpoint": str(ckpt),
        "matched_layers_over_total": report["matched_layers_over_total"],
        "p_gt_tau": report["p_gt_tau"],
        "q_mean": report["q_mean"],
        **gm_stats,
        **grads,
        **snn_metrics(val_rows, "val"),
        **snn_metrics(test_rows, "test"),
    }
    (out / "scorecard.json").write_text(json.dumps(card, indent=2, default=str) + "\n")
    print(json.dumps(card, indent=2, default=str), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
