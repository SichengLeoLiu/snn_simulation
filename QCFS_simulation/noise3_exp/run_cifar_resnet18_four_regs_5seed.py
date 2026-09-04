#!/usr/bin/env python3
"""CIFAR ResNet-18 5-seed: L2-all, L2-wo, MNE-L2 detach, no-detach, One-sided.

Protocol matches the VGG envelope/onesided runs:
  ANN train T=0, 300 epochs, lr=0.1, L=16
  SNN eval T=16, rate_uniform, post_input_if, σ=0…5
  Seeds 40–44. Selection/logging uses 5k train-holdout val; test is recorded only.

Coefficients (CIFAR VGG five-regs / frozen onesided):
  L2-all / L2-wo : optimizer WD = 5e-4
  MNE-L2 detach  : mne_l2, detach λ, rc = 1e-4
  no-detach MNE  : same formula, grads into λ and BN γ, rc = 1e-4
  One-sided      : α=4, τ=0.5, r_max=8, β=5e-4, warmup 30/50
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

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
from utils import get_torch_device  # noqa: E402

ARCH = "resnet18"
SEEDS = (40, 41, 42, 43, 44)
L2_WD = 5e-4
MNE_RC = 1e-4

METHODS = {
    "l2all": {
        "label": "L2-all",
        "regularizer": "weight_decay",
        "weight_decay": L2_WD,
        "reg_coeff": None,
        "extra": [],
    },
    "l2wo": {
        "label": "L2-wo",
        "regularizer": "weight_decay_weights_only",
        "weight_decay": L2_WD,
        "reg_coeff": None,
        "extra": [],
    },
    "mne": {
        "label": "MNE-L2 detach",
        "regularizer": "mne_l2",
        "weight_decay": 0.0,
        "reg_coeff": MNE_RC,
        "extra": ["--mne_detach_lambda"],
    },
    "nodetach": {
        "label": "MNE-L2 no-detach",
        "regularizer": "mne_l2",
        "weight_decay": 0.0,
        "reg_coeff": MNE_RC,
        "extra": ["--mne_no_detach_bn_affine"],
    },
    "onesided": {
        "label": "One-sided MNE",
        "regularizer": "calibrated_mne_l2",
        "weight_decay": 0.0,
        "reg_coeff": BETA,
        "extra": [
            "--calibrated_mne_alpha", str(ALPHA),
            "--calibrated_mne_onesided",
            "--calibrated_mne_tau", str(TAU),
            "--calibrated_mne_risk_min", str(RISK_MIN),
            "--calibrated_mne_risk_max", str(RISK_MAX),
            "--calibrated_mne_alpha_start_epoch", str(ALPHA_START),
            "--calibrated_mne_alpha_warmup_epochs", str(ALPHA_WARMUP),
            "--calibrated_mne_q_assignment", "risk",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHODS), default=None)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar_resnet18_four_regs_5seed",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    if not args.summarize and args.method is None:
        parser.error("specify --method or --summarize")
    return args


def config_name(method: str) -> str:
    return f"r18_{method}"


def suffix(args) -> str:
    return f"{config_name(args.method)}_seed{args.seed}_L{LVAL}_trainT0"


def ckpt_filename(args) -> str:
    return f"{ARCH}_L[{LVAL}]_{suffix(args)}.pth"


def cfg_dir(args) -> Path:
    return args.out_root / args.dataset / config_name(args.method) / f"seed{args.seed}"


def ckpt_path(args) -> Path:
    return cfg_dir(args) / "checkpoints" / ckpt_filename(args)


def load_model(ckpt: Path, device, dataset: str):
    model = modelpool(ARCH, dataset)
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


def train(args) -> Path:
    spec = METHODS[args.method]
    out = cfg_dir(args)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_path(args)
    if ckpt.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {ckpt}", flush=True)
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
    ]
    if spec["reg_coeff"] is not None:
        cmd += ["--reg_coeff", str(spec["reg_coeff"])]
    cmd += list(spec["extra"])
    if args.method in ("onesided", "nodetach"):
        cmd += ["--epoch_log_csv", str(out / "epoch_log.csv")]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"training finished but missing {ckpt}")
    return ckpt


def scorecard(val_rows, test_rows, args, ckpt: Path) -> dict:
    spec = METHODS[args.method]
    card = {
        "config": config_name(args.method),
        "label": spec["label"],
        "method": args.method,
        "dataset": args.dataset,
        "arch": ARCH,
        "seed": args.seed,
        "regularizer": spec["regularizer"],
        "weight_decay": spec["weight_decay"],
        "reg_coeff": spec["reg_coeff"],
        "checkpoint": str(ckpt),
    }
    card.update(snn_metrics(val_rows, "val"))
    card.update(snn_metrics(test_rows, "test"))
    return card


def _mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, var ** 0.5


def summarize(out_root: Path) -> None:
    cards = []
    for path in sorted(out_root.glob("*/*/seed*/scorecard.json")):
        cards.append(json.loads(path.read_text()))
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    grouped = {}
    for card in cards:
        key = (card["dataset"], card["method"])
        grouped.setdefault(key, []).append(card)
    print(
        f"{'dataset':<10} {'method':<10} {'n':>3} "
        f"{'test0':>16} {'test5':>16} {'AUC':>16} {'AUC3-5':>16}"
    )
    rows = []
    for ds in ("cifar10", "cifar100"):
        for method in METHODS:
            group = grouped.get((ds, method), [])
            if not group:
                print(f"{ds:<10} {method:<10} MISSING")
                continue
            def col(name):
                return _mean_std([float(c[name]) for c in group])
            t0m, t0s = col("test_clean")
            t5m, t5s = col("test_sigma5")
            am, as_ = col("test_auc_full")
            hm, hs = col("test_auc_high")
            print(
                f"{ds:<10} {method:<10} {len(group):3d} "
                f"{t0m:7.2f}±{t0s:<6.2f} {t5m:7.2f}±{t5s:<6.2f} "
                f"{am:7.1f}±{as_:<6.1f} {hm:7.1f}±{hs:<6.1f}"
            )
            rows.append(
                {
                    "dataset": ds,
                    "method": method,
                    "label": METHODS[method]["label"],
                    "n_seeds": len(group),
                    "test_clean_mean": t0m,
                    "test_clean_std": t0s,
                    "test_sigma5_mean": t5m,
                    "test_sigma5_std": t5s,
                    "test_auc_full_mean": am,
                    "test_auc_full_std": as_,
                    "test_auc_high_mean": hm,
                    "test_auc_high_std": hs,
                }
            )
    if rows:
        write_csv(out_root / "four_regs_5seed_summary.csv", rows)
        print(f"Wrote {out_root / 'four_regs_5seed_summary.csv'}")


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(args.out_root)
        return

    spec = METHODS[args.method]
    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] {args.dataset} ResNet-18 {spec['label']} seed={args.seed}  "
        f"Ttrain=0 Teval={TEST_T} post_input_if",
        flush=True,
    )
    ckpt = train(args)
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = load_model(ckpt, device, args.dataset)
    val_rows = sweep(model, val_loader(args, pin), device, "val", args.seed)
    write_csv(out / "val_sweep.csv", val_rows)
    test_rows = sweep(model, test_loader(args, pin), device, "test", args.seed)
    write_csv(out / "test_sweep.csv", test_rows)
    card = scorecard(val_rows, test_rows, args, ckpt)
    (out / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
