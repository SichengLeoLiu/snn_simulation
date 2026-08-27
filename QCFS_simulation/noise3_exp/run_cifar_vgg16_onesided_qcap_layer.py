#!/usr/bin/env python3
"""Frozen onesided MNE: q_max cap and layer-mean assignment.

Frozen recipe: α=4, τ=0.5, β=5e-4, risk clip [0.5, 8], warmup 30/50.
Do not retune from test curves.

    q_OS = min(q_max, 1 + α [r̂ − τ]_+)     (q_max=0 means uncapped)
    R    = (β/2) Σ q ||W||^2

Variants
--------
    cap16    channel-wise risk, q_max=16
    cap8     channel-wise risk, q_max=8
    cap4     channel-wise risk, q_max=4
    layer    layer-mean of uncapped q_OS
    layer8   layer-mean of q_OS after q_max=8

Uncapped channel-wise One-sided is the previous qabl_risk run; do not retrain it.
ANN train T=0; SNN eval T=16, rate_uniform, post_input_if.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_cifar_vgg16_onesided_q_assignment_ablation import (  # noqa: E402
    ALPHA,
    ALPHA_START,
    ALPHA_WARMUP,
    ARCH,
    BETA,
    EPOCHS,
    LVAL,
    LR,
    RISK_MAX,
    RISK_MIN,
    TAU,
    load_model,
    snn_metrics,
    sweep,
    test_loader,
    val_loader,
    write_csv,
)
from utils import get_torch_device  # noqa: E402


VARIANTS = {
    "cap16": {"label": "Capped-16", "assignment": "risk", "q_max": 16.0},
    "cap8": {"label": "Capped-8", "assignment": "risk", "q_max": 8.0},
    "cap4": {"label": "Capped-4", "assignment": "risk", "q_max": 4.0},
    "layer": {"label": "Layer-mean", "assignment": "layer_mean", "q_max": 0.0},
    "layer8": {"label": "Layer-mean+qmax8", "assignment": "layer_mean", "q_max": 8.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default=None)
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
        default=ROOT.parent
        / "important_results"
        / "cifar_vgg16_onesided_qcap_layer_seed42",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    if not args.summarize and args.variant is None:
        parser.error("specify --variant or --summarize")
    if args.variant is not None:
        spec = VARIANTS[args.variant]
        args.assignment = spec["assignment"]
        args.q_max = spec["q_max"]
        args.label = spec["label"]
    return args


def qmax_tag(q_max: float) -> str:
    if float(q_max) <= 0:
        return "qmaxnone"
    return f"qmax{float(q_max):g}".replace(".", "p")


def config_name(args) -> str:
    return f"qcap_{args.variant}_{qmax_tag(args.q_max)}_a4_tau0.5_b0.0005_rmax8"


def suffix(args) -> str:
    return f"{config_name(args)}_seed{args.seed}_L{LVAL}_trainT0"


def ckpt_filename(args) -> str:
    return f"{ARCH}_L[{LVAL}]_{suffix(args)}.pth"


def cfg_dir(args) -> Path:
    return args.out_root / args.dataset / config_name(args)


def ckpt_path(args) -> Path:
    return cfg_dir(args) / "checkpoints" / ckpt_filename(args)


def train(args) -> Path:
    out = cfg_dir(args)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_path(args)
    epoch_log = out / "epoch_log.csv"
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
        "--regularizer", "calibrated_mne_l2",
        "--weight_decay", "0",
        "--reg_coeff", str(BETA),
        "--calibrated_mne_alpha", str(ALPHA),
        "--calibrated_mne_onesided",
        "--calibrated_mne_tau", str(TAU),
        "--calibrated_mne_risk_min", str(RISK_MIN),
        "--calibrated_mne_risk_max", str(RISK_MAX),
        "--calibrated_mne_alpha_start_epoch", str(ALPHA_START),
        "--calibrated_mne_alpha_warmup_epochs", str(ALPHA_WARMUP),
        "--calibrated_mne_q_assignment", args.assignment,
        "--calibrated_mne_q_max", str(args.q_max),
        "--epoch_log_csv", str(epoch_log),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"training finished but missing {ckpt}")
    return ckpt


def scorecard(val_rows, test_rows, args, ckpt: Path) -> dict:
    card = {
        "config": config_name(args),
        "label": args.label,
        "variant": args.variant,
        "assignment": args.assignment,
        "q_max": args.q_max,
        "dataset": args.dataset,
        "seed": args.seed,
        "alpha": ALPHA,
        "tau": TAU,
        "beta": BETA,
        "risk_max": RISK_MAX,
        "checkpoint": str(ckpt),
        "epoch_log": str(cfg_dir(args) / "epoch_log.csv"),
    }
    card.update(snn_metrics(val_rows, "val"))
    card.update(snn_metrics(test_rows, "test"))
    return card


def summarize(out_root: Path) -> None:
    cards = []
    for path in sorted(out_root.glob("*/*/scorecard.json")):
        cards.append(json.loads(path.read_text()))
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    print(
        f"{'dataset':<10} {'var':<8} {'label':<18} "
        f"{'val0':>7} {'val5':>7} {'valAUC':>8} {'valHi':>8} "
        f"{'test0':>7} {'test5':>7} {'testAUC':>8} {'testHi':>8}"
    )
    for card in cards:
        print(
            f"{card['dataset']:<10} {card['variant']:<8} {card['label']:<18} "
            f"{card['val_clean']:7.2f} {card['val_sigma5']:7.2f} "
            f"{card['val_auc_full']:8.2f} {card['val_auc_high']:8.2f} "
            f"{card['test_clean']:7.2f} {card['test_sigma5']:7.2f} "
            f"{card['test_auc_full']:8.2f} {card['test_auc_high']:8.2f}"
        )
    write_csv(out_root / "qcap_layer_ranking.csv", cards)
    print(f"Wrote {out_root / 'qcap_layer_ranking.csv'}")


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(args.out_root)
        return

    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] {args.dataset} {args.label} ({args.variant}) "
        f"assign={args.assignment} q_max={args.q_max} seed={args.seed}  "
        f"frozen α={ALPHA} τ={TAU} rmax={RISK_MAX} β={BETA}",
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
