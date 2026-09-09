#!/usr/bin/env python3
"""ResNet-18/CIFAR Task-Cov-MNE screen with a fixed two-arm intervention.

The existing L2-wo and historical MNE-L2-detach checkpoints are deliberately
not retrained here.  This runner adds the two missing controls under the same
weights-only L2 budget (beta=5e-4):

  cov_mne       diagonal propagated input-noise second moment only (omega=1)
  task_cov_mne  the same covariance multiplied by detached task-margin weight

Both variants train a QCFS ANN at T=0 and calibrate their coefficients with
paired clean/noisy T=16 rate_uniform SNN forwards.  The covariance estimate
is channel-diagonal and spatially stationary, rather than a materialized full
convolution-patch covariance.  This is a first ResNet screen, not a claim
that full DAG-MNE has been implemented.
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
    BETA,
    EPOCHS,
    LR,
    LVAL,
    TEST_T,
    snn_metrics,
    sweep,
    test_loader,
    val_loader,
    write_csv,
)
from utils import get_torch_device  # noqa: E402

ARCH = "resnet18"
SEED = 42
CALIBRATION_SIGMA = 1.0
CALIBRATION_INTERVAL = 100
TASK_INTERVAL = 10
CALIBRATION_BATCH_SIZE = 4
EMA_RHO = 0.1
Q_FLOOR = 0.25
ALLOCATION_START = 30
ALLOCATION_WARMUP = 50

METHODS = {
    "cov_mne": {
        "label": "Cov-MNE (diagonal, $\\omega=1$)",
        "mode": "cov",
    },
    "task_cov_mne": {
        "label": "Task-Cov-MNE (diagonal)",
        "mode": "task",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar_resnet18_task_cov_mne_seed42",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    return args


def config_name(method: str) -> str:
    return f"r18_{method}_diag_t{TEST_T}_sig{CALIBRATION_SIGMA:g}"


def suffix(args) -> str:
    return f"{config_name(args.method)}_seed{args.seed}_L{LVAL}_trainT0"


def cfg_dir(args) -> Path:
    return args.out_root / args.dataset / config_name(args.method) / f"seed{args.seed}"


def ckpt_path(args) -> Path:
    return cfg_dir(args) / "checkpoints" / f"{ARCH}_L[{LVAL}]_{suffix(args)}.pth"


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
    model.set_first_layer_input_noise_position("post_input_if")
    model.set_first_layer_input_noise_type("gaussian")
    return model.to(device).eval()


def train(args) -> Path:
    spec = METHODS[args.method]
    out = cfg_dir(args)
    checkpoint = ckpt_path(args)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {checkpoint}", flush=True)
        return checkpoint
    if args.test_only:
        raise FileNotFoundError(checkpoint)

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
        "--ckpt-dir", str(checkpoint.parent),
        "-suffix", suffix(args),
        "--regularizer", "task_cov_mne",
        "--weight_decay", "0",
        "--reg_coeff", str(BETA),
        "--mne_layer_map", "resnet",
        "--task_cov_mode", spec["mode"],
        "--task_cov_deploy_T", str(TEST_T),
        "--task_cov_sigma", str(CALIBRATION_SIGMA),
        "--task_cov_calib_interval", str(CALIBRATION_INTERVAL),
        "--task_cov_task_interval", str(TASK_INTERVAL),
        "--task_cov_calib_batch_size", str(CALIBRATION_BATCH_SIZE),
        "--task_cov_ema_rho", str(EMA_RHO),
        "--task_cov_q_floor", str(Q_FLOOR),
        "--task_cov_start_epoch", str(ALLOCATION_START),
        "--task_cov_warmup_epochs", str(ALLOCATION_WARMUP),
        "--epoch_log_csv", str(out / "epoch_log.csv"),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not checkpoint.exists():
        raise FileNotFoundError(f"training finished but checkpoint is missing: {checkpoint}")
    return checkpoint


def scorecard(val_rows, test_rows, args, checkpoint: Path) -> dict:
    spec = METHODS[args.method]
    card = {
        "config": config_name(args.method),
        "label": spec["label"],
        "method": args.method,
        "dataset": args.dataset,
        "arch": ARCH,
        "seed": args.seed,
        "regularizer": "task_cov_mne",
        "task_cov_mode": spec["mode"],
        "layer_map": "resnet",
        "reg_coeff": BETA,
        "weight_decay": 0.0,
        "deploy_T": TEST_T,
        "calibration_sigma": CALIBRATION_SIGMA,
        "calibration_interval": CALIBRATION_INTERVAL,
        "task_interval": TASK_INTERVAL,
        "calibration_batch_size": CALIBRATION_BATCH_SIZE,
        "ema_rho": EMA_RHO,
        "q_floor": Q_FLOOR,
        "allocation_start_epoch": ALLOCATION_START,
        "allocation_warmup_epochs": ALLOCATION_WARMUP,
        "covariance_approximation": "channel-diagonal, spatially stationary",
        "checkpoint": str(checkpoint),
    }
    card.update(snn_metrics(val_rows, "val"))
    card.update(snn_metrics(test_rows, "test"))
    return card


def main() -> None:
    args = parse_args()
    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] {args.dataset} ResNet-18 {METHODS[args.method]['label']} "
        f"seed={args.seed}; Ttrain=0, Teval={TEST_T}, rate_uniform, post_input_if",
        flush=True,
    )
    checkpoint = train(args)
    device = get_torch_device(args.device)
    pin_memory = device.type == "cuda"
    model = load_model(checkpoint, device, args.dataset)
    val_rows = sweep(model, val_loader(args, pin_memory), device, "val", args.seed)
    write_csv(out / "val_sweep.csv", val_rows)
    test_rows = sweep(model, test_loader(args, pin_memory), device, "test", args.seed)
    write_csv(out / "test_sweep.csv", test_rows)
    card = scorecard(val_rows, test_rows, args, checkpoint)
    (out / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2), flush=True)


if __name__ == "__main__":
    main()
