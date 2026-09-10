#!/usr/bin/env python3
"""ResNet-18/CIFAR Task-Cov-MNE screen with a fixed two-arm intervention.

The two-arm screen is now 5-seed (40–44) on CIFAR-10/100 with a shared
test noise stream (eval seed 0). Extra evaluation noise seeds are written
under eval_seed{N}/ and must not replace the canonical scorecard.
Historical MNE-L2-detach checkpoints are still not controls; the fair
ResNet-map / val-tuned detach baseline is the separate P0 runner.

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
SEEDS = (40, 41, 42, 43, 44)
EVAL_NOISE_SEED = 0
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


def parse_seed_list(raw: str) -> list[int]:
    seeds = []
    for part in (raw or "").replace(":", ",").replace(" ", ",").split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHODS), default=None)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval-seed", type=int, default=int(os.environ.get("EVAL_SEED", str(EVAL_NOISE_SEED))))
    parser.add_argument(
        "--extra-eval-seeds",
        default=os.environ.get("EXTRA_EVAL_SEEDS", ""),
        help="Comma-separated extra noise seeds written under eval_seed{N}/, not the canonical scorecard",
    )
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
        default=ROOT.parent / "important_results" / "cifar_resnet18_task_cov_mne_seed42",
    )
    args = parser.parse_args()
    args.eval_seed = int(args.eval_seed)
    args.extra_eval_seeds = parse_seed_list(args.extra_eval_seeds)
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    if args.summarize:
        return args
    if args.method is None:
        parser.error("--method is required unless --summarize")
    return args


def config_name(method: str) -> str:
    return f"r18_{method}_diag_t{TEST_T}_sig{CALIBRATION_SIGMA:g}"


def suffix(args) -> str:
    return f"{config_name(args.method)}_seed{args.seed}_L{LVAL}_trainT0"


def cfg_dir(args) -> Path:
    return args.out_root / args.dataset / config_name(args.method) / f"seed{args.seed}"


def eval_dest(out: Path, eval_seed: int, canonical_seed: int) -> Path:
    if eval_seed == canonical_seed:
        return out
    return out / f"eval_seed{eval_seed}"


def write_eval(model, args, checkpoint: Path, device, eval_seed: int, dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    pin_memory = device.type == "cuda"
    val_rows = sweep(model, val_loader(args, pin_memory), device, "val", eval_seed)
    write_csv(dest / "val_sweep.csv", val_rows)
    test_rows = sweep(model, test_loader(args, pin_memory), device, "test", eval_seed)
    write_csv(dest / "test_sweep.csv", test_rows)
    saved = args.eval_seed
    args.eval_seed = eval_seed
    card = scorecard(val_rows, test_rows, args, checkpoint)
    args.eval_seed = saved
    (dest / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2), flush=True)
    return card


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
        "eval_seed": args.eval_seed,
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


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, var ** 0.5


def _acc_at_csv(path: Path, sigma: float) -> float | None:
    if not path.is_file():
        return None
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if abs(float(row["sigma"]) - sigma) < 1e-12:
                return float(row["accuracy"])
    return None


def summarize(out_root: Path) -> None:
    print(f"{'dataset':<10} {'method':<14} {'eval':>4} {'n':>3} {'test0':>16} {'test3':>16} {'test5':>16} {'AUC3-5':>16}")
    rows = []
    for dataset in ("cifar10", "cifar100"):
        for method in METHODS:
            grouped: dict[int, list[dict]] = {}
            sweeps: dict[int, list[float]] = {}
            for seed in SEEDS:
                seed_dir = out_root / dataset / config_name(method) / f"seed{seed}"
                canonical = seed_dir / "scorecard.json"
                extras = sorted(seed_dir.glob("eval_seed*/scorecard.json"))
                paths = ([canonical] if canonical.is_file() else []) + extras
                for path in paths:
                    card = json.loads(path.read_text())
                    eval_seed = int(card["eval_seed"])
                    grouped.setdefault(eval_seed, []).append(card)
                    sigma3 = _acc_at_csv(path.parent / "test_sweep.csv", 3.0)
                    if sigma3 is not None:
                        sweeps.setdefault(eval_seed, []).append(sigma3)
            if not grouped:
                print(f"{dataset:<10} {method:<14} MISSING")
                continue
            for eval_seed in sorted(grouped):
                group = grouped[eval_seed]
                t0m, t0s = _mean_std([float(card["test_clean"]) for card in group])
                t5m, t5s = _mean_std([float(card["test_sigma5"]) for card in group])
                hm, hs = _mean_std([float(card["test_auc_high"]) for card in group])
                s3 = sweeps.get(eval_seed, [])
                t3m, t3s = _mean_std(s3) if s3 else (float("nan"), float("nan"))
                print(
                    f"{dataset:<10} {method:<14} {eval_seed:4d} {len(group):3d} "
                    f"{t0m:7.2f}±{t0s:<6.2f} {t3m:7.2f}±{t3s:<6.2f} "
                    f"{t5m:7.2f}±{t5s:<6.2f} {hm:7.1f}±{hs:<6.1f}"
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "eval_seed": eval_seed,
                        "n_seeds": len(group),
                        "test_clean_mean": t0m,
                        "test_clean_std": t0s,
                        "test_sigma3_mean": t3m,
                        "test_sigma3_std": t3s,
                        "test_sigma5_mean": t5m,
                        "test_sigma5_std": t5s,
                        "test_auc_high_mean": hm,
                        "test_auc_high_std": hs,
                    }
                )
    if rows:
        write_csv(out_root / "task_cov_5seed_summary.csv", rows)
        print(f"Wrote {out_root / 'task_cov_5seed_summary.csv'}")


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(args.out_root)
        return
    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] {args.dataset} ResNet-18 {METHODS[args.method]['label']} "
        f"seed={args.seed} eval_seed={args.eval_seed} extra_eval_seeds={args.extra_eval_seeds}; "
        f"Ttrain=0, Teval={TEST_T}, rate_uniform, post_input_if",
        flush=True,
    )
    checkpoint = train(args)
    device = get_torch_device(args.device)
    model = load_model(checkpoint, device, args.dataset)
    canonical = out / "scorecard.json"
    if canonical.is_file() and not args.retrain:
        print(f"[SKIP CANONICAL EVAL] {canonical}", flush=True)
    else:
        write_eval(model, args, checkpoint, device, args.eval_seed, eval_dest(out, args.eval_seed, args.eval_seed))
    for extra in args.extra_eval_seeds:
        if extra == args.eval_seed:
            continue
        dest = eval_dest(out, extra, args.eval_seed)
        extra_card = dest / "scorecard.json"
        if extra_card.is_file() and not args.retrain:
            print(f"[SKIP EXTRA EVAL] {extra_card}", flush=True)
            continue
        print(f"[EXTRA EVAL] eval_seed={extra} -> {dest}", flush=True)
        write_eval(model, args, checkpoint, device, extra, dest)


if __name__ == "__main__":
    main()
