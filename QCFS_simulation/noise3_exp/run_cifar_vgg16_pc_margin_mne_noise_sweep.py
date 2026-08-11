#!/usr/bin/env python3
"""
CIFAR-10/100 VGG16: PC-MNE / Margin-MNE train + pre_first_conv absolute noise sweep.

Methods:
  pc_mne      - mean Gaussian P_cross on first-layer QCFS activations
  margin_mne  - hinge on per-activation ρ=d/s (tau default 2)
  mne_l2      - classic MNE-standard baseline (optional)
  weight_decay - optimizer WD all-param baseline (optional)

Protocol:
  train T=0 ANN, L=16, warmup reg 10 epochs
  test  T=16 rate_uniform, noise at pre_first_conv, sigma 0..1 step 0.05
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARCH = "vgg16"
LVAL = 16
TVAL = 16
IF_MODE = "rate_uniform"
EPOCHS = int(os.environ.get("CIFAR_EPOCHS", "300"))
LR = 0.1
BATCH = int(os.environ.get("CIFAR_BATCH", "128"))
NUM_WORKERS = int(os.environ.get("CIFAR_NUM_WORKERS", "8"))
DEFAULT_SEEDS = [42]

METHOD_SPECS = {
    "pc_mne": {
        "label": "PC-MNE",
        "regularizer": "pc_mne",
        "reg_coeff": 1.0,
        "weight_decay": 0.0,
        "train_args": [
            "--reg_warmup_epochs",
            "10",
            "--pc_mne_sigma",
            "1.0",
            "--pc_mne_protocol",
            "snn_indep",
            "--pc_mne_eval_T",
            str(TVAL),
        ],
    },
    "margin_mne": {
        "label": "Margin-MNE",
        "regularizer": "margin_mne",
        "reg_coeff": 1.0,
        "weight_decay": 0.0,
        "train_args": [
            "--reg_warmup_epochs",
            "10",
            "--pc_mne_sigma",
            "1.0",
            "--pc_mne_protocol",
            "snn_indep",
            "--pc_mne_eval_T",
            str(TVAL),
            "--margin_mne_tau",
            "2.0",
        ],
    },
    "mne_l2": {
        "label": "MNE-standard",
        "regularizer": "mne_l2",
        "reg_coeff": 1e-4,
        "weight_decay": 0.0,
        "train_args": ["--mne_detach_lambda"],
    },
    "weight_decay": {
        "label": "L2-all-WD",
        "regularizer": "weight_decay",
        "reg_coeff": 1.0,
        "weight_decay": 5e-4,
        "train_args": [],
    },
}

ALIASES = {
    "pc_mne": "pc_mne",
    "pcmne": "pc_mne",
    "margin_mne": "margin_mne",
    "marginmne": "margin_mne",
    "mne_l2": "mne_l2",
    "mnel2": "mne_l2",
    "weight_decay": "weight_decay",
    "l2": "weight_decay",
    "wd": "weight_decay",
}

PLOT_STYLES = {
    "pc_mne": {"color": "#d62728", "label": "PC-MNE"},
    "margin_mne": {"color": "#9467bd", "label": "Margin-MNE"},
    "mne_l2": {"color": "#ff7f0e", "label": "MNE-standard"},
    "weight_decay": {"color": "#1f77b4", "label": "L2-all-WD"},
}


def coeff_tag(v: float) -> str:
    return f"{v:.0e}".replace("-", "m").replace("+", "p").replace(".", "p")


def build_suffix(method: str, seed: int, run_tag: str | None) -> str:
    spec = METHOD_SPECS[method]
    mid = f"{run_tag}_" if run_tag else ""
    if method == "weight_decay":
        return f"{mid}pcmnecmp_seed{seed}_wd_l{LVAL}_{ARCH}"
    if method == "mne_l2":
        return f"{mid}pcmnecmp_seed{seed}_mne_l2_l{LVAL}_{ARCH}_rc{coeff_tag(spec['reg_coeff'])}"
    if method == "pc_mne":
        return f"{mid}pcmnecmp_seed{seed}_pc_mne_l{LVAL}_{ARCH}_rc{coeff_tag(spec['reg_coeff'])}"
    if method == "margin_mne":
        return f"{mid}pcmnecmp_seed{seed}_margin_mne_l{LVAL}_{ARCH}_rc{coeff_tag(spec['reg_coeff'])}_tau2"
    raise ValueError(method)


def ckpt_path(dataset: str, method: str, seed: int, run_tag: str | None) -> Path:
    return ROOT / f"{dataset}-checkpoints" / f"{ARCH}_L[{LVAL}]_{build_suffix(method, seed, run_tag)}.pth"


def train_one(dataset, method, seed, run_tag, retrain, epochs, reg_coeff, margin_tau):
    spec = dict(METHOD_SPECS[method])
    if reg_coeff is not None:
        spec["reg_coeff"] = float(reg_coeff)
    ckpt = ckpt_path(dataset, method, seed, run_tag)
    if retrain and ckpt.exists():
        ckpt.unlink()
        print(f"[RETRAIN] removed {ckpt.name}", flush=True)
    if ckpt.exists():
        print(f"[SKIP TRAIN] {ckpt.name}", flush=True)
        return ckpt

    cmd = [
        sys.executable,
        str(ROOT / "main_train.py"),
        "-data",
        dataset,
        "-arch",
        ARCH,
        "-L",
        str(LVAL),
        "--epochs",
        str(epochs),
        "-lr",
        str(LR),
        "-j",
        str(NUM_WORKERS),
        "-b",
        str(BATCH),
        "--seed",
        str(seed),
        "--device",
        "auto",
        "--time",
        "0",
        "--spike_schedule",
        "normal",
        "--regularizer",
        spec["regularizer"],
        "--weight_decay",
        str(spec["weight_decay"]),
        "--reg_coeff",
        str(spec["reg_coeff"]),
        "--suffix",
        build_suffix(method, seed, run_tag),
        "--ckpt-save-mode",
        "best",
    ] + list(spec["train_args"])
    if method == "margin_mne" and margin_tau is not None:
        # override default tau in train_args
        if "--margin_mne_tau" in cmd:
            i = cmd.index("--margin_mne_tau")
            cmd[i + 1] = str(margin_tau)
        else:
            cmd += ["--margin_mne_tau", str(margin_tau)]

    print(f"[TRAIN] {dataset} {method} seed={seed}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    print(f"[TRAIN DONE] {ckpt.name}", flush=True)
    return ckpt


def test_one(dataset, method, seed, ckpt, out, force_test):
    label = METHOD_SPECS[method]["label"].replace(" ", "_")
    test_dir = out / label / f"seed_{seed}"
    test_dir.mkdir(parents=True, exist_ok=True)
    matrix = (
        test_dir
        / f"noise_sweep_matrix_{dataset}_{ARCH}_T{TVAL}_mode_{IF_MODE}_schedule_normal_seed_{seed}.csv"
    )
    if force_test and matrix.exists():
        matrix.unlink()
    if matrix.exists():
        print(f"[SKIP TEST] {method} seed={seed}", flush=True)
        return matrix

    cmd = [
        sys.executable,
        str(ROOT / "main_test.py"),
        "-data",
        dataset,
        "-arch",
        ARCH,
        "-L",
        str(LVAL),
        "-T",
        str(TVAL),
        "-j",
        str(NUM_WORKERS),
        "-b",
        str(BATCH),
        "--seed",
        str(seed),
        "--device",
        "auto",
        "--mode",
        IF_MODE,
        "--spike_schedule",
        "normal",
        "--weights",
        str(ckpt),
        "--noise_sweep",
        "--noise_sigma_start",
        "0.0",
        "--noise_sigma_end",
        "1.0",
        "--noise_sigma_step",
        "0.05",
        "--first_layer_noise_position",
        "pre_first_conv",
        "--noise_output_dir",
        str(test_dir),
    ]
    print(f"[TEST] {method} seed={seed} pos=pre_first_conv", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    if not matrix.exists():
        cands = sorted(test_dir.glob("noise_sweep_matrix_*.csv"))
        if not cands:
            raise FileNotFoundError(matrix)
        matrix = cands[0]
    print(f"[TEST DONE] {matrix.name}", flush=True)
    return matrix


def read_matrix_acc(matrix: Path) -> list[tuple[float, float]]:
    rows = list(csv.DictReader(matrix.open()))
    if not rows:
        return []
    # main_test writes wide CSV: columns = L[,T], sigma0, sigma1, ...
    row = rows[0]
    out = []
    for k, v in row.items():
        if k in ("L", "T"):
            continue
        try:
            out.append((float(k), float(v)))
        except (TypeError, ValueError):
            continue
    out.sort(key=lambda x: x[0])
    return out


def plot_and_save(dataset, agg_rows, out: Path):
    if not agg_rows:
        return
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    methods = []
    for r in agg_rows:
        if r["method"] not in methods:
            methods.append(r["method"])
    for method in methods:
        rr = [r for r in agg_rows if r["method"] == method]
        rr.sort(key=lambda x: float(x["sigma"]))
        xs = [float(r["sigma"]) for r in rr]
        ys = [float(r["acc_mean"]) for r in rr]
        st = PLOT_STYLES.get(method, {"color": "#333333", "label": method})
        ax.plot(xs, ys, marker="o", lw=2.0, ms=4.5, color=st["color"], label=st["label"])
    ax.set_xlabel(r"Absolute noise $\sigma$ (pre-first-conv)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{dataset.upper()} VGG16 · PC/Margin-MNE · pre_first_conv")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    png = out / f"{dataset}_vgg16_pc_margin_mne_pre_first_conv_noise_sweep.png"
    fig.savefig(png, dpi=180)
    plt.close(fig)
    print(f"[PLOT] {png}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    p.add_argument(
        "--methods",
        nargs="+",
        default=["pc_mne", "margin_mne", "mne_l2", "weight_decay"],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--reg-coeff", type=float, default=None, help="Override PC/Margin-MNE beta")
    p.add_argument("--margin-tau", type=float, default=2.0)
    p.add_argument("--run-tag", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--force-test", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    methods = [ALIASES.get(m.lower(), m) for m in args.methods]
    for m in methods:
        if m not in METHOD_SPECS:
            raise ValueError(f"Unknown method {m}; choices={list(METHOD_SPECS)}")

    out = Path(args.out_dir) if args.out_dir else (
        ROOT.parent
        / "important_results"
        / f"{args.dataset}_vgg16_pc_margin_mne_pre_first_conv_seed{'_'.join(map(str, args.seeds))}"
    )
    out.mkdir(parents=True, exist_ok=True)
    raw_csv = out / f"{args.dataset}_vgg16_pc_margin_mne_noise_sweep_raw.csv"
    mean_csv = out / f"{args.dataset}_vgg16_pc_margin_mne_noise_sweep_mean_std.csv"

    raw_rows = []
    if not args.plot_only:
        for method in methods:
            for seed in args.seeds:
                ckpt = train_one(
                    args.dataset,
                    method,
                    seed,
                    args.run_tag,
                    args.retrain,
                    args.epochs,
                    args.reg_coeff,
                    args.margin_tau,
                )
                matrix = test_one(
                    args.dataset, method, seed, ckpt, out, args.force_test
                )
                for sigma, acc in read_matrix_acc(matrix):
                    raw_rows.append(
                        {
                            "dataset": args.dataset,
                            "method": method,
                            "label": METHOD_SPECS[method]["label"],
                            "seed": seed,
                            "sigma": f"{sigma:.2f}",
                            "acc": f"{acc:.6f}",
                            "checkpoint": str(ckpt),
                            "matrix_csv": str(matrix),
                        }
                    )
        with raw_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
            w.writeheader()
            w.writerows(raw_rows)
        print(f"[TABLE] raw: {raw_csv}", flush=True)
    else:
        raw_rows = list(csv.DictReader(raw_csv.open()))

    bucket = defaultdict(list)
    for r in raw_rows:
        bucket[(r["method"], float(r["sigma"]))].append(float(r["acc"]))
    agg = []
    for (method, sigma), vals in sorted(bucket.items(), key=lambda x: (x[0][0], x[0][1])):
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        agg.append(
            {
                "method": method,
                "label": METHOD_SPECS[method]["label"],
                "sigma": f"{sigma:.2f}",
                "acc_mean": f"{mean:.6f}",
                "acc_std": f"{std:.6f}",
                "n_seeds": len(vals),
            }
        )
    with mean_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)
    print(f"[TABLE] mean: {mean_csv}", flush=True)
    plot_and_save(args.dataset, agg, out)


if __name__ == "__main__":
    main()
