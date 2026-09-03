#!/usr/bin/env python3
"""CIFAR VGG-16 5-seed MNE-L2 no-detach vs frozen 5-seed baselines.

Trains only nodetach. Seed 42 reuses the component-ablation checkpoint.
Do not retune rc. Compare against already-run 5-seed:
  One-sided (α=4, τ=0.5, β=5e-4, r_max=8)
  MNE-L2 detach (old_detach, rc=1e-4)
  L2-wo / L2-all (optimizer WD=5e-4)

Protocol: ANN train T=0, SNN eval T=16, rate_uniform, post_input_if.
Seeds 40–44. Selection uses 5k train-holdout val; test is recorded only.

Nodetach flags match suite=component --variant nodetach --intensity fixed:
  --mne_layer_map legacy --mne_no_detach_bn_affine   (no --mne_detach_lambda)
  rc=1e-4, BN fold on, grads into λ and BN γ.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_cifar_vgg16_onesided_q_assignment_ablation import (  # noqa: E402
    EPOCHS,
    LR,
    LVAL,
    TEST_T,
    load_model,
    snn_metrics,
    sweep,
    test_loader,
    val_loader,
    write_csv,
)
from utils import get_torch_device  # noqa: E402

ARCH = "vgg16"
SEEDS = (40, 41, 42, 43, 44)
MNE_RC = 1e-4
REUSE_DEFAULT = Path(
    "/scratch/gs14/sl9144/snn_results/cifar_vgg16_mne_component_ablation_seed42"
)
REPO_RESULTS = ROOT.parent / "important_results"

PLOT_METHODS = (
    ("l2all", "L2-all"),
    ("l2wo", "L2-wo"),
    ("mne", r"MNE-L2 $L^2 M_{\mathrm{eff}}/\lambda^2$, detach"),
    ("nodetach", r"MNE-L2, no detach"),
    ("onesided", r"One-sided $\alpha=4,\tau=0.5,r_{\max}=8$"),
)
PLOT_COLORS = {
    "l2all": "#6F3FA0",
    "l2wo": "#009E73",
    "mne": "#D62728",
    "nodetach": "#8c564b",
    "onesided": "#2ca02c",
}
PLOT_LS = {
    "l2all": "--",
    "l2wo": "-.",
    "mne": "-",
    "nodetach": "-",
    "onesided": "-",
}
FIVE_VARIANT = {
    "l2all": "weight_decay",
    "l2wo": "weight_decay_weights_only",
    "mne": "old_detach",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar_vgg16_nodetach_5seed",
    )
    parser.add_argument(
        "--reuse-root",
        type=Path,
        default=Path(os.environ.get("REUSE_ROOT", str(REUSE_DEFAULT))),
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=ROOT.parent / "plots" / "cifar10_cifar100_vgg16_nodetach_5seed_vs_baselines_test.png",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    return args


def suffix(args) -> str:
    return f"v16_nodetach_seed{args.seed}_L{LVAL}_trainT0"


def ckpt_filename(args) -> str:
    return f"{ARCH}_L[{LVAL}]_{suffix(args)}.pth"


def cfg_dir(args) -> Path:
    return args.out_root / args.dataset / "v16_nodetach" / f"seed{args.seed}"


def ckpt_path(args) -> Path:
    return cfg_dir(args) / "checkpoints" / ckpt_filename(args)


def reuse_ckpt(args) -> Path | None:
    if args.retrain or args.seed != 42:
        return None
    name = f"{ARCH}_L[{LVAL}]_comp_nodetach_fixed_seed42_L{LVAL}_trainT0.pth"
    roots = [
        args.reuse_root,
        REPO_RESULTS / "cifar_vgg16_mne_component_ablation_seed42",
    ]
    for root in roots:
        path = root / args.dataset / "comp_nodetach_fixed" / "checkpoints" / name
        if path.is_file():
            return path
    return None


def train_cmd(args) -> list[str]:
    out = cfg_dir(args)
    return [
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
        "--ckpt-dir", str(out / "checkpoints"),
        "-suffix", suffix(args),
        "--regularizer", "mne_l2",
        "--weight_decay", "0",
        "--reg_coeff", str(MNE_RC),
        "--mne_layer_map", "legacy",
        "--mne_no_detach_bn_affine",
        "--epoch_log_csv", str(out / "epoch_log.csv"),
        "--mapping_diag_dir", str(out / "mapping_init"),
    ]


def train(args) -> Path:
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
    cmd = train_cmd(args)
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return ckpt
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"training finished but missing {ckpt}")
    return ckpt


def _mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0
    return mean, statistics.stdev(xs)


def summarize(out_root: Path) -> None:
    cards = []
    for path in sorted(out_root.glob("*/v16_nodetach/seed*/scorecard.json")):
        cards.append(json.loads(path.read_text()))
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    print(f"{'dataset':<10} {'n':>3} {'test0':>16} {'test5':>16}")
    rows = []
    for ds in ("cifar10", "cifar100"):
        group = [c for c in cards if c["dataset"] == ds]
        if not group:
            print(f"{ds:<10} MISSING")
            continue
        t0m, t0s = _mean_std([float(c["test_clean"]) for c in group])
        t5m, t5s = _mean_std([float(c["test_sigma5"]) for c in group])
        print(f"{ds:<10} {len(group):3d} {t0m:7.2f}±{t0s:<6.2f} {t5m:7.2f}±{t5s:<6.2f}")
        rows.append(
            {
                "dataset": ds,
                "method": "nodetach",
                "n_seeds": len(group),
                "test_clean_mean": t0m,
                "test_clean_std": t0s,
                "test_sigma5_mean": t5m,
                "test_sigma5_std": t5s,
            }
        )
    if rows:
        write_csv(out_root / "nodetach_5seed_summary.csv", rows)
        print(f"Wrote {out_root / 'nodetach_5seed_summary.csv'}")


def _read_sweep(path: Path) -> list[tuple[float, float]]:
    pairs = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split", "test") != "test":
                continue
            pairs.append((float(row["sigma"]), float(row["accuracy"])))
    pairs.sort()
    return pairs


def _agg_sweeps(paths: list[Path]) -> tuple[list[float], list[float], list[float], int]:
    buckets: dict[float, list[float]] = defaultdict(list)
    for path in paths:
        for sigma, acc in _read_sweep(path):
            buckets[round(sigma, 6)].append(acc)
    sigmas = sorted(buckets)
    means, stds = [], []
    for sigma in sigmas:
        vals = buckets[sigma]
        means.append(statistics.mean(vals))
        stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
    n = max((len(v) for v in buckets.values()), default=0)
    return sigmas, means, stds, n


def _five_regs_series(dataset: str, variant: str) -> tuple[list[float], list[float], list[float], int] | None:
    path = REPO_RESULTS / f"{dataset}_vgg16_five_regs_sigma0_5_5seed" / "mne_stability_ablation_mean_std.csv"
    if not path.is_file():
        return None
    xs, ys, ss = [], [], []
    n = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] != variant:
                continue
            xs.append(float(row["sigma"]))
            ys.append(float(row["acc_mean"]))
            ss.append(float(row["acc_std"]))
            n = max(n, int(float(row["n"])))
    if not xs:
        return None
    return xs, ys, ss, n


def _onesided_paths(dataset: str) -> list[Path]:
    found = []
    for seed in SEEDS:
        candidates = [
            REPO_RESULTS
            / f"{dataset}_vgg16_hybrid_envelope_screen_seed{seed}"
            / "onesided_a4_tau0.5_b0.0005_rmax8"
            / "test_sweep.csv",
            REPO_RESULTS
            / "onesided_green_scratch"
            / f"{dataset}_vgg16_hybrid_envelope_screen_seed{seed}"
            / "onesided_a4_tau0.5_b0.0005_rmax8"
            / "test_sweep.csv",
        ]
        for path in candidates:
            if path.is_file():
                found.append(path)
                break
    return found


def _nodetach_paths(out_root: Path, dataset: str) -> list[Path]:
    found = []
    for seed in SEEDS:
        path = out_root / dataset / "v16_nodetach" / f"seed{seed}" / "test_sweep.csv"
        if path.is_file():
            found.append(path)
    if found:
        return found
    fallback = (
        REPO_RESULTS
        / "cifar_vgg16_mne_component_ablation_seed42"
        / dataset
        / "comp_nodetach_fixed"
        / "test_sweep.csv"
    )
    return [fallback] if fallback.is_file() else []


def collect_series(out_root: Path) -> dict:
    series = {}
    for ds in ("cifar10", "cifar100"):
        series[ds] = {}
        for key, variant in FIVE_VARIANT.items():
            packed = _five_regs_series(ds, variant)
            if packed is not None:
                series[ds][key] = packed
        onesided = _onesided_paths(ds)
        if onesided:
            series[ds]["onesided"] = _agg_sweeps(onesided)
        nodetach = _nodetach_paths(out_root, ds)
        if nodetach:
            series[ds]["nodetach"] = _agg_sweeps(nodetach)
    return series


def plot_comparison(out_root: Path, plot_out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = collect_series(out_root)
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharex=True)
    rows = []
    for ax, ds, title, ylim in (
        (axes[0], "cifar10", "CIFAR-10 VGG-16 test", (0, 100)),
        (axes[1], "cifar100", "CIFAR-100 VGG-16 test", (0, 70)),
    ):
        for key, label in PLOT_METHODS:
            packed = series[ds].get(key)
            if packed is None:
                continue
            xs, ys, ss, n = packed
            nlab = f"n={n}" if n != 5 else "5-seed"
            ax.plot(
                xs,
                ys,
                color=PLOT_COLORS[key],
                linestyle=PLOT_LS[key],
                linewidth=2.0,
                label=f"{label} ({nlab})",
            )
            if n > 1:
                lo = [y - s for y, s in zip(ys, ss)]
                hi = [y + s for y, s in zip(ys, ss)]
                ax.fill_between(xs, lo, hi, color=PLOT_COLORS[key], alpha=0.15, linewidth=0)
            if 0.0 in xs and 5.0 in xs:
                i0, i5 = xs.index(0.0), xs.index(5.0)
                rows.append(
                    {
                        "dataset": ds,
                        "method": key,
                        "n_seeds": n,
                        "test_clean_mean": ys[i0],
                        "test_clean_std": ss[i0],
                        "test_sigma5_mean": ys[i5],
                        "test_sigma5_std": ss[i5],
                    }
                )
        ax.set_title(title)
        ax.set_xlabel(r"post-IF $\sigma$")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlim(0, 5)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower left")
    fig.suptitle(
        "VGG-16 5-seed · T=16 rate_uniform, post-IF  ·  "
        "no-detach vs One-sided / MNE-L2 / L2-wo / L2-all",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(plot_out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {plot_out}")
    if rows:
        csv_path = out_root / "nodetach_vs_baselines_test_mean_std.csv"
        out_root.mkdir(parents=True, exist_ok=True)
        write_csv(csv_path, rows)
        print(f"Wrote {csv_path}")
        print(f"{'dataset':<10} {'method':<10} {'n':>3} {'test0':>16} {'test5':>16}")
        for row in rows:
            print(
                f"{row['dataset']:<10} {row['method']:<10} {int(row['n_seeds']):3d} "
                f"{row['test_clean_mean']:7.2f}±{row['test_clean_std']:<6.2f} "
                f"{row['test_sigma5_mean']:7.2f}±{row['test_sigma5_std']:<6.2f}"
            )


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.plot:
        plot_comparison(args.out_root, args.plot_out)
        return
    if args.summarize:
        summarize(args.out_root)
        return

    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] {args.dataset} VGG-16 nodetach seed={args.seed}  "
        f"Ttrain=0 Teval={TEST_T} rc={MNE_RC} post_input_if",
        flush=True,
    )
    if args.dry_run:
        print("[DRY RUN]", " ".join(train_cmd(args)), flush=True)
        reused = reuse_ckpt(args)
        print(f"[DRY RUN] reuse={reused}", flush=True)
        return
    ckpt = train(args)
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = load_model(ckpt, device, args.dataset)
    model._mne_layer_map = "legacy"
    val_rows = sweep(model, val_loader(args, pin), device, "val", args.seed)
    write_csv(out / "val_sweep.csv", val_rows)
    test_rows = sweep(model, test_loader(args, pin), device, "test", args.seed)
    write_csv(out / "test_sweep.csv", test_rows)
    card = {
        "config": "v16_nodetach",
        "label": "MNE-L2 no-detach",
        "method": "nodetach",
        "dataset": args.dataset,
        "arch": ARCH,
        "seed": args.seed,
        "regularizer": "mne_l2",
        "weight_decay": 0.0,
        "reg_coeff": MNE_RC,
        "detach_lambda": False,
        "detach_bn_affine": False,
        "checkpoint": str(ckpt),
    }
    card.update(snn_metrics(val_rows, "val"))
    card.update(snn_metrics(test_rows, "test"))
    (out / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
