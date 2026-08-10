#!/usr/bin/env python3
"""
CIFAR-10/100 BN-free MLP: train + absolute post-input-IF noise sweep.

Methods (no BatchNorm in the architecture):
  no_reg                 - no regularization
  weight_decay_weights_only - L2 on Linear weights only
  manual_l2_all          - L2 on all trainable params (weights, bias, λ)
  old_detach             - MNE-standard (MNE-L2 on matched Linear weights;
                           λ used in formula but detached; bias not regularized)

Protocol: ANN train T=0, SNN test T=L, mode=rate_uniform.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


METHOD_SPECS = {
    "no_reg": {
        "regularizer": "weight_decay",
        "label": "No-reg",
        "reg_coeff": None,
        "weight_decay": 0.0,
        "train_args": [],
    },
    "weight_decay_weights_only": {
        "regularizer": "weight_decay_weights_only",
        "label": "L2 weights-only",
        "reg_coeff": None,
        "weight_decay": 5e-4,
        "train_args": [],
    },
    "manual_l2_all": {
        "regularizer": "manual_l2_all",
        "label": "L2 all",
        "reg_coeff": 2.5e-4,
        "weight_decay": 0.0,
        "train_args": [],
    },
    "old_detach": {
        "regularizer": "mne_l2",
        "label": "MNE-standard",
        "reg_coeff": 1e-4,
        "weight_decay": 0.0,
        "train_args": ["--mne_detach_lambda"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cifar10", "cifar100"],
        choices=["cifar10", "cifar100"],
    )
    parser.add_argument(
        "--arch",
        default="fc5_cifar",
        help="BN-free CIFAR MLP arch, e.g. fc5_cifar / fc5_cifar_w2 / fc3_cifar_h512",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_SPECS),
        choices=sorted(METHOD_SPECS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--train-T", type=int, default=0)
    parser.add_argument("--test-T", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--if-mode", default="rate_uniform")
    parser.add_argument("--spike-schedule", default="normal")
    parser.add_argument(
        "--first-layer-noise-position",
        default="post_input_if",
        choices=["post_input_if", "pre_input_if", "input_image"],
    )
    parser.add_argument("--first-layer-noise-type", default="gaussian")
    parser.add_argument("--noise-sigma-start", type=float, default=0.0)
    parser.add_argument("--noise-sigma-end", type=float, default=1.0)
    parser.add_argument("--noise-sigma-step", type=float, default=0.1)
    parser.add_argument("--mne-rc", type=float, default=1e-4, help="Override MNE-standard rc.")
    parser.add_argument("--l2-all-rc", type=float, default=2.5e-4, help="Override L2-all rc.")
    parser.add_argument("--l2-wd", type=float, default=5e-4, help="Override weights-only wd.")
    parser.add_argument("--ckpt-save-mode", default="best", choices=["best", "last"])
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--retest", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar_mlp_bnfree_four_regs",
    )
    args = parser.parse_args()
    # Apply coefficient overrides into local copies used by jobs.
    METHOD_SPECS["old_detach"]["reg_coeff"] = float(args.mne_rc)
    METHOD_SPECS["manual_l2_all"]["reg_coeff"] = float(args.l2_all_rc)
    METHOD_SPECS["weight_decay_weights_only"]["weight_decay"] = float(args.l2_wd)
    return args


def _rc_tag(rc) -> str:
    if rc is None:
        return "none"
    text = f"{float(rc):.8g}".replace("-", "m").replace(".", "p")
    return text


def _suffix(dataset: str, method: str, seed: int, args) -> str:
    spec = METHOD_SPECS[method]
    return (
        f"{dataset}_{args.arch}_{method}_rc{_rc_tag(spec['reg_coeff'])}"
        f"_seed{seed}_L{args.L}_trainT{args.train_T}"
    )


def _checkpoint_path(dataset: str, suffix: str, args) -> Path:
    return ROOT / f"{dataset}-checkpoints" / f"{args.arch}_L[{args.L}]_{suffix}.pth"


def _matrix_path(noise_dir: Path, dataset: str, seed: int, args) -> Path:
    return noise_dir / (
        f"noise_sweep_matrix_{dataset}_{args.arch}_T{args.test_T}"
        f"_mode_{args.if_mode}_schedule_{args.spike_schedule}_seed_{seed}.csv"
    )


def _run(cmd: list[str], dry_run: bool) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def train_one(dataset: str, method: str, seed: int, args) -> Path:
    spec = METHOD_SPECS[method]
    suffix = _suffix(dataset, method, seed, args)
    ckpt = _checkpoint_path(dataset, suffix, args)
    if ckpt.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {ckpt}", flush=True)
        return ckpt
    cmd = [
        sys.executable,
        str(ROOT / "main_train.py"),
        "-data",
        dataset,
        "-arch",
        args.arch,
        "-L",
        str(args.L),
        "-T",
        str(args.train_T),
        "--epochs",
        str(args.epochs),
        "-lr",
        str(args.lr),
        "-b",
        str(args.batch_size),
        "-j",
        str(args.workers),
        "--seed",
        str(seed),
        "--regularizer",
        spec["regularizer"],
        "-wd",
        str(spec["weight_decay"]),
        "--ckpt-save-mode",
        args.ckpt_save_mode,
        "-suffix",
        suffix,
    ]
    if spec["reg_coeff"] is not None:
        cmd += ["--reg_coeff", str(spec["reg_coeff"])]
    cmd += list(spec["train_args"])
    _run(cmd, dry_run=args.dry_run)
    return ckpt


def test_one(dataset: str, method: str, seed: int, ckpt: Path, args) -> tuple[Path, dict]:
    suffix = _suffix(dataset, method, seed, args)
    noise_dir = args.out_root / dataset / args.arch / method / f"seed_{seed}"
    matrix = _matrix_path(noise_dir, dataset, seed, args)
    if matrix.exists() and not args.retest:
        print(f"[SKIP TEST] {matrix}", flush=True)
        return matrix, _read_matrix(matrix, args.L)

    cmd = [
        sys.executable,
        str(ROOT / "main_test.py"),
        "-data",
        dataset,
        "-arch",
        args.arch,
        "-L",
        str(args.L),
        "-T",
        str(args.test_T),
        "--mode",
        args.if_mode,
        "--spike_schedule",
        args.spike_schedule,
        "--noise_sweep",
        "--noise_sigma_start",
        str(args.noise_sigma_start),
        "--noise_sigma_end",
        str(args.noise_sigma_end),
        "--noise_sigma_step",
        str(args.noise_sigma_step),
        "--noise_output_dir",
        str(noise_dir),
        "--first_layer_noise_position",
        args.first_layer_noise_position,
        "--first_layer_noise_type",
        args.first_layer_noise_type,
        "-w",
        str(ckpt),
        "-suffix",
        suffix,
        "-b",
        str(args.batch_size),
        "-j",
        str(args.workers),
        "--seed",
        str(seed),
    ]
    _run(cmd, dry_run=args.dry_run)
    if args.dry_run:
        return matrix, {}
    return matrix, _read_matrix(matrix, args.L)


def _read_matrix(path: Path, L: int) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing noise matrix: {path}")
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                row_l = int(float(row.get("L", "")))
            except ValueError:
                continue
            if row_l != int(L):
                continue
            values = {}
            for key, value in row.items():
                if key in ("L", "T") or value is None or str(value).strip() == "":
                    continue
                try:
                    values[float(key)] = float(value)
                except ValueError:
                    continue
            return dict(sorted(values.items()))
    raise ValueError(f"No row for L={L} in {path}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(raw_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for row in raw_rows:
        key = (row["dataset"], row["method"], row["method_label"], float(row["sigma"]))
        grouped[key].append(float(row["acc"]))

    mean_rows = []
    for (dataset, method, label, sigma), vals in sorted(grouped.items()):
        mean_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "method_label": label,
                "sigma": f"{sigma:.6g}",
                "n": len(vals),
                "acc_mean": f"{statistics.mean(vals):.6f}",
                "acc_std": (
                    f"{statistics.stdev(vals):.6f}" if len(vals) > 1 else "0.000000"
                ),
            }
        )

    by_curve = defaultdict(dict)
    for row in mean_rows:
        key = (row["dataset"], row["method"], row["method_label"])
        by_curve[key][float(row["sigma"])] = row

    summary_rows = []
    for (dataset, method, label), sigma_map in sorted(by_curve.items()):
        sigmas = sorted(sigma_map)
        clean = float(sigma_map[sigmas[0]]["acc_mean"])
        end = float(sigma_map[sigmas[-1]]["acc_mean"])
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "method_label": label,
                "clean_acc": f"{clean:.6f}",
                "end_sigma": f"{sigmas[-1]:.6g}",
                "end_acc": f"{end:.6f}",
                "absolute_drop": f"{clean - end:.6f}",
                "end_retention": f"{end / clean:.6f}" if clean > 0 else "nan",
            }
        )
    return mean_rows, summary_rows


def _plot(mean_rows: list[dict], out_root: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib unavailable; skip plots", flush=True)
        return

    order = ["no_reg", "weight_decay_weights_only", "manual_l2_all", "old_detach"]
    colors = {
        "no_reg": "#7f7f7f",
        "weight_decay_weights_only": "#2ca02c",
        "manual_l2_all": "#d62728",
        "old_detach": "#1f77b4",
    }
    by_dataset = defaultdict(list)
    for row in mean_rows:
        by_dataset[row["dataset"]].append(row)

    for dataset, rows in by_dataset.items():
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        for method in order:
            pts = [r for r in rows if r["method"] == method]
            if not pts:
                continue
            pts = sorted(pts, key=lambda r: float(r["sigma"]))
            xs = [float(p["sigma"]) for p in pts]
            ys = [float(p["acc_mean"]) for p in pts]
            yerr = [float(p["acc_std"]) for p in pts]
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker="o",
                linewidth=2.0,
                label=METHOD_SPECS[method]["label"],
                color=colors.get(method),
            )
        ax.set_xlabel(r"Absolute noise $\sigma$ (post-input-IF)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{dataset} BN-free MLP four-reg noise sweep")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        out = out_root / f"{dataset}_mlp_bnfree_four_regs_noise_sweep.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"[DONE] plot: {out}", flush=True)


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    for dataset in args.datasets:
        for method in args.methods:
            for seed in args.seeds:
                print(
                    f"[JOB] dataset={dataset} method={method} "
                    f"({METHOD_SPECS[method]['label']}) seed={seed}",
                    flush=True,
                )
                if args.skip_train:
                    suffix = _suffix(dataset, method, seed, args)
                    ckpt = _checkpoint_path(dataset, suffix, args)
                    if not ckpt.exists() and not args.dry_run:
                        raise FileNotFoundError(ckpt)
                else:
                    ckpt = train_one(dataset, method, seed, args)

                if args.skip_test:
                    continue
                matrix, values = test_one(dataset, method, seed, ckpt, args)
                for sigma, acc in values.items():
                    raw_rows.append(
                        {
                            "dataset": dataset,
                            "arch": args.arch,
                            "method": method,
                            "method_label": METHOD_SPECS[method]["label"],
                            "regularizer": METHOD_SPECS[method]["regularizer"],
                            "reg_coeff": (
                                ""
                                if METHOD_SPECS[method]["reg_coeff"] is None
                                else f"{METHOD_SPECS[method]['reg_coeff']:.8g}"
                            ),
                            "weight_decay": f"{METHOD_SPECS[method]['weight_decay']:.8g}",
                            "seed": seed,
                            "L": args.L,
                            "train_T": args.train_T,
                            "test_T": args.test_T,
                            "noise_position": args.first_layer_noise_position,
                            "sigma": f"{sigma:.6g}",
                            "acc": f"{acc:.6f}",
                            "checkpoint": str(ckpt),
                            "matrix_csv": str(matrix),
                        }
                    )

    if raw_rows:
        mean_rows, summary_rows = _aggregate(raw_rows)
        _write_csv(args.out_root / "cifar_mlp_bnfree_four_regs_raw.csv", raw_rows)
        _write_csv(args.out_root / "cifar_mlp_bnfree_four_regs_mean_std.csv", mean_rows)
        _write_csv(args.out_root / "cifar_mlp_bnfree_four_regs_summary.csv", summary_rows)
        _plot(mean_rows, args.out_root)
        print(f"[DONE] raw:     {args.out_root / 'cifar_mlp_bnfree_four_regs_raw.csv'}", flush=True)
        print(f"[DONE] mean:    {args.out_root / 'cifar_mlp_bnfree_four_regs_mean_std.csv'}", flush=True)
        print(f"[DONE] summary: {args.out_root / 'cifar_mlp_bnfree_four_regs_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
