from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ARCH = "vgg16"
DEFAULT_L = 16
DEFAULT_T = 16
DEFAULT_IF_MODE = "rate_uniform"
DEFAULT_SEEDS = [40, 41, 42, 43, 44]


def _fmt_float(x: float) -> str:
    text = f"{float(x):.6g}"
    return text.replace("+", "").replace("-", "m").replace(".", "p")


def _fmt_coeff(x) -> str:
    if x is None:
        return "none"
    return f"{float(x):.6g}"


def _rel_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _variant_specs(l_ref: float) -> dict:
    return {
        "old_no_detach": {
            "regularizer": "mne_l2",
            "label": "old_mne_no_detach_lambda",
            "train_args": [],
        },
        "old_detach": {
            "regularizer": "mne_l2",
            "label": "old_mne_detach_lambda",
            "train_args": ["--mne_detach_lambda"],
        },
        "lref_only": {
            "regularizer": "stable_mne_l2",
            "label": "stable_lref_only",
            "train_args": [
                "--mne_detach_lambda",
                "--stable_mne_no_fan_in_norm",
                "--stable_mne_layer_reduce",
                "sum",
                "--stable_mne_detach_bn_affine",
                "--stable_mne_l_ref",
                str(l_ref),
            ],
        },
        "fanin_mean": {
            "regularizer": "stable_mne_l2",
            "label": "stable_fanin_mean_bn_detached",
            "train_args": [
                "--mne_detach_lambda",
                "--stable_mne_detach_bn_affine",
                "--stable_mne_l_ref",
                str(l_ref),
            ],
        },
        "full_bn": {
            "regularizer": "stable_mne_l2",
            "label": "stable_fanin_mean_bn_affine_trainable",
            "train_args": [
                "--mne_detach_lambda",
                "--stable_mne_l_ref",
                str(l_ref),
            ],
        },
    }


def _baseline_specs() -> dict:
    return {
        "weight_decay": {
            "regularizer": "weight_decay",
            "label": "weight_decay",
            "reg_coeff": None,
            "weight_decay": 5e-4,
            "train_args": [],
        },
        "no_reg": {
            "regularizer": "weight_decay",
            "label": "no_regularization",
            "reg_coeff": None,
            "weight_decay": 0.0,
            "train_args": [],
        },
    }


def _suffix(dataset, variant_key, rc, seed, args) -> str:
    # Checkpoint identity is ANN (train T=0) + L; test T only affects noise matrix names.
    parts = [
        "mneablate",
        dataset,
        variant_key,
        f"rc{_fmt_float(rc)}" if rc is not None else "rcnone",
        f"seed{seed}",
        f"L{args.L}",
    ]
    if args.reg_warmup_epochs > 0:
        parts.append(f"warm{args.reg_warmup_epochs}")
    return "_".join(parts)


def _checkpoint_path(dataset: str, suffix: str, args) -> Path:
    ckpt_dir = ROOT / f"{dataset}-checkpoints"
    # ANN training: only L in filename (same convention as other CIFAR launchers).
    name = f"{args.arch}_L[{args.L}]_{suffix}.pth"
    return ckpt_dir / name


def _run(cmd, *, dry_run: bool):
    print(" ".join(str(x) for x in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def train_one(dataset: str, variant_key: str, spec: dict, rc, seed: int, args) -> Path:
    suffix = _suffix(dataset, variant_key, rc, seed, args)
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
        "0",  # always train ANN
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
        str(spec.get("weight_decay", args.weight_decay)),
        "--ckpt-save-mode",
        args.ckpt_save_mode,
        "-suffix",
        suffix,
    ]
    if rc is not None:
        cmd += ["--reg_coeff", str(rc)]
    if args.reg_warmup_epochs > 0 and spec["regularizer"] != "weight_decay":
        cmd += ["--reg_warmup_epochs", str(args.reg_warmup_epochs)]
    cmd += ["--mne_eps", str(args.mne_eps)]
    if args.mne_use_max:
        cmd += ["--mne_use_max"]
    cmd += list(spec.get("train_args", []))
    _run(cmd, dry_run=args.dry_run)
    return ckpt


def _matrix_path(noise_dir: Path, dataset: str, seed: int, args) -> Path:
    name = (
        f"noise_sweep_matrix_{dataset}_{args.arch}_T{args.T}_mode_{args.if_mode}"
        f"_schedule_{args.spike_schedule}_seed_{seed}.csv"
    )
    return noise_dir / name


def test_one(dataset: str, variant_key: str, rc, seed: int, ckpt: Path, args) -> tuple[Path, dict]:
    suffix = _suffix(dataset, variant_key, rc, seed, args)
    noise_dir = args.out_root / dataset / suffix
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
        str(args.T),
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
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
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


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["variant"],
            row["variant_label"],
            row["regularizer"],
            row["reg_coeff"],
            row["sigma"],
        )
        grouped[key].append(float(row["acc"]))

    mean_rows = []
    for key, vals in sorted(grouped.items()):
        dataset, variant, variant_label, regularizer, reg_coeff, sigma = key
        mean_rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "variant_label": variant_label,
                "regularizer": regularizer,
                "reg_coeff": reg_coeff,
                "sigma": sigma,
                "n": len(vals),
                "acc_mean": f"{statistics.mean(vals):.6f}",
                "acc_std": f"{statistics.stdev(vals):.6f}" if len(vals) > 1 else "0.000000",
            }
        )

    by_variant = defaultdict(dict)
    for row in mean_rows:
        key = (
            row["dataset"],
            row["variant"],
            row["variant_label"],
            row["regularizer"],
            row["reg_coeff"],
        )
        by_variant[key][float(row["sigma"])] = row

    summary_rows = []
    for key, sigma_rows in sorted(by_variant.items()):
        sigmas = sorted(sigma_rows)
        if not sigmas:
            continue
        start_sigma = sigmas[0]
        end_sigma = sigmas[-1]
        clean = float(sigma_rows[start_sigma]["acc_mean"])
        end_acc = float(sigma_rows[end_sigma]["acc_mean"])
        retention = end_acc / clean if clean > 0 else float("nan")
        dataset, variant, variant_label, regularizer, reg_coeff = key
        summary_rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "variant_label": variant_label,
                "regularizer": regularizer,
                "reg_coeff": reg_coeff,
                "clean_sigma": f"{start_sigma:.6g}",
                "clean_acc_mean": f"{clean:.6f}",
                "clean_acc_std": sigma_rows[start_sigma]["acc_std"],
                "end_sigma": f"{end_sigma:.6g}",
                "end_acc_mean": f"{end_acc:.6f}",
                "end_acc_std": sigma_rows[end_sigma]["acc_std"],
                "absolute_drop": f"{clean - end_acc:.6f}",
                "relative_retention": f"{retention:.6f}",
            }
        )
    return mean_rows, summary_rows


def _expand_jobs(args) -> list[dict]:
    specs = _variant_specs(args.stable_mne_l_ref)
    baseline_specs = _baseline_specs()
    jobs = []

    for baseline in args.baselines:
        spec = baseline_specs[baseline]
        for dataset in args.datasets:
            for seed in args.seeds:
                jobs.append(
                    {
                        "dataset": dataset,
                        "variant": baseline,
                        "spec": spec,
                        "reg_coeff": spec["reg_coeff"],
                        "seed": seed,
                    }
                )

    for variant in args.variants:
        spec = specs[variant]
        for dataset in args.datasets:
            for rc in args.rcs:
                for seed in args.seeds:
                    jobs.append(
                        {
                            "dataset": dataset,
                            "variant": variant,
                            "spec": spec,
                            "reg_coeff": rc,
                            "seed": seed,
                        }
                    )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MNE-L2 stability ablations and post-IF noise sweeps."
    )
    parser.add_argument("--datasets", nargs="+", default=["cifar10"], choices=["cifar10", "cifar100"])
    parser.add_argument("--arch", default=ARCH)
    parser.add_argument("--L", default=DEFAULT_L, type=int)
    parser.add_argument(
        "--T",
        default=DEFAULT_T,
        type=int,
        help="SNN timesteps used only for noise-sweep testing (training is always ANN T=0)",
    )
    parser.add_argument("--epochs", default=int(os.environ.get("CIFAR_EPOCHS", "300")), type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--batch-size", default=int(os.environ.get("CIFAR_BATCH", "128")), type=int)
    parser.add_argument("--workers", default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")), type=int)
    parser.add_argument("--seeds", nargs="+", default=DEFAULT_SEEDS, type=int)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["old_detach", "fanin_mean", "full_bn"],
        choices=sorted(_variant_specs(DEFAULT_L).keys()),
    )
    parser.add_argument("--rcs", nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], type=float)
    parser.add_argument("--baselines", nargs="*", default=["weight_decay"], choices=sorted(_baseline_specs().keys()))
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--mne-eps", default=1e-6, type=float)
    parser.add_argument("--mne-use-max", action="store_true")
    parser.add_argument("--stable-mne-l-ref", default=16.0, type=float)
    parser.add_argument("--reg-warmup-epochs", default=0, type=int)
    parser.add_argument("--if-mode", default=DEFAULT_IF_MODE)
    parser.add_argument("--spike-schedule", default="normal")
    parser.add_argument(
        "--first-layer-noise-position",
        default="post_input_if",
        choices=["post_input_if", "pre_input_if", "input_image"],
    )
    parser.add_argument("--first-layer-noise-type", default="gaussian", choices=["gaussian", "pink"])
    parser.add_argument("--noise-sigma-start", default=0.0, type=float)
    parser.add_argument("--noise-sigma-end", default=1.0, type=float)
    parser.add_argument("--noise-sigma-step", default=0.05, type=float)
    parser.add_argument("--ckpt-save-mode", default="best", choices=["best", "last"])
    parser.add_argument(
        "--out-root",
        default="../important_results/mne_stability_ablation_post_input_if",
        type=Path,
    )
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--retest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    return args


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    jobs = _expand_jobs(args)
    print(f"[INFO] expanded {len(jobs)} jobs", flush=True)

    raw_rows = []
    for job in jobs:
        dataset = job["dataset"]
        variant = job["variant"]
        spec = job["spec"]
        rc = job["reg_coeff"]
        seed = job["seed"]
        print(
            f"[JOB] dataset={dataset} variant={variant} rc={_fmt_coeff(rc)} seed={seed}",
            flush=True,
        )
        ckpt = train_one(dataset, variant, spec, rc, seed, args)
        matrix, sigma_to_acc = test_one(dataset, variant, rc, seed, ckpt, args)
        for sigma, acc in sigma_to_acc.items():
            raw_rows.append(
                {
                    "dataset": dataset,
                    "arch": args.arch,
                    "variant": variant,
                    "variant_label": spec["label"],
                    "regularizer": spec["regularizer"],
                    "reg_coeff": _fmt_coeff(rc),
                    "weight_decay": spec.get("weight_decay", args.weight_decay),
                    "seed": seed,
                    "L": args.L,
                    "T": args.T,
                    "if_mode": args.if_mode,
                    "spike_schedule": args.spike_schedule,
                    "noise_position": args.first_layer_noise_position,
                    "noise_type": args.first_layer_noise_type,
                    "sigma": f"{sigma:.6g}",
                    "acc": f"{acc:.6f}",
                    "checkpoint": _rel_path(ckpt),
                    "matrix_csv": _rel_path(matrix),
                }
            )

    if args.dry_run:
        print("[DRY RUN] no CSV written", flush=True)
        return

    raw_path = args.out_root / "mne_stability_ablation_raw.csv"
    mean_path = args.out_root / "mne_stability_ablation_mean_std.csv"
    summary_path = args.out_root / "mne_stability_ablation_summary.csv"

    raw_fields = [
        "dataset",
        "arch",
        "variant",
        "variant_label",
        "regularizer",
        "reg_coeff",
        "weight_decay",
        "seed",
        "L",
        "T",
        "if_mode",
        "spike_schedule",
        "noise_position",
        "noise_type",
        "sigma",
        "acc",
        "checkpoint",
        "matrix_csv",
    ]
    mean_rows, summary_rows = _aggregate(raw_rows)
    _write_csv(raw_path, raw_rows, raw_fields)
    _write_csv(
        mean_path,
        mean_rows,
        [
            "dataset",
            "variant",
            "variant_label",
            "regularizer",
            "reg_coeff",
            "sigma",
            "n",
            "acc_mean",
            "acc_std",
        ],
    )
    _write_csv(
        summary_path,
        summary_rows,
        [
            "dataset",
            "variant",
            "variant_label",
            "regularizer",
            "reg_coeff",
            "clean_sigma",
            "clean_acc_mean",
            "clean_acc_std",
            "end_sigma",
            "end_acc_mean",
            "end_acc_std",
            "absolute_drop",
            "relative_retention",
        ],
    )
    print(f"[DONE] raw: {raw_path}", flush=True)
    print(f"[DONE] mean/std: {mean_path}", flush=True)
    print(f"[DONE] summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
