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
        "calibrated_mne_a0p1": {
            "regularizer": "calibrated_mne_l2",
            "label": "l2_calibrated_mne_alpha_0p1",
            "weight_decay": 0.0,
            "train_args": [
                "--calibrated_mne_alpha",
                "0.1",
                "--calibrated_mne_risk_min",
                "0.5",
                "--calibrated_mne_risk_max",
                "2.0",
                "--calibrated_mne_alpha_start_epoch",
                "30",
                "--calibrated_mne_alpha_warmup_epochs",
                "50",
            ],
        },
        "old_no_detach": {
            "regularizer": "mne_l2",
            "label": "old_mne_no_detach_lambda",
            "train_args": [],
        },
        "raw_w_lambda": {
            "regularizer": "mne_l2",
            "label": "mne_raw_weight_lambda_trainable",
            "train_args": ["--mne_no_bn_fold"],
        },
        "folded_w_lambda": {
            "regularizer": "mne_l2",
            "label": "mne_bn_folded_weight_lambda_trainable",
            "train_args": [],
        },
        "l2_numerator": {
            "regularizer": "mne_l2",
            "label": "mne_l2_numerator_lambda_trainable",
            "train_args": ["--mne_no_bn_fold", "--mne_frobenius"],
        },
        "l2_numerator_detach": {
            "regularizer": "mne_l2",
            "label": "mne_l2_numerator_detach_lambda",
            "train_args": ["--mne_no_bn_fold", "--mne_frobenius", "--mne_detach_lambda"],
        },
        "old_detach": {
            "regularizer": "mne_l2",
            "label": "old_mne_detach_lambda",
            "train_args": ["--mne_detach_lambda"],
        },
        "mne_l2_all": {
            "regularizer": "mne_l2_all",
            "label": "mne_l2_plus_residual_l2_all_params",
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
        "hinge_log": {
            "regularizer": "hinge_mne",
            "label": "hinge_mne_log",
            "train_args": ["--mne_detach_lambda"],
        },
        "hinge_linear": {
            "regularizer": "hinge_mne",
            "label": "hinge_mne_linear",
            "train_args": ["--mne_detach_lambda", "--hinge_mne_linear"],
        },
        "hinge_log_fanin": {
            "regularizer": "hinge_mne",
            "label": "hinge_mne_log_fanin_norm",
            "train_args": [
                "--mne_detach_lambda",
                "--hinge_mne_normalize_by_fan_in",
            ],
        },
        "group_lasso": {
            "regularizer": "group_lasso",
            "label": "filter_group_lasso",
            "train_args": [],
        },
        "spectral_norm": {
            "regularizer": "spectral_norm",
            "label": "spectral_norm_power_iteration",
            "train_args": [],
        },
        "orthogonal": {
            "regularizer": "orthogonal",
            "label": "orthogonal_frobenius",
            "train_args": [],
        },
        "manual_l2": {
            "regularizer": "manual_l2",
            "label": "manual_l2_no_weight_decay",
            "train_args": [],
        },
        "l1_all": {
            "regularizer": "l1_all",
            "label": "l1_all_parameters",
            "train_args": [],
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
        "weight_decay_weights_only": {
            "regularizer": "weight_decay_weights_only",
            "label": "weight_decay_conv_linear_only",
            "reg_coeff": None,
            "weight_decay": 5e-4,
            "train_args": [],
        },
        "manual_l2_all": {
            "regularizer": "manual_l2_all",
            "label": "manual_l2_all_parameters",
            "reg_coeff": 2.5e-4,
            "weight_decay": 0.0,
            "train_args": [],
        },
        "manual_l2_w_bn": {
            "regularizer": "manual_l2_w_bn",
            "label": "manual_l2_weights_plus_bn",
            "reg_coeff": 2.5e-4,
            "weight_decay": 0.0,
            "train_args": [],
        },
        "manual_l2_w_bn_gamma": {
            "regularizer": "manual_l2_w_bn_gamma",
            "label": "manual_l2_weights_plus_bn_gamma",
            "reg_coeff": 2.5e-4,
            "weight_decay": 0.0,
            "train_args": [],
        },
        "manual_l2_w_bn_beta": {
            "regularizer": "manual_l2_w_bn_beta",
            "label": "manual_l2_weights_plus_bn_beta",
            "reg_coeff": 2.5e-4,
            "weight_decay": 0.0,
            "train_args": [],
        },
        "manual_l2_w_if": {
            "regularizer": "manual_l2_w_if",
            "label": "manual_l2_weights_plus_if",
            "reg_coeff": 2.5e-4,
            "weight_decay": 0.0,
            "train_args": [],
        },
        "manual_l2_w_bn_if": {
            "regularizer": "manual_l2_w_bn_if",
            "label": "manual_l2_weights_plus_bn_plus_if",
            "reg_coeff": 2.5e-4,
            "weight_decay": 0.0,
            "train_args": [],
        },
        "no_reg": {
            "regularizer": "weight_decay",
            "label": "no_regularization",
            "reg_coeff": None,
            "weight_decay": 0.0,
            "train_args": [],
        },
        "l1": {
            "regularizer": "l1",
            "label": "l1_rc1e-5",
            "reg_coeff": 1e-5,
            "weight_decay": 0.0,
            "train_args": [],
        },
        "l1_all": {
            "regularizer": "l1_all",
            "label": "l1_all_parameters_rc1e-5",
            "reg_coeff": 1e-5,
            "weight_decay": 0.0,
            "train_args": [],
        },
    }


def _suffix(dataset, variant_key, rc, seed, args, hinge_tau=None) -> str:
    # Checkpoint identity follows training T; test T only affects noise matrix names.
    parts = [
        "mneablate",
        dataset,
        variant_key,
        f"rc{_fmt_float(rc)}" if rc is not None else "rcnone",
        f"seed{seed}",
        f"L{args.L}",
        f"trainT{args.train_T}",
    ]
    if hinge_tau is not None:
        parts.append(f"tau{_fmt_float(hinge_tau)}")
    if getattr(args, "reg_warmup_epochs", 0) > 0:
        parts.append(f"warm{args.reg_warmup_epochs}")
    return "_".join(parts)


def _checkpoint_path(dataset: str, suffix: str, args) -> Path:
    ckpt_dir = ROOT / f"{dataset}-checkpoints"
    name = f"{args.arch}_L[{args.L}]"
    if args.train_T > 0:
        name += f"_T[{args.train_T}]"
    name += f"_{suffix}.pth"
    return ckpt_dir / name


def _legacy_suffix(suffix: str) -> str:
    return "_".join(part for part in suffix.split("_") if not part.startswith("trainT"))


def _resolve_checkpoint(dataset: str, variant_key: str, rc, seed: int, hinge_tau, args) -> Path:
    suffix = _suffix(dataset, variant_key, rc, seed, args, hinge_tau=hinge_tau)
    ckpt = _checkpoint_path(dataset, suffix, args)
    if ckpt.exists():
        return ckpt
    if int(args.train_T) == 0:
        legacy = _checkpoint_path(dataset, _legacy_suffix(suffix), args)
        if legacy.exists():
            print(f"[CKPT FALLBACK] {legacy}", flush=True)
            return legacy
    if variant_key == "l1":
        alt = (
            f"strict_seed{seed}_schemeC_noout_l1_l{args.L}_{args.arch}"
            f"_rc{_fmt_float(1e-5)}"
        )
        alt_ckpt = _checkpoint_path(dataset, alt, args)
        if alt_ckpt.exists():
            print(f"[CKPT FALLBACK] {alt_ckpt}", flush=True)
            return alt_ckpt
    return ckpt


def _run(cmd, *, dry_run: bool):
    print(" ".join(str(x) for x in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def train_one(dataset: str, variant_key: str, spec: dict, rc, seed: int, hinge_tau, args) -> Path:
    suffix = _suffix(dataset, variant_key, rc, seed, args, hinge_tau=hinge_tau)
    ckpt = _resolve_checkpoint(dataset, variant_key, rc, seed, hinge_tau, args)
    if ckpt.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {ckpt}", flush=True)
        return ckpt
    if args.test_only:
        if args.dry_run:
            print(f"[DRY RUN] would require checkpoint {ckpt}", flush=True)
            return ckpt
        raise FileNotFoundError(f"--test-only but checkpoint is missing: {ckpt}")

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
        str(spec.get("weight_decay", args.weight_decay)),
        "--ckpt-save-mode",
        args.ckpt_save_mode,
        "-suffix",
        suffix,
    ]
    if rc is not None:
        cmd += ["--reg_coeff", str(rc)]
    if args.reg_warmup_epochs > 0 and spec["regularizer"] in (
        "mne_l2",
        "mne_l2_all",
        "stable_mne_l2",
        "hinge_mne",
        "conv_mne_l2",
        "group_lasso",
        "spectral_norm",
        "orthogonal",
        "manual_l2",
    ):
        cmd += ["--reg_warmup_epochs", str(args.reg_warmup_epochs)]
    cmd += ["--mne_eps", str(args.mne_eps)]
    if args.mne_use_max:
        cmd += ["--mne_use_max"]
    if hinge_tau is not None:
        cmd += ["--hinge_mne_tau", str(hinge_tau)]
    if spec["regularizer"] == "group_lasso":
        cmd += ["--group_lasso_eps", str(args.group_lasso_eps)]
    if spec["regularizer"] == "spectral_norm":
        cmd += ["--spectral_power_iters", str(args.spectral_power_iters)]
    cmd += list(spec.get("train_args", []))
    _run(cmd, dry_run=args.dry_run)
    return ckpt


def _matrix_path(noise_dir: Path, dataset: str, seed: int, args) -> Path:
    name = (
        f"noise_sweep_matrix_{dataset}_{args.arch}_T{args.test_T}_mode_{args.if_mode}"
        f"_schedule_{args.spike_schedule}_seed_{seed}.csv"
    )
    return noise_dir / name


def test_one(dataset: str, variant_key: str, rc, seed: int, hinge_tau, ckpt: Path, args) -> tuple[Path, dict]:
    suffix = _suffix(dataset, variant_key, rc, seed, args, hinge_tau=hinge_tau)
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
            row["hinge_tau"],
            row["sigma"],
        )
        grouped[key].append(float(row["acc"]))

    mean_rows = []
    for key, vals in sorted(grouped.items()):
        dataset, variant, variant_label, regularizer, reg_coeff, hinge_tau, sigma = key
        mean_rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "variant_label": variant_label,
                "regularizer": regularizer,
                "reg_coeff": reg_coeff,
                "hinge_tau": hinge_tau,
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
            row["hinge_tau"],
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
        dataset, variant, variant_label, regularizer, reg_coeff, hinge_tau = key
        summary_rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "variant_label": variant_label,
                "regularizer": regularizer,
                "reg_coeff": reg_coeff,
                "hinge_tau": hinge_tau,
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
                        "hinge_tau": None,
                        "seed": seed,
                    }
                )

    for variant in args.variants:
        spec = specs[variant]
        tau_values = args.hinge_taus if spec["regularizer"] == "hinge_mne" else [None]
        if spec["regularizer"] == "group_lasso":
            rc_values = args.group_lasso_rcs
        elif spec["regularizer"] == "spectral_norm":
            rc_values = args.spectral_norm_rcs
        elif spec["regularizer"] == "orthogonal":
            rc_values = args.orthogonal_rcs
        elif spec["regularizer"] == "manual_l2":
            rc_values = args.manual_l2_rcs
        elif spec["regularizer"] == "l1_all":
            rc_values = args.l1_all_rcs
        elif spec["regularizer"] == "calibrated_mne_l2":
            rc_values = args.calibrated_mne_rcs
        else:
            rc_values = args.rcs
        for dataset in args.datasets:
            for rc in rc_values:
                for hinge_tau in tau_values:
                    for seed in args.seeds:
                        jobs.append(
                            {
                                "dataset": dataset,
                                "variant": variant,
                                "spec": spec,
                                "reg_coeff": rc,
                                "hinge_tau": hinge_tau,
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
        "--train-T",
        dest="train_T",
        default=0,
        type=int,
        help="Training timesteps. Use 0 for ANN training before SNN conversion.",
    )
    parser.add_argument(
        "--test-T",
        dest="test_T",
        default=DEFAULT_T,
        type=int,
        help="SNN timesteps used for converted-model noise-sweep testing.",
    )
    parser.add_argument(
        "--T",
        dest="test_T",
        type=int,
        help="Backward-compatible alias for --test-T.",
    )
    parser.add_argument("--epochs", default=int(os.environ.get("CIFAR_EPOCHS", "300")), type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--batch-size", default=int(os.environ.get("CIFAR_BATCH", "128")), type=int)
    parser.add_argument("--workers", default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")), type=int)
    parser.add_argument("--seeds", nargs="+", default=DEFAULT_SEEDS, type=int)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=["old_detach", "hinge_log"],
        choices=sorted(_variant_specs(DEFAULT_L).keys()),
        help="Ablation variants. Pass --variants with no values to run baselines only.",
    )
    parser.add_argument("--rcs", nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], type=float)
    parser.add_argument(
        "--group-lasso-rcs",
        nargs="+",
        default=[1e-5, 3e-5, 1e-4],
        type=float,
        help="Coefficient grid used only by the group_lasso variant.",
    )
    parser.add_argument(
        "--spectral-norm-rcs",
        nargs="+",
        default=[1e-4, 3e-4, 1e-3],
        type=float,
        help="Coefficient grid used only by the spectral_norm variant.",
    )
    parser.add_argument(
        "--orthogonal-rcs",
        nargs="+",
        default=[1e-4],
        type=float,
        help="Coefficient grid used only by the orthogonal variant.",
    )
    parser.add_argument(
        "--manual-l2-rcs",
        nargs="+",
        default=[2.5e-4],
        type=float,
        help="Coefficient grid for explicit sum(||W||^2); 2.5e-4 matches weight_decay=5e-4 gradients.",
    )
    parser.add_argument(
        "--l1-all-rcs",
        nargs="+",
        default=[1e-5],
        type=float,
        help="Coefficient grid used only by the l1_all variant.",
    )
    parser.add_argument(
        "--calibrated-mne-rcs",
        nargs="+",
        default=[5e-4],
        type=float,
        help="reg_coeff for calibrated_mne_l2; 5e-4 matches weights-only WD when alpha=0.",
    )
    parser.add_argument("--group-lasso-eps", default=1e-12, type=float)
    parser.add_argument("--spectral-power-iters", default=3, type=int)
    parser.add_argument("--hinge-taus", nargs="+", default=[1.0, 2.0, 4.0], type=float)
    parser.add_argument("--baselines", nargs="*", default=["weight_decay", "l1"], choices=sorted(_baseline_specs().keys()))
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
        choices=["post_input_if", "pre_input_if", "pre_first_conv", "input_image"],
        help=(
            "post_input_if=首个 IF 后；pre_input_if=Conv/BN 后、首个 IF 前；"
            "pre_first_conv=展开时间维后、首个 Conv/BN 前；input_image=直接加到输入图像"
        ),
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
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Skip training and fail if the checkpoint is missing.",
    )
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
        hinge_tau = job["hinge_tau"]
        seed = job["seed"]
        print(
            f"[JOB] dataset={dataset} variant={variant} rc={_fmt_coeff(rc)} tau={_fmt_coeff(hinge_tau)} seed={seed}",
            flush=True,
        )
        ckpt = train_one(dataset, variant, spec, rc, seed, hinge_tau, args)
        matrix, sigma_to_acc = test_one(dataset, variant, rc, seed, hinge_tau, ckpt, args)
        for sigma, acc in sigma_to_acc.items():
            raw_rows.append(
                {
                    "dataset": dataset,
                    "arch": args.arch,
                    "variant": variant,
                    "variant_label": spec["label"],
                    "regularizer": spec["regularizer"],
                    "reg_coeff": _fmt_coeff(rc),
                    "hinge_tau": _fmt_coeff(hinge_tau),
                    "weight_decay": spec.get("weight_decay", args.weight_decay),
                    "seed": seed,
                    "L": args.L,
                    "train_T": args.train_T,
                    "test_T": args.test_T,
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
        "hinge_tau",
        "weight_decay",
        "seed",
        "L",
        "train_T",
        "test_T",
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
            "hinge_tau",
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
            "hinge_tau",
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
