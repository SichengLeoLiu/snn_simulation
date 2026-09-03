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


DEFAULT_MODELS = ["fc3_h8", "fc3_h256", "cnn2", "cnn2_c4_c8"]
DEFAULT_SEEDS = [40]


def _fmt_float(x) -> str:
    if x is None:
        return "none"
    text = f"{float(x):.6g}"
    return text.replace("+", "").replace("-", "m").replace(".", "p")


def _rel_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _parse_rc_map(entries: list[str] | None) -> dict[str, float]:
    rc_map = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(
                "--spectral-mne-rc-map entries must look like model=reg_coeff"
            )
        model, value = entry.split("=", 1)
        model = model.strip()
        if not model:
            raise ValueError("--spectral-mne-rc-map contains an empty model name")
        rc_map[model] = float(value)
    return rc_map


def _method_specs(args) -> dict:
    return {
        "old_detach": {
            "regularizer": "mne_l2",
            "label": "old_mne_detach_lambda",
            "rcs": args.old_mne_rcs,
            "wd": 0.0,
            "train_args": ["--mne_detach_lambda"],
        },
        "old_detach_current": {
            "regularizer": "mne_l2",
            "label": "old_mne_detach_lambda_current",
            "rcs": args.old_mne_rcs,
            "wd": 0.0,
            "train_args": ["--mne_detach_lambda"],
        },
        "mne_scale_trainable": {
            "regularizer": "mne_l2",
            "label": "mne_scale_trainable",
            "rcs": args.old_mne_rcs,
            "wd": 0.0,
            "train_args": ["--mne_no_detach_bn_stats"],
        },
        "mne_global_wd_all": {
            "regularizer": "mne_l2",
            "label": f"mne_detach_plus_wd_all{_fmt_float(args.weight_decay)}",
            "rcs": args.old_mne_rcs,
            "wd": args.weight_decay,
            "train_args": ["--mne_detach_lambda"],
        },
        "spectral_mne": {
            "regularizer": "spectral_mne",
            "label": "spectral_mne_detach_lambda",
            "rcs": args.spectral_mne_rcs,
            "rc_map": args.spectral_mne_rc_map,
            "wd": 0.0,
            "train_args": [
                "--mne_detach_lambda",
                "--spectral_mne_power_iters",
                str(args.spectral_mne_power_iters),
                "--spectral_mne_layer_reduce",
                args.spectral_mne_layer_reduce,
            ],
        },
        "l2": {
            "regularizer": "weight_decay",
            "label": f"l2_wd{_fmt_float(args.weight_decay)}",
            "rcs": [None],
            "wd": args.weight_decay,
            "train_args": [],
        },
        "wd_all_current": {
            "regularizer": "weight_decay",
            "label": f"wd_all_current{_fmt_float(args.weight_decay)}",
            "rcs": [None],
            "wd": args.weight_decay,
            "train_args": [],
        },
        "manual_l2_all": {
            "regularizer": "manual_l2_all",
            "label": f"manual_l2_all_rc{_fmt_float(args.matched_l2_rc)}",
            "rcs": [args.matched_l2_rc],
            "wd": 0.0,
            "train_args": [],
        },
        "manual_l2_w_bn": {
            "regularizer": "manual_l2_w_bn",
            "label": f"manual_l2_weights_bn_rc{_fmt_float(args.matched_l2_rc)}",
            "rcs": [args.matched_l2_rc],
            "wd": 0.0,
            "train_args": [],
        },
        "manual_l2_w_if": {
            "regularizer": "manual_l2_w_if",
            "label": f"manual_l2_weights_if_rc{_fmt_float(args.matched_l2_rc)}",
            "rcs": [args.matched_l2_rc],
            "wd": 0.0,
            "train_args": [],
        },
        "manual_l2_w_bn_if": {
            "regularizer": "manual_l2_w_bn_if",
            "label": f"manual_l2_weights_bn_if_rc{_fmt_float(args.matched_l2_rc)}",
            "rcs": [args.matched_l2_rc],
            "wd": 0.0,
            "train_args": [],
        },
        "manual_l2": {
            "regularizer": "manual_l2",
            "label": f"manual_l2_weight_only_rc{_fmt_float(args.matched_l2_rc)}",
            "rcs": [args.matched_l2_rc],
            "wd": 0.0,
            "train_args": [],
        },
        "wd_weight_only": {
            "regularizer": "weight_decay_weights_only",
            "label": f"wd_weight_only{_fmt_float(args.weight_decay)}",
            "rcs": [None],
            "wd": args.weight_decay,
            "train_args": [],
        },
        "weight_decay": {
            "regularizer": "weight_decay",
            "label": "weight_decay",
            "rcs": [None],
            "wd": args.weight_decay,
            "train_args": [],
        },
        "no_reg": {
            "regularizer": "weight_decay",
            "label": "no_regularization",
            "rcs": [None],
            "wd": 0.0,
            "train_args": [],
        },
        "no_reg_current": {
            "regularizer": "weight_decay",
            "label": "no_regularization_current",
            "rcs": [None],
            "wd": 0.0,
            "train_args": [],
        },
        "l1": {
            "regularizer": "l1",
            "label": f"l1_rc{_fmt_float(args.l1_rc)}",
            "rcs": [args.l1_rc],
            "wd": 0.0,
            "train_args": [],
        },
        "l1_all": {
            "regularizer": "l1_all",
            "label": f"l1_all_rc{_fmt_float(args.l1_all_rc)}",
            "rcs": [args.l1_all_rc],
            "wd": 0.0,
            "train_args": [],
        },
        "elastic_net_all": {
            "regularizer": "elastic_net_all",
            "label": f"elastic_net_all_rc{_fmt_float(args.elastic_net_rc)}",
            "rcs": [args.elastic_net_rc],
            "wd": 0.0,
            "train_args": ["--elastic_l1_ratio", str(args.elastic_l1_ratio)],
        },
        "scale_l2": {
            "regularizer": "scale_l2",
            "label": f"scale_l2_rc{_fmt_float(args.scale_l2_rc)}",
            "rcs": [args.scale_l2_rc],
            "wd": 0.0,
            "train_args": [],
        },
        "stable_fanin_mean": {
            "regularizer": "stable_mne_l2",
            "label": f"stable_fanin_mean_rc{_fmt_float(args.stable_mne_rc)}",
            "rcs": [args.stable_mne_rc],
            "wd": 0.0,
            "train_args": [
                "--mne_detach_lambda",
                "--stable_mne_detach_bn_affine",
                "--stable_mne_l_ref",
                str(args.stable_mne_l_ref),
            ],
        },
        "orthogonal": {
            "regularizer": "orthogonal",
            "label": f"orthogonal_rc{_fmt_float(args.orthogonal_rc)}",
            "rcs": [args.orthogonal_rc],
            "wd": 0.0,
            "train_args": [],
        },
        "effective_l2": {
            "regularizer": "effective_l2",
            "label": f"effective_l2_rc{_fmt_float(args.effective_l2_rc)}",
            "rcs": [args.effective_l2_rc],
            "wd": 0.0,
            "train_args": [],
        },
        "threshold_l2": {
            "regularizer": "threshold_l2",
            "label": f"threshold_l2_rc{_fmt_float(args.threshold_l2_rc)}",
            "rcs": [args.threshold_l2_rc],
            "wd": 0.0,
            "train_args": [],
        },
        "l2_sp": {
            "regularizer": "l2_sp",
            "label": f"l2_sp_rc{_fmt_float(args.l2_sp_rc)}",
            "rcs": [args.l2_sp_rc],
            "wd": 0.0,
            "train_args": [],
        },
        "group_lasso": {
            "regularizer": "group_lasso",
            "label": f"group_lasso_rc{_fmt_float(args.group_lasso_rc)}",
            "rcs": [args.group_lasso_rc],
            "wd": 0.0,
            "train_args": [],
        },
        "spectral_norm": {
            "regularizer": "spectral_norm",
            "label": f"spectral_norm_rc{_fmt_float(args.spectral_norm_rc)}",
            "rcs": [args.spectral_norm_rc],
            "wd": 0.0,
            "train_args": [],
        },
    }


def _suffix(model: str, method: str, rc, seed: int, args) -> str:
    dataset_tag = "fashion" if args.dataset == "fashion_mnist" else args.dataset
    parts = [
        f"{dataset_tag}_spectral_mne",
        model,
        method,
        f"rc{_fmt_float(rc)}",
        f"seed{seed}",
        f"ep{args.epochs}",
        f"L{args.L}",
        f"trainT{args.train_T}",
    ]
    if args.reg_warmup_epochs > 0 and method in ("old_detach", "spectral_mne"):
        parts.append(f"warm{args.reg_warmup_epochs}")
    return "_".join(parts)


def _checkpoint_path(model: str, suffix: str, args) -> Path:
    name = f"{model}_L[{args.L}]"
    if args.train_T > 0:
        name += f"_T[{args.train_T}]"
    return ROOT / f"{args.dataset}-checkpoints" / f"{name}_{suffix}.pth"


def _rcs_for_model(spec: dict, model: str) -> list:
    rc_map = spec.get("rc_map") or {}
    if model in rc_map:
        return [rc_map[model]]
    return spec["rcs"]


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(str(x) for x in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def train_one(model: str, method: str, spec: dict, rc, seed: int, args) -> Path:
    suffix = _suffix(model, method, rc, seed, args)
    ckpt = _checkpoint_path(model, suffix, args)
    if ckpt.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {ckpt}", flush=True)
        return ckpt

    cmd = [
        sys.executable,
        str(ROOT / "main_train.py"),
        "-data",
        args.dataset,
        "-arch",
        model,
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
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--regularizer",
        spec["regularizer"],
        "-wd",
        str(spec["wd"]),
        "--ckpt-save-mode",
        args.ckpt_save_mode,
        "-suffix",
        suffix,
        "--mne_eps",
        str(args.mne_eps),
    ]
    if rc is not None:
        cmd += ["--reg_coeff", str(rc)]
    if args.reg_warmup_epochs > 0 and spec["regularizer"] != "weight_decay":
        cmd += ["--reg_warmup_epochs", str(args.reg_warmup_epochs)]
    if spec["regularizer"] == "group_lasso":
        cmd += ["--group_lasso_eps", str(args.group_lasso_eps)]
    if spec["regularizer"] == "spectral_norm":
        cmd += ["--spectral_power_iters", str(args.spectral_power_iters)]
    cmd += list(spec.get("train_args", []))
    _run(cmd, dry_run=args.dry_run)
    return ckpt


def _matrix_path(noise_dir: Path, model: str, seed: int, args) -> Path:
    name = (
        f"noise_sweep_matrix_{args.dataset}_{model}_T{args.test_T}_mode_{args.if_mode}"
        f"_schedule_{args.spike_schedule}_seed_{seed}.csv"
    )
    return noise_dir / name


def test_one(model: str, method: str, rc, seed: int, ckpt: Path, args) -> tuple[Path, dict]:
    suffix = _suffix(model, method, rc, seed, args)
    noise_dir = args.out_root / model / method / f"rc{_fmt_float(rc)}" / f"seed_{seed}"
    matrix = _matrix_path(noise_dir, model, seed, args)
    if matrix.exists() and not args.retest:
        print(f"[SKIP TEST] {matrix}", flush=True)
        return matrix, _read_matrix(matrix, args.L)

    cmd = [
        sys.executable,
        str(ROOT / "main_test.py"),
        "-data",
        args.dataset,
        "-arch",
        model,
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
        "--device",
        args.device,
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
            row["model"],
            row["method"],
            row["method_label"],
            row["regularizer"],
            row["reg_coeff"],
            row["sigma"],
        )
        grouped[key].append(float(row["acc"]))

    mean_rows = []
    for key, vals in sorted(grouped.items()):
        model, method, label, regularizer, rc, sigma = key
        mean_rows.append(
            {
                "model": model,
                "method": method,
                "method_label": label,
                "regularizer": regularizer,
                "reg_coeff": rc,
                "sigma": sigma,
                "n": len(vals),
                "acc_mean": f"{statistics.mean(vals):.6f}",
                "acc_std": f"{statistics.stdev(vals):.6f}" if len(vals) > 1 else "0.000000",
            }
        )

    by_method = defaultdict(dict)
    for row in mean_rows:
        key = (
            row["model"],
            row["method"],
            row["method_label"],
            row["regularizer"],
            row["reg_coeff"],
        )
        by_method[key][float(row["sigma"])] = row

    summary_rows = []
    for key, sigma_rows in sorted(by_method.items()):
        sigmas = sorted(sigma_rows)
        if not sigmas:
            continue
        clean_sigma = sigmas[0]
        end_sigma = sigmas[-1]
        clean = float(sigma_rows[clean_sigma]["acc_mean"])
        end_acc = float(sigma_rows[end_sigma]["acc_mean"])
        model, method, label, regularizer, rc = key
        summary_rows.append(
            {
                "model": model,
                "method": method,
                "method_label": label,
                "regularizer": regularizer,
                "reg_coeff": rc,
                "clean_sigma": f"{clean_sigma:.6g}",
                "clean_acc_mean": f"{clean:.6f}",
                "clean_acc_std": sigma_rows[clean_sigma]["acc_std"],
                "end_sigma": f"{end_sigma:.6g}",
                "end_acc_mean": f"{end_acc:.6f}",
                "end_acc_std": sigma_rows[end_sigma]["acc_std"],
                "absolute_drop": f"{clean - end_acc:.6f}",
                "relative_retention": f"{(end_acc / clean if clean > 0 else float('nan')):.6f}",
            }
        )
    return mean_rows, summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MNIST-family FC3/CNN2 Spectral-MNE ablation."
    )
    parser.add_argument(
        "--dataset",
        default="fashion_mnist",
        choices=["fashion_mnist", "mnist"],
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["old_detach", "spectral_mne"],
        choices=[
            "old_detach",
            "old_detach_current",
            "mne_scale_trainable",
            "mne_global_wd_all",
            "spectral_mne",
            "l2",
            "wd_all_current",
            "manual_l2_all",
            "manual_l2_w_bn",
            "manual_l2_w_if",
            "manual_l2_w_bn_if",
            "manual_l2",
            "wd_weight_only",
            "weight_decay",
            "no_reg",
            "no_reg_current",
            "l1",
            "l1_all",
            "elastic_net_all",
            "scale_l2",
            "stable_fanin_mean",
            "orthogonal",
            "effective_l2",
            "threshold_l2",
            "l2_sp",
            "group_lasso",
            "spectral_norm",
        ],
    )
    parser.add_argument("--old-mne-rcs", nargs="+", default=[1e-4], type=float)
    parser.add_argument("--spectral-mne-rcs", nargs="+", default=[1e-6, 3e-6, 1e-5], type=float)
    parser.add_argument(
        "--spectral-mne-rc-map",
        nargs="*",
        default=None,
        help="Optional per-model Spectral-MNE coefficients, e.g. fc3_h8=1e-6 cnn2=1e-5.",
    )
    parser.add_argument("--l1-rc", default=1e-5, type=float)
    parser.add_argument("--l1-all-rc", default=1e-5, type=float)
    parser.add_argument("--elastic-net-rc", default=2.5e-4, type=float)
    parser.add_argument("--elastic-l1-ratio", default=0.04, type=float)
    parser.add_argument("--scale-l2-rc", default=2.5e-4, type=float)
    parser.add_argument("--stable-mne-rc", default=1e-4, type=float)
    parser.add_argument("--stable-mne-l-ref", default=16.0, type=float)
    parser.add_argument("--orthogonal-rc", default=1e-4, type=float)
    parser.add_argument("--effective-l2-rc", default=1e-4, type=float)
    parser.add_argument("--threshold-l2-rc", default=1e-4, type=float)
    parser.add_argument("--l2-sp-rc", default=1e-2, type=float)
    parser.add_argument("--group-lasso-rc", default=1e-4, type=float)
    parser.add_argument("--spectral-norm-rc", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument(
        "--matched-l2-rc",
        default=2.5e-4,
        type=float,
        help="Coefficient for sum(p^2); use weight_decay/2 for a matched gradient.",
    )
    parser.add_argument("--L", default=16, type=int)
    parser.add_argument("--train-T", dest="train_T", default=0, type=int)
    parser.add_argument("--test-T", dest="test_T", default=16, type=int)
    parser.add_argument("--epochs", default=int(os.environ.get("FASHION_EPOCHS", "20")), type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--batch-size", default=int(os.environ.get("FASHION_BATCH", "128")), type=int)
    parser.add_argument("--workers", default=int(os.environ.get("FASHION_NUM_WORKERS", "2")), type=int)
    parser.add_argument("--device", default=os.environ.get("FASHION_DEVICE", "auto"))
    parser.add_argument("--seeds", nargs="+", default=DEFAULT_SEEDS, type=int)
    parser.add_argument("--mne-eps", default=1e-6, type=float)
    parser.add_argument("--group-lasso-eps", default=1e-12, type=float)
    parser.add_argument("--spectral-power-iters", default=3, type=int)
    parser.add_argument("--spectral-mne-power-iters", default=3, type=int)
    parser.add_argument("--spectral-mne-layer-reduce", default="sum", choices=["sum", "mean"])
    parser.add_argument("--reg-warmup-epochs", default=0, type=int)
    parser.add_argument("--if-mode", default="rate_uniform")
    parser.add_argument("--spike-schedule", default="normal")
    parser.add_argument(
        "--first-layer-noise-position",
        default="post_input_if",
        choices=["post_input_if", "pre_input_if", "input_image"],
    )
    parser.add_argument("--first-layer-noise-type", default="gaussian", choices=["gaussian", "pink"])
    parser.add_argument("--noise-sigma-start", default=0.0, type=float)
    parser.add_argument("--noise-sigma-end", default=1.0, type=float)
    parser.add_argument("--noise-sigma-step", default=1.0, type=float)
    parser.add_argument("--ckpt-save-mode", default="best", choices=["best", "last"])
    parser.add_argument(
        "--out-root",
        default="../important_results/fashion_spectral_mne_ablation",
        type=Path,
    )
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--retest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train missing checkpoints without running the legacy input-noise sweep.",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    args.spectral_mne_rc_map = _parse_rc_map(args.spectral_mne_rc_map)
    return args


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    specs = _method_specs(args)

    raw_rows = []
    jobs = []
    for model in args.models:
        for method in args.methods:
            spec = specs[method]
            for rc in _rcs_for_model(spec, model):
                for seed in args.seeds:
                    jobs.append((model, method, spec, rc, seed))
    print(f"[INFO] expanded {len(jobs)} jobs", flush=True)

    for model, method, spec, rc, seed in jobs:
        print(
            f"[JOB] model={model} method={method} rc={_fmt_float(rc)} seed={seed}",
            flush=True,
        )
        ckpt = train_one(model, method, spec, rc, seed, args)
        if args.train_only:
            continue
        matrix, sigma_to_acc = test_one(model, method, rc, seed, ckpt, args)
        for sigma, acc in sigma_to_acc.items():
            raw_rows.append(
                {
                    "dataset": args.dataset,
                    "model": model,
                    "method": method,
                    "method_label": spec["label"],
                    "regularizer": spec["regularizer"],
                    "reg_coeff": _fmt_float(rc),
                    "weight_decay": spec["wd"],
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

    if args.dry_run or args.train_only:
        mode = "DRY RUN" if args.dry_run else "TRAIN ONLY"
        print(f"[{mode}] no evaluation CSV written", flush=True)
        return

    dataset_tag = "fashion" if args.dataset == "fashion_mnist" else args.dataset
    raw_path = args.out_root / f"{dataset_tag}_spectral_mne_raw.csv"
    mean_path = args.out_root / f"{dataset_tag}_spectral_mne_mean_std.csv"
    summary_path = args.out_root / f"{dataset_tag}_spectral_mne_summary.csv"
    raw_fields = [
        "dataset",
        "model",
        "method",
        "method_label",
        "regularizer",
        "reg_coeff",
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
            "model",
            "method",
            "method_label",
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
            "model",
            "method",
            "method_label",
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
