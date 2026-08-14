#!/usr/bin/env python3
"""
Fashion-MNIST cnn2_c8_c16: four-method post-IF Gaussian noise sweep.

Methods match the CIFAR-10 VGG16 σ∈[0,3] figure:
  weight_decay              L2-all (optimizer WD on all params)
  weight_decay_weights_only L2-wo (Conv/Linear weights only)
  old_detach                Old MNE (detach λ, rc=1e-4)
  calibrated_mne_a0p1       Calibrated MNE (α=0.1, rc=5e-4)

Reuses existing 30-epoch seed-42 checkpoints when present; trains missing ones.
Default sweep is σ=0, 0.25, ..., 3.0 so the high-noise 1–3 range can be plotted
together with clean accuracy.
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


def _method_specs(args) -> dict:
    return {
        "weight_decay": {
            "regularizer": "weight_decay",
            "label": "L2-all (all params)",
            "weight_decay": args.weight_decay,
            "reg_coeff": None,
            "train_args": [],
            "existing_suffixes": [
                f"fashion_spectral_mne_{args.model}_l2_rcnone_seed{{seed}}_ep{{epochs}}_L{{L}}_trainT0",
            ],
        },
        "weight_decay_weights_only": {
            "regularizer": "weight_decay_weights_only",
            "label": "L2-wo (weights only)",
            "weight_decay": args.weight_decay,
            "reg_coeff": None,
            "train_args": [],
            "existing_suffixes": [
                f"fashion_calibrated_mne_{args.model}_l2_wo_seed{{seed}}_ep{{epochs}}_L{{L}}",
                f"fashion_spectral_mne_{args.model}_wd_weight_only_rcnone_seed{{seed}}_ep{{epochs}}_L{{L}}_trainT0",
            ],
        },
        "old_detach": {
            "regularizer": "mne_l2",
            "label": r"Old MNE (detach λ)",
            "weight_decay": 0.0,
            "reg_coeff": args.mne_rc,
            "train_args": ["--mne_detach_lambda"],
            "existing_suffixes": [
                f"fashion_spectral_mne_{args.model}_old_detach_rc{_fmt_float(args.mne_rc)}"
                f"_seed{{seed}}_ep{{epochs}}_L{{L}}_trainT0",
            ],
        },
        "calibrated_mne_a0p1": {
            "regularizer": "calibrated_mne_l2",
            "label": r"Calibrated MNE (α=0.1)",
            "weight_decay": 0.0,
            "reg_coeff": args.calibrated_mne_rc,
            "train_args": [
                "--calibrated_mne_alpha",
                "0.1",
                "--calibrated_mne_risk_min",
                str(args.risk_min),
                "--calibrated_mne_risk_max",
                str(args.risk_max),
                "--calibrated_mne_alpha_start_epoch",
                str(args.alpha_start_epoch),
                "--calibrated_mne_alpha_warmup_epochs",
                str(args.alpha_warmup_epochs),
            ],
            "existing_suffixes": [
                f"fashion_calibrated_mne_{args.model}_cmne_a0p1_seed{{seed}}_ep{{epochs}}_L{{L}}",
            ],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="cnn2_c8_c16")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="If set, run these architectures instead of --model.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "weight_decay",
            "weight_decay_weights_only",
            "old_detach",
            "calibrated_mne_a0p1",
        ],
        choices=[
            "weight_decay",
            "weight_decay_weights_only",
            "old_detach",
            "calibrated_mne_a0p1",
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--train-T", type=int, default=0)
    parser.add_argument("--test-T", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--mne-rc", type=float, default=1e-4)
    parser.add_argument("--calibrated-mne-rc", type=float, default=5e-4)
    parser.add_argument("--risk-min", type=float, default=0.5)
    parser.add_argument("--risk-max", type=float, default=2.0)
    parser.add_argument("--alpha-start-epoch", type=int, default=5)
    parser.add_argument("--alpha-warmup-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--if-mode", default="rate_uniform")
    parser.add_argument("--spike-schedule", default="normal")
    parser.add_argument(
        "--first-layer-noise-position",
        default="post_input_if",
        choices=["post_input_if", "pre_input_if", "input_image"],
    )
    parser.add_argument("--first-layer-noise-type", default="gaussian")
    parser.add_argument("--noise-sigma-start", type=float, default=0.0)
    parser.add_argument("--noise-sigma-end", type=float, default=3.0)
    parser.add_argument("--noise-sigma-step", type=float, default=0.25)
    parser.add_argument("--ckpt-save-mode", default="best", choices=["best", "last"])
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--retest", action="store_true")
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Skip training and fail if no checkpoint is found.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent
        / "important_results"
        / "fashion_cnn2_widths_four_regs_sigma0_3_seed42",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    args.models = args.models or [args.model]
    return args


def _ckpt_path(suffix: str, args) -> Path:
    return ROOT / "fashion_mnist-checkpoints" / f"{args.model}_L[{args.L}]_{suffix}.pth"


def _train_suffix(method: str, seed: int, args) -> str:
    return (
        f"fashion_four_regs_{args.model}_{method}_seed{seed}"
        f"_ep{args.epochs}_L{args.L}"
    )


def _fill_suffix(template: str, seed: int, args) -> str:
    return template.format(seed=seed, epochs=args.epochs, L=args.L)


def resolve_checkpoint(method: str, spec: dict, seed: int, args) -> tuple[Path, str]:
    if not args.retrain:
        for template in spec["existing_suffixes"]:
            suffix = _fill_suffix(template, seed, args)
            ckpt = _ckpt_path(suffix, args)
            if ckpt.exists():
                return ckpt, suffix
        train_suffix = _train_suffix(method, seed, args)
        train_ckpt = _ckpt_path(train_suffix, args)
        if train_ckpt.exists():
            return train_ckpt, train_suffix
    suffix = _train_suffix(method, seed, args)
    return _ckpt_path(suffix, args), suffix


def _run(cmd: list[str], dry_run: bool) -> None:
    print("[CMD]", " ".join(str(x) for x in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def train_one(method: str, spec: dict, seed: int, args) -> tuple[Path, str]:
    ckpt, suffix = resolve_checkpoint(method, spec, seed, args)
    if ckpt.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {ckpt}", flush=True)
        return ckpt, suffix
    if args.test_only:
        raise FileNotFoundError(f"--test-only but checkpoint missing: {ckpt}")
    if args.dry_run:
        print(f"[DRY RUN TRAIN] {ckpt}", flush=True)
        return ckpt, suffix

    cmd = [
        sys.executable,
        str(ROOT / "main_train.py"),
        "-data",
        "fashion_mnist",
        "-arch",
        args.model,
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
        str(spec["weight_decay"]),
        "--ckpt-save-mode",
        args.ckpt_save_mode,
        "-suffix",
        suffix,
    ]
    if spec["reg_coeff"] is not None:
        cmd += ["--reg_coeff", str(spec["reg_coeff"])]
    cmd += list(spec["train_args"])
    _run(cmd, dry_run=False)
    return ckpt, suffix


def _matrix_path(noise_dir: Path, seed: int, args) -> Path:
    return noise_dir / (
        f"noise_sweep_matrix_fashion_mnist_{args.model}_T{args.test_T}"
        f"_mode_{args.if_mode}_schedule_{args.spike_schedule}_seed_{seed}.csv"
    )


def _read_matrix(path: Path, level: int) -> dict[float, float]:
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                row_l = int(float(row.get("L", "")))
            except (TypeError, ValueError):
                continue
            if row_l != int(level):
                continue
            values = {}
            for key, value in row.items():
                if key in ("L", "T") or value in (None, ""):
                    continue
                try:
                    values[float(key)] = float(value)
                except ValueError:
                    continue
            return dict(sorted(values.items()))
    raise ValueError(f"No L={level} row in {path}")


def test_one(method: str, suffix: str, seed: int, ckpt: Path, args) -> tuple[Path, dict]:
    noise_dir = args.out_root / args.model / method / f"seed_{seed}"
    matrix = _matrix_path(noise_dir, seed, args)
    if matrix.exists() and not args.retest:
        print(f"[SKIP TEST] {matrix}", flush=True)
        return matrix, _read_matrix(matrix, args.L)

    cmd = [
        sys.executable,
        str(ROOT / "main_test.py"),
        "-data",
        "fashion_mnist",
        "-arch",
        args.model,
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


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(raw_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for row in raw_rows:
        key = (row["method"], row["method_label"], float(row["sigma"]))
        grouped[key].append(float(row["acc"]))

    mean_rows = []
    for (method, label, sigma), vals in sorted(grouped.items()):
        mean_rows.append(
            {
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

    by_method = defaultdict(dict)
    for row in mean_rows:
        by_method[(row["method"], row["method_label"])][float(row["sigma"])] = row

    summary_rows = []
    for (method, label), sigma_map in sorted(by_method.items()):
        sigmas = sorted(sigma_map)
        clean = float(sigma_map[sigmas[0]]["acc_mean"])
        end = float(sigma_map[sigmas[-1]]["acc_mean"])
        summary_rows.append(
            {
                "method": method,
                "method_label": label,
                "clean_sigma": f"{sigmas[0]:.6g}",
                "clean_acc_mean": f"{clean:.6f}",
                "end_sigma": f"{sigmas[-1]:.6g}",
                "end_acc_mean": f"{end:.6f}",
                "absolute_drop": f"{clean - end:.6f}",
                "relative_retention": f"{end / clean:.6f}" if clean > 0 else "nan",
            }
        )
    return mean_rows, summary_rows


def _run_one_model(model: str, args) -> list[dict]:
    args.model = model
    specs = _method_specs(args)
    raw_rows = []
    for seed in args.seeds:
        for method in args.methods:
            spec = specs[method]
            print(
                f"[JOB] model={args.model} method={method} "
                f"({spec['label']}) seed={seed}",
                flush=True,
            )
            ckpt, suffix = train_one(method, spec, seed, args)
            matrix, values = test_one(method, suffix, seed, ckpt, args)
            for sigma, acc in values.items():
                raw_rows.append(
                    {
                        "dataset": "fashion_mnist",
                        "arch": args.model,
                        "method": method,
                        "method_label": spec["label"],
                        "regularizer": spec["regularizer"],
                        "reg_coeff": _fmt_float(spec["reg_coeff"]),
                        "weight_decay": spec["weight_decay"],
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
    return raw_rows


RAW_FIELDS = [
    "dataset",
    "arch",
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


def _write_outputs(raw_rows: list[dict], out_dir: Path, model: str, args) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "fashion_four_regs_raw.csv"
    mean_path = out_dir / "fashion_four_regs_mean_std.csv"
    summary_path = out_dir / "fashion_four_regs_summary.csv"
    mean_rows, summary_rows = _aggregate(raw_rows)
    _write_csv(raw_path, raw_rows, RAW_FIELDS)
    _write_csv(
        mean_path,
        mean_rows,
        ["method", "method_label", "sigma", "n", "acc_mean", "acc_std"],
    )
    _write_csv(
        summary_path,
        summary_rows,
        [
            "method",
            "method_label",
            "clean_sigma",
            "clean_acc_mean",
            "end_sigma",
            "end_acc_mean",
            "absolute_drop",
            "relative_retention",
        ],
    )
    print(f"[DONE] raw: {raw_path}", flush=True)
    print(f"[DONE] mean/std: {mean_path}", flush=True)
    print(f"[DONE] summary: {summary_path}", flush=True)

    plot_script = Path(__file__).with_name("plot_fashion_four_regs_noise_sweep.py")
    plot_out = out_dir / f"fashion_{model}_four_regs_sigma0_3_seed42.png"
    title = rf"Fashion-MNIST {model} seed42 · post-IF noise $\sigma \in [0, 3]$"
    if plot_script.exists() and mean_rows:
        _run(
            [
                sys.executable,
                str(plot_script),
                "--csv",
                str(mean_path),
                "--out",
                str(plot_out),
                "--title",
                title,
            ],
            dry_run=False,
        )


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for model in args.models:
        model_rows = _run_one_model(model, args)
        all_rows.extend(model_rows)
        if args.dry_run:
            continue
        model_dir = args.out_root / model if len(args.models) > 1 else args.out_root
        _write_outputs(model_rows, model_dir, model, args)

    if args.dry_run:
        print("[DRY RUN] no CSV written", flush=True)
        return

    if len(args.models) > 1 and all_rows:
        _write_csv(args.out_root / "fashion_four_regs_raw.csv", all_rows, RAW_FIELDS)
        plot_script = Path(__file__).with_name("plot_fashion_four_regs_noise_sweep.py")
        panel_out = args.out_root / "fashion_cnn2_widths_four_regs_sigma0_3_seed42.png"
        if plot_script.exists():
            _run(
                [
                    sys.executable,
                    str(plot_script),
                    "--csv",
                    str(args.out_root / "fashion_four_regs_raw.csv"),
                    "--out",
                    str(panel_out),
                    "--panel-by-arch",
                    "--title",
                    r"Fashion-MNIST cnn2 widths seed42 · post-IF $\sigma \in [0, 3]$",
                ],
                dry_run=False,
            )


if __name__ == "__main__":
    main()
