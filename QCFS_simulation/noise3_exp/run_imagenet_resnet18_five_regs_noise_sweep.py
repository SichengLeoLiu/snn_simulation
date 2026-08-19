#!/usr/bin/env python3
"""ImageNet ResNet-18 seed42：CIFAR 五方法噪声扫描。

Methods:
  weight_decay              L2-all (optimizer WD=1e-4, ImageNet default)
  weight_decay_weights_only L2-wo (Conv/Linear weights only, WD=1e-4)
  l1                        L1-wo (rc=1e-5)
  old_detach                Old MNE (detach λ, rc=1e-4)
  calibrated_mne_a0p1       Calibrated MNE (α=0.1, rc=1e-4)

Train: ANN T=0, 90 epochs. H200 defaults: batch=256, lr=0.1 (linear-scaled from V100 128/0.05).
Test: T=16, rate_uniform, σ=0…5 step 0.25.
Noise: post_input_if (conv1 IF 后) or pre_input_if / pre_first_conv (conv1 前，像素).
One --method per PBS job so five GPUs can run in parallel.
Reuse checkpoints with --test-only and a new --out-root.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Preprocess.imagenet_hf_env import configure_imagenet_hf_env  # noqa: E402

configure_imagenet_hf_env(verbose=True)

ARCH = "resnet18"
DATASET = "imagenet"
SEED = 42
LVAL = 16
TRAIN_T = 0
TEST_T = 16
IF_MODE = "rate_uniform"

METHOD_ORDER = [
    "weight_decay",
    "weight_decay_weights_only",
    "l1",
    "old_detach",
    "calibrated_mne_a0p1",
]


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
            "legacy_names": [
                f"{ARCH}_L[{LVAL}]_seed{SEED}_schemeC_noout_wd_l{LVAL}_{ARCH}.pth",
            ],
        },
        "weight_decay_weights_only": {
            "regularizer": "weight_decay_weights_only",
            "label": "L2-wo (weights only)",
            "weight_decay": args.weight_decay,
            "reg_coeff": None,
            "train_args": [],
            "legacy_names": [],
        },
        "l1": {
            "regularizer": "l1",
            "label": r"L1-wo (weights only, $10^{-5}$)",
            "weight_decay": 0.0,
            "reg_coeff": args.l1_rc,
            "train_args": [],
            "legacy_names": [],
        },
        "old_detach": {
            "regularizer": "mne_l2",
            "label": r"Old MNE (detach $\lambda$)",
            "weight_decay": 0.0,
            "reg_coeff": args.mne_rc,
            "train_args": ["--mne_detach_lambda"],
            "legacy_names": [
                f"{ARCH}_L[{LVAL}]_seed{SEED}_schemeC_noout_mne_l2_l{LVAL}_{ARCH}"
                f"_rc{_fmt_float(args.mne_rc).replace('0p0001', '1em04')}.pth",
                f"{ARCH}_L[{LVAL}]_seed{SEED}_schemeC_noout_mne_l2_l{LVAL}_{ARCH}_rc1em04.pth",
            ],
        },
        "calibrated_mne_a0p1": {
            "regularizer": "calibrated_mne_l2",
            "label": r"Calibrated MNE ($\alpha = 0.1$)",
            "weight_decay": 0.0,
            "reg_coeff": args.calibrated_mne_rc,
            "train_args": [
                "--calibrated_mne_alpha",
                "0.1",
                "--calibrated_mne_risk_min",
                "0.5",
                "--calibrated_mne_risk_max",
                "2.0",
                "--calibrated_mne_alpha_start_epoch",
                str(args.alpha_start_epoch),
                "--calibrated_mne_alpha_warmup_epochs",
                str(args.alpha_warmup_epochs),
            ],
            "legacy_names": [],
        },
    }


def _ckpt_dir() -> Path:
    return ROOT / f"{DATASET}-checkpoints"


def _train_suffix(method: str, spec: dict, args) -> str:
    rc = spec["reg_coeff"]
    return (
        f"mneablate_{DATASET}_{method}"
        f"_rc{_fmt_float(rc)}_seed{SEED}_L{args.L}_trainT{TRAIN_T}"
    )


def _ckpt_path(suffix: str, args) -> Path:
    return _ckpt_dir() / f"{args.arch}_L[{args.L}]_{suffix}.pth"


def resolve_checkpoint(method: str, spec: dict, args) -> tuple[Path, str]:
    suffix = _train_suffix(method, spec, args)
    primary = _ckpt_path(suffix, args)
    if primary.exists() and not args.retrain:
        return primary, suffix
    if not args.retrain:
        for name in spec.get("legacy_names", []):
            legacy = _ckpt_dir() / name
            if legacy.exists():
                print(f"[CKPT FALLBACK] {legacy}", flush=True)
                return legacy, legacy.stem.replace(f"{args.arch}_L[{args.L}]_", "", 1)
    return primary, suffix


def _run(cmd: list, *, dry_run: bool) -> None:
    print("[CMD]", " ".join(str(x) for x in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def train_one(method: str, spec: dict, args) -> tuple[Path, str]:
    ckpt, suffix = resolve_checkpoint(method, spec, args)
    if ckpt.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {ckpt}", flush=True)
        return ckpt, suffix
    if args.test_only:
        raise FileNotFoundError(f"--test-only but checkpoint missing: {ckpt}")

    cmd = [
        sys.executable,
        str(ROOT / "main_train.py"),
        "-data",
        DATASET,
        "-arch",
        args.arch,
        "-L",
        str(args.L),
        "-T",
        str(TRAIN_T),
        "--epochs",
        str(args.epochs),
        "-lr",
        str(args.lr),
        "-b",
        str(args.batch_size),
        "-j",
        str(args.workers),
        "--seed",
        str(SEED),
        "--device",
        "auto",
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
    print(f"[TRAIN] {method} epochs={args.epochs} batch={args.batch_size}", flush=True)
    _run(cmd, dry_run=args.dry_run)
    if args.dry_run:
        return ckpt, suffix
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing after train: {ckpt}")
    print(f"[TRAIN DONE] {ckpt.name}", flush=True)
    return ckpt, suffix


def _matrix_path(noise_dir: Path, args) -> Path:
    return noise_dir / (
        f"noise_sweep_matrix_{DATASET}_{args.arch}_T{args.test_T}"
        f"_mode_{args.if_mode}_schedule_{args.spike_schedule}_seed_{SEED}.csv"
    )


def _read_matrix(path: Path, level: int) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
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


def test_one(method: str, suffix: str, ckpt: Path, args) -> tuple[Path, dict]:
    noise_dir = args.out_root / method
    noise_dir.mkdir(parents=True, exist_ok=True)
    matrix = _matrix_path(noise_dir, args)
    if matrix.exists() and not args.retest:
        print(f"[SKIP TEST] {matrix}", flush=True)
        return matrix, _read_matrix(matrix, args.L)

    cmd = [
        sys.executable,
        str(ROOT / "main_test.py"),
        "-data",
        DATASET,
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
        "gaussian",
        "-w",
        str(ckpt),
        "-suffix",
        suffix,
        "-b",
        str(args.batch_size),
        "-j",
        str(args.workers),
        "--seed",
        str(SEED),
        "--device",
        "auto",
    ]
    print(f"[TEST] {method} σ={args.noise_sigma_start}…{args.noise_sigma_end}", flush=True)
    _run(cmd, dry_run=args.dry_run)
    if args.dry_run:
        return matrix, {}
    if not matrix.exists():
        raise FileNotFoundError(f"matrix missing: {matrix}")
    print(f"[TEST DONE] {matrix.name}", flush=True)
    return matrix, _read_matrix(matrix, args.L)


def _write_method_rows(method: str, spec: dict, ckpt: Path, matrix: Path, curve: dict, args) -> None:
    rows = []
    for sigma, acc in curve.items():
        rows.append(
            {
                "dataset": DATASET,
                "arch": args.arch,
                "variant": method,
                "variant_label": spec["label"],
                "regularizer": spec["regularizer"],
                "reg_coeff": _fmt_float(spec["reg_coeff"]),
                "weight_decay": spec["weight_decay"],
                "seed": SEED,
                "L": args.L,
                "train_T": TRAIN_T,
                "test_T": args.test_T,
                "sigma": f"{sigma:.6g}",
                "acc": f"{acc:.6f}",
                "checkpoint": _rel_path(ckpt),
                "matrix_csv": _rel_path(matrix),
            }
        )
    path = args.out_root / method / "rows.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[TABLE] {path} n={len(rows)}", flush=True)


def aggregate(args, specs: dict) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for method in METHOD_ORDER:
        path = args.out_root / method / "rows.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                grouped[method].append(row)

    raw_rows = []
    mean_rows = []
    summary_rows = []
    for method in METHOD_ORDER:
        rows = grouped.get(method, [])
        if not rows:
            continue
        spec = specs[method]
        raw_rows.extend(rows)
        by_sigma = defaultdict(list)
        for row in rows:
            by_sigma[float(row["sigma"])].append(float(row["acc"]))
        acc_at = {}
        for sigma, vals in sorted(by_sigma.items()):
            mean = sum(vals) / len(vals)
            acc_at[sigma] = mean
            mean_rows.append(
                {
                    "dataset": DATASET,
                    "variant": method,
                    "variant_label": spec["label"],
                    "regularizer": spec["regularizer"],
                    "reg_coeff": _fmt_float(spec["reg_coeff"]),
                    "sigma": f"{sigma:.6g}",
                    "n": len(vals),
                    "acc_mean": f"{mean:.6f}",
                    "acc_std": "0.000000",
                }
            )
        sigmas = sorted(acc_at)
        clean = acc_at[sigmas[0]]
        end = acc_at[sigmas[-1]]
        summary_rows.append(
            {
                "dataset": DATASET,
                "variant": method,
                "variant_label": spec["label"],
                "clean_acc_mean": f"{clean:.6f}",
                "end_sigma": f"{sigmas[-1]:.6g}",
                "end_acc_mean": f"{end:.6f}",
                "absolute_drop": f"{clean - end:.6f}",
            }
        )

    def _dump(name: str, rows: list[dict]) -> None:
        if not rows:
            print(f"[AGG] skip empty {name}", flush=True)
            return
        path = args.out_root / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[AGG] {path} n={len(rows)}", flush=True)

    _dump("mne_stability_ablation_raw.csv", raw_rows)
    _dump("mne_stability_ablation_mean_std.csv", mean_rows)
    _dump("mne_stability_ablation_summary.csv", summary_rows)
    _plot(mean_rows, args)


def _plot(mean_rows: list[dict], args) -> None:
    if not mean_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "weight_decay": dict(label="L2-all (all params)", color="#0072B2", ls="--"),
        "weight_decay_weights_only": dict(label="L2-wo (weights only)", color="#009E73", ls="-"),
        "l1": dict(label=r"L1-wo (weights only, $10^{-5}$)", color="#D55E00", ls="-"),
        "old_detach": dict(label=r"Old MNE (detach $\lambda$)", color="#E69F00", ls="-"),
        "calibrated_mne_a0p1": dict(label=r"Calibrated MNE ($\alpha = 0.1$)", color="#6A3D9A", ls="-"),
    }
    grouped = defaultdict(list)
    for row in mean_rows:
        grouped[row["variant"]].append(row)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.8, 5.8), dpi=220)
    for method in METHOD_ORDER:
        rows = sorted(grouped.get(method, []), key=lambda r: float(r["sigma"]))
        if not rows:
            continue
        st = styles[method]
        ax.plot(
            [float(r["sigma"]) for r in rows],
            [float(r["acc_mean"]) for r in rows],
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=st["color"],
            linestyle=st["ls"],
            label=st["label"],
        )
    ax.set_xlabel(r"Gaussian noise $\sigma$")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_xlim(-0.1, 5.1)
    ax.set_xticks(list(range(6)))
    ax.legend(loc="best", frameon=True)
    pos = str(args.first_layer_noise_position)
    pos_title = {
        "post_input_if": r"post-IF",
        "pre_input_if": r"pre-conv1",
        "pre_first_conv": r"pre-conv1",
        "input_image": r"input-image",
    }.get(pos, pos)
    ax.set_title(rf"ImageNet ResNet-18 seed42 · five regs · {pos_title} $\sigma \in [0, 5]$")
    fig.tight_layout()
    out = args.out_root / f"imagenet_resnet18_five_regs_{pos}_sigma0_5_seed42.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {out}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=False,
        choices=METHOD_ORDER + ["all"],
        help="Run one method, or all sequentially.",
    )
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--arch", default=ARCH)
    parser.add_argument("--L", default=LVAL, type=int)
    parser.add_argument("--epochs", default=int(os.environ.get("IMAGENET_EPOCHS", "90")), type=int)
    parser.add_argument("--lr", default=float(os.environ.get("IMAGENET_LR", "0.1")), type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--mne-rc", default=1e-4, type=float)
    parser.add_argument("--calibrated-mne-rc", default=1e-4, type=float)
    parser.add_argument("--l1-rc", default=1e-5, type=float)
    parser.add_argument("--alpha-start-epoch", default=9, type=int)
    parser.add_argument("--alpha-warmup-epochs", default=15, type=int)
    parser.add_argument("--batch-size", default=int(os.environ.get("IMAGENET_BATCH", "256")), type=int)
    parser.add_argument("--workers", default=int(os.environ.get("IMAGENET_NUM_WORKERS", "8")), type=int)
    parser.add_argument("--test-T", dest="test_T", default=TEST_T, type=int)
    parser.add_argument("--if-mode", default=IF_MODE)
    parser.add_argument("--spike-schedule", default="normal")
    parser.add_argument(
        "--first-layer-noise-position",
        default="post_input_if",
        choices=["post_input_if", "pre_input_if", "pre_first_conv", "input_image"],
        help=(
            "post_input_if=conv1(含IF)后、maxpool前；"
            "pre_input_if/pre_first_conv=展开T后、conv1前（像素）；"
            "input_image=原始图像上加噪后再展开T"
        ),
    )
    parser.add_argument("--noise-sigma-start", default=0.0, type=float)
    parser.add_argument("--noise-sigma-end", default=5.0, type=float)
    parser.add_argument("--noise-sigma-step", default=0.25, type=float)
    parser.add_argument("--ckpt-save-mode", default="best", choices=["best", "last"])
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--retest", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "imagenet_resnet18_five_regs_sigma0_5_seed42",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    if not args.aggregate and not args.method:
        parser.error("specify --method or --aggregate")
    return args


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    specs = _method_specs(args)
    methods = METHOD_ORDER if args.method == "all" else ([args.method] if args.method else [])
    for method in methods:
        spec = specs[method]
        print(f"\n=== {method} ({spec['label']}) ===", flush=True)
        ckpt, suffix = train_one(method, spec, args)
        matrix, curve = test_one(method, suffix, ckpt, args)
        if curve:
            _write_method_rows(method, spec, ckpt, matrix, curve, args)
    aggregate(args, specs)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
