#!/usr/bin/env python3
"""ImageNet ResNet-18 seed-42 FG-MNE-U screen (H200).

Same-map ResNet comparison. Coefficients locked to the ImageNet L2-wo budget:

    l2wo      reuse five-regs L2-wo checkpoint (WD=1e-4); eval only
    detach    train MNE-L2, resnet map, detach λ/γ, rc=1e-4
    nodetach  train MNE-L2, resnet map, grads into λ and BN γ
    fgmneu    train FG-MNE-U: no-detach MNE + unmatched-head L2, η_U=1e-4

Do not reuse the five-regs old_detach checkpoint in the main table: it used
the default legacy map. Do not retune rc or η_U from the test curve.
Checkpoints and logs go to scratch. Protocol: T=0 train, T=16 rate_uniform
post-IF Gaussian, seed 42. ImageNet noise_sweep reports Top-1 only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Preprocess.imagenet_hf_env import configure_imagenet_hf_env  # noqa: E402

configure_imagenet_hf_env(verbose=True)

from Models import modelpool  # noqa: E402
from utils import collect_weight_layer_matches, unmatched_weight_rows  # noqa: E402

ARCH = "resnet18"
DATASET = "imagenet"
LAYER_MAP = "resnet"
SEED = 42
LVAL = 16
TRAIN_T = 0
TEST_T = 16
IF_MODE = "rate_uniform"
MNE_RC = 1e-4
L2_WD = 1e-4
SCRATCH_ROOT = Path("/scratch/gs14/sl9144/snn_results/imagenet_resnet18_fgmneu_seed42")
L2WO_CKPT = Path(
    "/scratch/gs14/sl9144/snn_ckpts/imagenet-checkpoints/"
    f"{ARCH}_L[{LVAL}]_mneablate_{DATASET}_weight_decay_weights_only"
    f"_rcnone_seed{SEED}_L{LVAL}_trainT{TRAIN_T}.pth"
)
L2WO_MATRIX_CANDIDATES = [
    ROOT.parent
    / "important_results"
    / "imagenet_resnet18_five_regs_sigma0_5_seed42"
    / "weight_decay_weights_only"
    / (
        f"noise_sweep_matrix_{DATASET}_{ARCH}_T{TEST_T}"
        f"_mode_{IF_MODE}_schedule_normal_seed_{SEED}.csv"
    ),
    Path("/scratch/gs14/sl9144/snn_results/imagenet_resnet18_five_regs_sigma0_5_seed42")
    / "weight_decay_weights_only"
    / (
        f"noise_sweep_matrix_{DATASET}_{ARCH}_T{TEST_T}"
        f"_mode_{IF_MODE}_schedule_normal_seed_{SEED}.csv"
    ),
]
METHOD_ORDER = ["l2wo", "detach", "nodetach", "fgmneu"]
TRAIN_METHODS = ("detach", "nodetach", "fgmneu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHOD_ORDER + ["all"], default=None)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--L", default=LVAL, type=int)
    parser.add_argument("--epochs", default=int(os.environ.get("IMAGENET_EPOCHS", "90")), type=int)
    parser.add_argument("--lr", default=float(os.environ.get("IMAGENET_LR", "0.1")), type=float)
    parser.add_argument("--batch-size", default=int(os.environ.get("IMAGENET_BATCH", "256")), type=int)
    parser.add_argument("--workers", default=int(os.environ.get("IMAGENET_NUM_WORKERS", "8")), type=int)
    parser.add_argument("--test-T", dest="test_T", default=TEST_T, type=int)
    parser.add_argument("--if-mode", default=IF_MODE)
    parser.add_argument("--spike-schedule", default="normal")
    parser.add_argument("--first-layer-noise-position", default="post_input_if")
    parser.add_argument("--noise-sigma-start", default=0.0, type=float)
    parser.add_argument("--noise-sigma-end", default=5.0, type=float)
    parser.add_argument("--noise-sigma-step", default=0.25, type=float)
    parser.add_argument("--ckpt-save-mode", default="best")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--retest", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "imagenet_resnet18_fgmneu_seed42",
    )
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=Path(os.environ.get("IMAGENET_CKPT_ROOT", str(SCRATCH_ROOT / "ckpts"))),
    )
    args = parser.parse_args()
    args.arch = ARCH
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    if not args.ckpt_root.is_absolute():
        args.ckpt_root = (ROOT / args.ckpt_root).resolve()
    if args.self_check or args.aggregate:
        return args
    if args.method is None:
        parser.error("--method is required unless --self-check/--aggregate")
    return args


def method_specs() -> dict:
    return {
        "l2wo": {
            "label": "L2-wo",
            "regularizer": "weight_decay_weights_only",
            "weight_decay": L2_WD,
            "reg_coeff": None,
            "train": False,
            "train_args": [],
        },
        "detach": {
            "label": r"MNE-L2 detach, resnet map",
            "regularizer": "mne_l2",
            "weight_decay": 0.0,
            "reg_coeff": MNE_RC,
            "train": True,
            "train_args": ["--mne_detach_lambda", "--mne_layer_map", LAYER_MAP],
        },
        "nodetach": {
            "label": r"MNE-L2 no-detach, resnet map",
            "regularizer": "mne_l2",
            "weight_decay": 0.0,
            "reg_coeff": MNE_RC,
            "train": True,
            "train_args": ["--mne_no_detach_bn_affine", "--mne_layer_map", LAYER_MAP],
        },
        "fgmneu": {
            "label": r"FG-MNE-U (no-detach + unmatched L2)",
            "regularizer": "mne_l2_unmatched",
            "weight_decay": 0.0,
            "reg_coeff": MNE_RC,
            "train": True,
            "train_args": [
                "--mne_no_detach_bn_affine",
                "--mne_layer_map",
                LAYER_MAP,
                "--unmatched_l2_coeff",
                str(L2_WD),
                "--mne_unmatched_scope",
                "head",
            ],
        },
    }


def suffix(method: str) -> str:
    return f"img_{method}_resnetmap_seed{SEED}_L{LVAL}_trainT{TRAIN_T}"


def ckpt_path(args, method: str) -> Path:
    return args.ckpt_root / method / f"{ARCH}_L[{args.L}]_{suffix(method)}.pth"


def l2wo_reuse_ckpt() -> Path | None:
    candidates = [
        L2WO_CKPT,
        ROOT / "imagenet-checkpoints" / L2WO_CKPT.name,
        Path("/scratch/gs14/sl9144/snn_results/imagenet-checkpoints") / L2WO_CKPT.name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return L2WO_CKPT if L2WO_CKPT.is_file() else None


def _rel_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _fmt_float(x) -> str:
    if x is None:
        return "none"
    text = f"{float(x):.6g}"
    return text.replace("+", "").replace("-", "m").replace(".", "p")


def matrix_path(noise_dir: Path, args) -> Path:
    return noise_dir / (
        f"noise_sweep_matrix_{DATASET}_{args.arch}_T{args.test_T}"
        f"_mode_{args.if_mode}_schedule_{args.spike_schedule}_seed_{SEED}.csv"
    )


def read_matrix(path: Path, level: int) -> dict:
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


def train_cmd(args, method: str, spec: dict) -> list[str]:
    checkpoint = ckpt_path(args, method)
    out = args.out_root / method
    cmd = [
        sys.executable,
        str(ROOT / "main_train.py"),
        "-data",
        DATASET,
        "-arch",
        ARCH,
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
        args.device,
        "--spike_schedule",
        args.spike_schedule,
        "--regularizer",
        spec["regularizer"],
        "-wd",
        str(spec["weight_decay"]),
        "--ckpt-save-mode",
        args.ckpt_save_mode,
        "--ckpt-dir",
        str(checkpoint.parent),
        "-suffix",
        suffix(method),
        "--mapping_diag_dir",
        str(out / "mapping_init"),
        "--epoch_log_csv",
        str(out / "epoch_log.csv"),
    ]
    if spec["reg_coeff"] is not None:
        cmd += ["--reg_coeff", str(spec["reg_coeff"])]
    cmd += list(spec["train_args"])
    return cmd


def train_one(args, method: str, spec: dict) -> Path:
    if method == "l2wo":
        reused = l2wo_reuse_ckpt()
        if reused is None:
            raise FileNotFoundError(
                "L2-wo ImageNet checkpoint missing; will not retrain L2-wo. "
                f"Expected {L2WO_CKPT}"
            )
        print(f"[REUSE L2-wo] {reused}", flush=True)
        return reused
    checkpoint = ckpt_path(args, method)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {checkpoint}", flush=True)
        return checkpoint
    if args.test_only:
        raise FileNotFoundError(checkpoint)
    cmd = train_cmd(args, method, spec)
    print("[CMD]", " ".join(cmd), flush=True)
    if args.dry_run:
        return checkpoint
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def copy_l2wo_matrix(args) -> Path | None:
    dest = matrix_path(args.out_root / "l2wo", args)
    if dest.exists() and not args.retest:
        return dest
    for src in L2WO_MATRIX_CANDIDATES:
        if src.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())
            print(f"[REUSE L2-wo MATRIX] {src} -> {dest}", flush=True)
            return dest
    return None


def test_one(args, method: str, ckpt: Path):
    noise_dir = args.out_root / method
    noise_dir.mkdir(parents=True, exist_ok=True)
    matrix = matrix_path(noise_dir, args)
    if method == "l2wo" and (not args.retest):
        reused = copy_l2wo_matrix(args)
        if reused is not None:
            return reused, read_matrix(reused, args.L)
    if matrix.exists() and not args.retest:
        print(f"[SKIP TEST] {matrix}", flush=True)
        return matrix, read_matrix(matrix, args.L)
    cmd = [
        sys.executable,
        str(ROOT / "main_test.py"),
        "-data",
        DATASET,
        "-arch",
        ARCH,
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
        suffix(method) if method != "l2wo" else ckpt.stem.replace(f"{ARCH}_L[{args.L}]_", "", 1),
        "-b",
        str(args.batch_size),
        "-j",
        str(args.workers),
        "--seed",
        str(SEED),
        "--device",
        args.device,
    ]
    print("[CMD]", " ".join(cmd), flush=True)
    if args.dry_run:
        return matrix, {}
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not matrix.exists():
        raise FileNotFoundError(matrix)
    return matrix, read_matrix(matrix, args.L)


def write_method_rows(method: str, spec: dict, ckpt: Path, matrix: Path, curve: dict, args) -> None:
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
                "layer_map": LAYER_MAP if method != "l2wo" else "n/a",
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[TABLE] {path} n={len(rows)}", flush=True)


def auc(curve: dict, lo: float, hi: float) -> float:
    xs = [sigma for sigma in sorted(curve) if lo - 1e-12 <= sigma <= hi + 1e-12]
    if len(xs) < 2:
        return float("nan")
    area = 0.0
    for left, right in zip(xs[:-1], xs[1:]):
        area += 0.5 * (curve[left] + curve[right]) * (right - left)
    return float(area)


def write_scorecard(method: str, spec: dict, ckpt: Path, curve: dict, args) -> None:
    clean = curve.get(0.0, float("nan"))
    s5 = curve.get(5.0, float("nan"))
    card = {
        "method": method,
        "label": spec["label"],
        "regularizer": spec["regularizer"],
        "layer_map": LAYER_MAP if method != "l2wo" else "n/a",
        "eta_mne": None if method == "l2wo" else MNE_RC,
        "eta_u": L2_WD if method in ("l2wo", "fgmneu") else 0.0,
        "checkpoint": str(ckpt),
        "test_clean": clean,
        "test_sigma5": s5,
        "test_auc_full": auc(curve, 0.0, 5.0),
        "test_auc_high": auc(curve, 3.0, 5.0),
        "selection_uses_test": False,
        "metric": "top1",
        "note": "ImageNet noise_sweep reports Top-1 only; Top-5 is not in main_test.py",
    }
    path = args.out_root / method / "scorecard.json"
    path.write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2), flush=True)


def aggregate(args, specs: dict) -> None:
    raw_rows = []
    summary_rows = []
    curves = {}
    for method in METHOD_ORDER:
        path = args.out_root / method / "rows.csv"
        if not path.exists():
            continue
        spec = specs[method]
        curve = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                raw_rows.append(row)
                curve[float(row["sigma"])] = float(row["acc"])
        curves[method] = curve
        summary_rows.append(
            {
                "dataset": DATASET,
                "variant": method,
                "variant_label": spec["label"],
                "layer_map": LAYER_MAP if method != "l2wo" else "n/a",
                "clean_acc": f"{curve.get(0.0, float('nan')):.6f}",
                "sigma5_acc": f"{curve.get(5.0, float('nan')):.6f}",
                "auc_full": f"{auc(curve, 0.0, 5.0):.6f}",
                "auc_high": f"{auc(curve, 3.0, 5.0):.6f}",
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

    _dump("fgmneu_raw.csv", raw_rows)
    _dump("fgmneu_summary.csv", summary_rows)
    if not curves:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[PLOT] skip ({exc})", flush=True)
        return
    styles = {
        "l2wo": dict(label="L2-wo", color="#009E73", ls="-"),
        "detach": dict(label=r"MNE detach", color="#E69F00", ls="--"),
        "nodetach": dict(label=r"MNE no-detach", color="#6A3D9A", ls="-"),
        "fgmneu": dict(label=r"FG-MNE-U", color="#D55E00", ls="-"),
    }
    fig, ax = plt.subplots(figsize=(8.8, 5.8), dpi=220)
    for method in METHOD_ORDER:
        curve = curves.get(method)
        if not curve:
            continue
        st = styles[method]
        xs = sorted(curve)
        ax.plot(
            xs,
            [curve[x] for x in xs],
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
    ax.legend(loc="best", frameon=True)
    ax.set_title(r"ImageNet ResNet-18 seed42 · FG-MNE-U · resnet map · post-IF")
    fig.tight_layout()
    out = args.out_root / "imagenet_resnet18_fgmneu_seed42.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {out}", flush=True)


def self_check() -> None:
    model = modelpool(ARCH, DATASET)
    model._mne_layer_map = LAYER_MAP
    rows = collect_weight_layer_matches(model, layer_map=LAYER_MAP)
    unmatched = unmatched_weight_rows(model, "head", layer_map=LAYER_MAP)
    unmatched_all = unmatched_weight_rows(model, "all", layer_map=LAYER_MAP)
    names = [row["name"] for row in unmatched]
    all_names = [row["name"] for row in unmatched_all]
    n_matched = sum(1 for row in rows if row["matched"])
    print(
        f"ImageNet {ARCH} map={LAYER_MAP} matched={n_matched} "
        f"unmatched_head={names} unmatched_all={all_names}",
        flush=True,
    )
    if names != ["fc"] or all_names != ["fc"]:
        raise AssertionError(f"expected unmatched ['fc'], got head={names} all={all_names}")
    args = argparse.Namespace(
        L=LVAL,
        epochs=90,
        lr=0.1,
        batch_size=256,
        workers=8,
        device="cpu",
        spike_schedule="normal",
        ckpt_save_mode="best",
        out_root=ROOT.parent / "important_results" / "imagenet_resnet18_fgmneu_seed42",
        ckpt_root=SCRATCH_ROOT / "ckpts",
    )
    specs = method_specs()
    cmd = train_cmd(args, "fgmneu", specs["fgmneu"])
    assert cmd[cmd.index("--regularizer") + 1] == "mne_l2_unmatched"
    assert cmd[cmd.index("--mne_layer_map") + 1] == "resnet"
    assert cmd[cmd.index("--unmatched_l2_coeff") + 1] == str(L2_WD)
    assert "--mne_no_detach_bn_affine" in cmd
    assert "--mne_detach_lambda" not in cmd
    det = train_cmd(args, "detach", specs["detach"])
    assert "--mne_detach_lambda" in det
    assert "--mne_layer_map" in det
    nd = train_cmd(args, "nodetach", specs["nodetach"])
    assert "--mne_no_detach_bn_affine" in nd and "--mne_detach_lambda" not in nd
    assert specs["l2wo"]["train"] is False
    print("[self-check] ImageNet FG-MNE-U flags ok", flush=True)


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    specs = method_specs()
    if args.self_check:
        self_check()
        return
    if args.aggregate:
        aggregate(args, specs)
        return
    methods = METHOD_ORDER if args.method == "all" else [args.method]
    for method in methods:
        spec = specs[method]
        print(f"\n=== {method} ({spec['label']}) ===", flush=True)
        if method in TRAIN_METHODS:
            print(
                f"[INFO] train FG-MNE family method={method} map={LAYER_MAP} "
                f"η_MNE={MNE_RC} η_U={L2_WD if method == 'fgmneu' else 0.0}",
                flush=True,
            )
        ckpt = train_one(args, method, spec)
        matrix, curve = test_one(args, method, ckpt)
        if curve:
            write_method_rows(method, spec, ckpt, matrix, curve, args)
            write_scorecard(method, spec, ckpt, curve, args)
            (args.out_root / method / "scope.json").write_text(
                json.dumps(
                    {
                        "method": method,
                        "layer_map": LAYER_MAP if method != "l2wo" else "n/a",
                        "eta_mne": None if method == "l2wo" else MNE_RC,
                        "eta_u": L2_WD if method in ("l2wo", "fgmneu") else 0.0,
                        "checkpoint": str(ckpt),
                    },
                    indent=2,
                )
                + "\n"
            )
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
