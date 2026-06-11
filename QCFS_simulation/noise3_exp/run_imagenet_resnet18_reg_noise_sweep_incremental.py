"""
ImageNet ResNet-18：按正则方法单独训练 + 噪声注入，结果增量写入同一表格并出折线图。

每次运行只处理一种正则（--method），可多次提交 PBS，结果累积到同一 CSV/图中。

用法示例：
  python noise3_exp/run_imagenet_resnet18_reg_noise_sweep_incremental.py --method mne_l2
  python noise3_exp/run_imagenet_resnet18_reg_noise_sweep_incremental.py --method weight_decay
  python noise3_exp/run_imagenet_resnet18_reg_noise_sweep_incremental.py --method mne_l2_wd

  # 仅从已有表格重绘折线图
  python noise3_exp/run_imagenet_resnet18_reg_noise_sweep_incremental.py --plot-only

环境变量（可选）：
  IMAGENET_EPOCHS  默认 90
  IMAGENET_BATCH   默认 128（训练与噪声扫描共用）
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Preprocess.imagenet_hf_env import configure_imagenet_hf_env  # noqa: E402

configure_imagenet_hf_env(verbose=True)

OUT = (
    ROOT
    / "noise3_exp"
    / "imagenet_resnet18_mne_l2_wd_combo_noise_sweep_rate_uniform_L16_T16"
)
OUT.mkdir(parents=True, exist_ok=True)

ARCH = "resnet18"
DATASET = "imagenet"
SEED = 42
LVAL = 16
TVAL = 16
IF_MODE = "rate_uniform"
EPOCHS = int(os.environ.get("IMAGENET_EPOCHS", "90"))
LR = 0.05
BATCH = int(os.environ.get("IMAGENET_BATCH", "128"))
SCHEME_TAG = "schemeC_noout"

MNE_RC = 1e-4
WD_BASE = 1e-4
WD_COMBO = 1e-4

METHOD_ALIASES = {
    "weight_decay": "weight_decay",
    "wd": "weight_decay",
    "mne_l2": "mne_l2",
    "mnel2": "mne_l2",
    "mne_l2_wd": "mne_l2_wd",
    "mne_l2+wd": "mne_l2_wd",
    "combo": "mne_l2_wd",
}

METHOD_CONFIG = {
    "weight_decay": ("weight_decay", None, WD_BASE),
    "mne_l2": ("mne_l2 rc=1e-4", MNE_RC, 0.0),
    "mne_l2_wd": ("mne_l2+wd rc=1e-4 wd=1e-4", MNE_RC, WD_COMBO),
}

PLOT_ORDER = ["weight_decay", "mne_l2 rc=1e-4", "mne_l2+wd rc=1e-4 wd=1e-4"]

LINE_STYLES = {
    "weight_decay": {"color": "#ff7f0e", "label": "weight_decay (wd=1e-4)"},
    "mne_l2 rc=1e-4": {"color": "#1f77b4", "label": "mne_l2 (rc=1e-4)"},
    "mne_l2+wd rc=1e-4 wd=1e-4": {
        "color": "#2ca02c",
        "label": "mne_l2+wd (rc=1e-4, wd=1e-4)",
    },
}

RAW_CSV = OUT / "imagenet_resnet18_mne_l2_wd_combo_noise_sweep_raw.csv"
SUMMARY_CSV = OUT / "imagenet_resnet18_mne_l2_wd_combo_best_test_summary.csv"

RAW_FIELDS = [
    "label",
    "reg_coeff",
    "weight_decay",
    "seed",
    "L",
    "T",
    "if_mode",
    "sigma",
    "acc",
    "checkpoint",
    "matrix_csv",
]

SUMMARY_FIELDS = ["label", "reg_coeff", "weight_decay", "acc_sigma0", "checkpoint"]


def coeff_tag(v: float) -> str:
    return f"{v:.0e}".replace("-", "m").replace("+", "p")


def build_suffix(reg_coeff: Optional[float], wd: float) -> str:
    if reg_coeff is None:
        return f"seed{SEED}_{SCHEME_TAG}_wd_l{LVAL}_{ARCH}"
    if wd > 0:
        return (
            f"seed{SEED}_{SCHEME_TAG}_mne_l2_wd_l{LVAL}_{ARCH}"
            f"_rc{coeff_tag(reg_coeff)}_wd{coeff_tag(wd)}"
        )
    return f"seed{SEED}_{SCHEME_TAG}_mne_l2_l{LVAL}_{ARCH}_rc{coeff_tag(reg_coeff)}"


def ckpt_path(reg_coeff: Optional[float], wd: float) -> Path:
    suffix = build_suffix(reg_coeff, wd)
    return ROOT / f"{DATASET}-checkpoints" / f"{ARCH}_L[{LVAL}]_{suffix}.pth"


def resolve_method(name: str) -> str:
    key = METHOD_ALIASES.get(name.strip().lower())
    if key is None:
        choices = ", ".join(sorted(set(METHOD_ALIASES.values())))
        raise ValueError(f"未知 --method {name!r}，可选: {choices}")
    return key


def train_one(label: str, reg_coeff: Optional[float], wd: float) -> Path:
    ckpt = ckpt_path(reg_coeff, wd)
    if ckpt.exists():
        print(f"[SKIP TRAIN] {ckpt.name}", flush=True)
        return ckpt

    if reg_coeff is None:
        regularizer, rc = "weight_decay", 1.0
    else:
        regularizer, rc = "mne_l2", reg_coeff

    cmd = [
        sys.executable,
        str(ROOT / "main_train.py"),
        "-data",
        DATASET,
        "-arch",
        ARCH,
        "-L",
        str(LVAL),
        "--epochs",
        str(EPOCHS),
        "-lr",
        str(LR),
        "-j",
        "8",
        "-b",
        str(BATCH),
        "--seed",
        str(SEED),
        "--device",
        "auto",
        "--time",
        "0",
        "--spike_schedule",
        "normal",
        "--regularizer",
        regularizer,
        "--weight_decay",
        str(wd),
        "--reg_coeff",
        str(rc),
        "--suffix",
        build_suffix(reg_coeff, wd),
    ]
    if regularizer == "mne_l2":
        cmd.append("--mne_detach_lambda")

    print(f"[TRAIN] {label} epochs={EPOCHS} batch={BATCH}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
    print(f"[TRAIN DONE] {ckpt.name}", flush=True)
    return ckpt


def test_noise_sweep(label: str, ckpt: Path) -> Path:
    safe_label = label.replace(" ", "_").replace("=", "")
    out_dir = OUT / safe_label
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = (
        out_dir
        / f"noise_sweep_matrix_{DATASET}_{ARCH}_T{TVAL}_mode_{IF_MODE}_schedule_normal_seed_{SEED}.csv"
    )
    if matrix.exists():
        print(f"[SKIP TEST] {label}", flush=True)
        return matrix

    cmd = [
        sys.executable,
        str(ROOT / "main_test.py"),
        "-data",
        DATASET,
        "-arch",
        ARCH,
        "-L",
        str(LVAL),
        "-T",
        str(TVAL),
        "-j",
        "8",
        "-b",
        str(BATCH),
        "--seed",
        str(SEED),
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
        "0.1",
        "--noise_output_dir",
        str(out_dir),
    ]
    print(f"[TEST] {label} mode={IF_MODE}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    if not matrix.exists():
        cands = sorted(
            out_dir.glob(
                f"noise_sweep_matrix_*_T{TVAL}_mode_{IF_MODE}_schedule_normal_seed_{SEED}.csv"
            )
        )
        if not cands:
            raise FileNotFoundError(f"matrix missing: {out_dir}")
        matrix = cands[0]
    print(f"[TEST DONE] {matrix.name}", flush=True)
    return matrix


def read_matrix(mat: Path) -> list[tuple[float, float]]:
    with mat.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)
    start = 2 if header[:2] == ["L", "T"] else 1
    sigmas = [float(x) for x in header[start:]]
    accs = [float(x) for x in row[start:]]
    return list(zip(sigmas, accs))


def curve_to_raw_rows(
    label: str,
    reg_coeff: Optional[float],
    wd: float,
    ckpt: Path,
    matrix: Path,
    curve: list[tuple[float, float]],
) -> list[dict]:
    rc_val = "" if reg_coeff is None else reg_coeff
    rows = []
    for sigma, acc in curve:
        rows.append(
            {
                "label": label,
                "reg_coeff": rc_val,
                "weight_decay": wd,
                "seed": SEED,
                "L": LVAL,
                "T": TVAL,
                "if_mode": IF_MODE,
                "sigma": sigma,
                "acc": acc,
                "checkpoint": str(ckpt.relative_to(ROOT)),
                "matrix_csv": str(matrix.relative_to(ROOT)),
            }
        )
    return rows


def load_raw_rows() -> list[dict]:
    if not RAW_CSV.exists():
        return []
    with RAW_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def upsert_raw_rows(label: str, new_rows: list[dict]) -> None:
    kept = [r for r in load_raw_rows() if r["label"] != label]
    kept.extend(new_rows)
    kept.sort(key=lambda r: (PLOT_ORDER.index(r["label"]) if r["label"] in PLOT_ORDER else 99, float(r["sigma"])))
    with RAW_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(kept)
    print(f"[TABLE] updated {RAW_CSV} ({len(set(r['label'] for r in kept))} method(s))", flush=True)


def raw_rows_to_curves(rows: list[dict]) -> dict[str, list[tuple[float, float]]]:
    curves: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        label = row["label"]
        curves.setdefault(label, []).append((float(row["sigma"]), float(row["acc"])))
    for label in curves:
        curves[label] = sorted(curves[label], key=lambda x: x[0])
    return curves


def rebuild_summary(rows: list[dict]) -> None:
    curves = raw_rows_to_curves(rows)
    summary_rows = []
    for label in PLOT_ORDER:
        if label not in curves:
            continue
        curve = dict(curves[label])
        acc0 = curve.get(0.0, curve.get(0))
        sample = next(r for r in rows if r["label"] == label)
        rc = sample["reg_coeff"]
        wd = sample["weight_decay"]
        rc_str = f"{float(rc):.0e}" if rc != "" else ""
        wd_f = float(wd)
        wd_str = f"{wd_f:.0e}" if wd_f > 0 else "0"
        summary_rows.append(
            {
                "label": label,
                "reg_coeff": rc_str,
                "weight_decay": wd_str,
                "acc_sigma0": f"{acc0:.6f}",
                "checkpoint": sample["checkpoint"],
            }
        )

    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[TABLE] updated {SUMMARY_CSV}", flush=True)


def plot_results(curves: dict[str, list[tuple[float, float]]]) -> None:
    if not curves:
        print("[PLOT] 表格为空，跳过绘图", flush=True)
        return

    plt.rcParams.update({"font.size": 11, "legend.fontsize": 10})
    for no_caption in (False, True):
        fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=180)
        all_y = []
        for label in PLOT_ORDER:
            if label not in curves:
                continue
            pts = curves[label]
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            all_y.extend(y)
            style = LINE_STYLES.get(label, {"color": "#333333", "label": label})
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.2,
                markersize=5,
                color=style["color"],
                label=style["label"],
            )
        ax.set_xlabel("Gaussian noise sigma")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([round(i * 0.1, 1) for i in range(11)])
        if all_y:
            ax.set_ylim(min(all_y) - 1.0, max(all_y) + 1.0)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", frameon=False)
        if not no_caption:
            n_methods = len(curves)
            ax.set_title(
                f"ImageNet {ARCH} noise sweep ({n_methods} method(s), "
                f"seed={SEED}, L={LVAL}, T={TVAL}, {IF_MODE}, {SCHEME_TAG})"
            )
        fig.tight_layout()
        suffix = "_no_caption" if no_caption else ""
        out_png = OUT / f"imagenet_{ARCH}_mne_l2_wd_combo_noise_sweep{suffix}.png"
        fig.savefig(out_png)
        plt.close(fig)
        print(f"[PLOT] saved {out_png}", flush=True)


def print_summary(rows: list[dict]) -> None:
    curves = raw_rows_to_curves(rows)
    if not curves:
        return
    print("\n========== Combined table (sigma=0, rate_uniform) ==========", flush=True)
    print(f"{'label':<32} {'acc@sigma=0':>12}", flush=True)
    print("-" * 46, flush=True)
    for label in PLOT_ORDER:
        if label not in curves:
            continue
        acc0 = dict(curves[label]).get(0.0, dict(curves[label]).get(0))
        print(f"{label:<32} {acc0:>12.3f}", flush=True)
    print("-" * 46, flush=True)


def run_method(method_key: str) -> None:
    label, reg_coeff, wd = METHOD_CONFIG[method_key]
    print(f"\n=== Run method: {method_key} ({label}) ===", flush=True)

    ckpt = train_one(label, reg_coeff, wd)
    matrix = test_noise_sweep(label, ckpt)
    curve = read_matrix(matrix)

    new_rows = curve_to_raw_rows(label, reg_coeff, wd, ckpt, matrix, curve)
    upsert_raw_rows(label, new_rows)
    all_rows = load_raw_rows()
    rebuild_summary(all_rows)
    plot_results(raw_rows_to_curves(all_rows))
    print_summary(all_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ImageNet ResNet-18：单正则训练 + 噪声扫描，增量写入汇总表并出折线图"
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="正则方法: weight_decay | mne_l2 | mne_l2_wd（别名: wd, mnel2, combo）",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="不训练/测试，仅从已有汇总表重绘折线图",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.plot_only:
        rows = load_raw_rows()
        curves = raw_rows_to_curves(rows)
        plot_results(curves)
        print_summary(rows)
        return

    if args.method is None:
        raise SystemExit("请指定 --method，或使用 --plot-only 仅重绘图")

    method_key = resolve_method(args.method)
    run_method(method_key)


if __name__ == "__main__":
    main()
