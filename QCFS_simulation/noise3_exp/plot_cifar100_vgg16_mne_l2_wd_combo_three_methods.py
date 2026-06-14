"""
从 CIFAR-100 VGG16 mne_l2+wd combo 实验结果绘制三条曲线子集图。

保留：
  - weight_decay
  - mne_l2+wd rc=1e-4 wd=1e-4
  - mne_l2 rc=1e-4

用法：
  python noise3_exp/plot_cifar100_vgg16_mne_l2_wd_combo_three_methods.py
  python noise3_exp/plot_cifar100_vgg16_mne_l2_wd_combo_three_methods.py \\
      --raw-csv path/to/cifar100_vgg16_mne_l2_wd_combo_noise_sweep_raw.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = (
    ROOT
    / "noise3_exp"
    / "cifar100_vgg16_mne_l2_wd_combo_noise_sweep_rate_uniform_L16_T16"
)
DEFAULT_RAW = DEFAULT_OUT_DIR / "cifar100_vgg16_mne_l2_wd_combo_noise_sweep_raw.csv"

SELECTED_LABELS = [
    "weight_decay",
    "mne_l2+wd rc=1e-4 wd=1e-4",
    "mne_l2 rc=1e-4",
]

LINE_STYLES = {
    "weight_decay": {"color": "#ff7f0e", "label": "weight_decay"},
    "mne_l2+wd rc=1e-4 wd=1e-4": {
        "color": "#98df8a",
        "label": "mne_l2+wd rc=1e-4 wd=1e-4",
    },
    "mne_l2 rc=1e-4": {"color": "#1f77b4", "label": "mne_l2 rc=1e-4"},
}

ARCH = "vgg16"
DATASET = "cifar100"
SEED = 42
LVAL = 16
TVAL = 16
IF_MODE = "rate_uniform"
SCHEME_TAG = "schemeC_noout"

# 当 raw/matrix CSV 不在本地时，从原 6 曲线图转录的参考值（可 `--use-reference` 强制使用）
REFERENCE_CURVES: dict[str, list[tuple[float, float]]] = {
    "weight_decay": [
        (0.0, 62.0),
        (0.1, 62.5),
        (0.2, 62.3),
        (0.3, 62.1),
        (0.4, 60.8),
        (0.5, 58.2),
        (0.6, 53.1),
        (0.7, 45.8),
        (0.8, 36.2),
        (0.9, 25.1),
        (1.0, 16.0),
    ],
    "mne_l2+wd rc=1e-4 wd=1e-4": [
        (0.0, 64.1),
        (0.1, 64.2),
        (0.2, 64.1),
        (0.3, 63.9),
        (0.4, 63.8),
        (0.5, 63.5),
        (0.6, 63.1),
        (0.7, 62.9),
        (0.8, 62.3),
        (0.9, 61.5),
        (1.0, 60.3),
    ],
    "mne_l2 rc=1e-4": [
        (0.0, 59.5),
        (0.1, 59.6),
        (0.2, 59.5),
        (0.3, 59.6),
        (0.4, 59.5),
        (0.5, 59.6),
        (0.6, 59.5),
        (0.7, 59.6),
        (0.8, 59.5),
        (0.9, 59.6),
        (1.0, 59.5),
    ],
}


def read_matrix(mat: Path) -> list[tuple[float, float]]:
    with mat.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)
    start = 2 if header[:2] == ["L", "T"] else 1
    sigmas = [float(x) for x in header[start:]]
    accs = [float(x) for x in row[start:]]
    return list(zip(sigmas, accs))


def load_curves_from_raw(raw_csv: Path) -> dict[str, list[tuple[float, float]]]:
    curves: dict[str, list[tuple[float, float]]] = {}
    with raw_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            label = row["label"]
            if label not in SELECTED_LABELS:
                continue
            curves.setdefault(label, []).append((float(row["sigma"]), float(row["acc"])))
    for label in curves:
        curves[label] = sorted(curves[label], key=lambda x: x[0])
    return curves


def load_curves_from_matrices(out_dir: Path) -> dict[str, list[tuple[float, float]]]:
    curves: dict[str, list[tuple[float, float]]] = {}
    for label in SELECTED_LABELS:
        safe_label = label.replace(" ", "_").replace("=", "")
        subdir = out_dir / safe_label
        matrix = (
            subdir
            / f"noise_sweep_matrix_{DATASET}_{ARCH}_T{TVAL}_mode_{IF_MODE}_schedule_normal_seed_{SEED}.csv"
        )
        if not matrix.exists():
            cands = sorted(
                subdir.glob(
                    f"noise_sweep_matrix_*_T{TVAL}_mode_{IF_MODE}_schedule_normal_seed_{SEED}.csv"
                )
            )
            if not cands:
                continue
            matrix = cands[0]
        curves[label] = read_matrix(matrix)
    return curves


def load_curves(
    raw_csv: Path | None,
    out_dir: Path,
    use_reference: bool = False,
) -> dict[str, list[tuple[float, float]]]:
    if use_reference:
        return {k: REFERENCE_CURVES[k] for k in SELECTED_LABELS}

    if raw_csv and raw_csv.exists():
        curves = load_curves_from_raw(raw_csv)
        if len(curves) == len(SELECTED_LABELS):
            return curves

    curves = load_curves_from_matrices(out_dir)
    if len(curves) == len(SELECTED_LABELS):
        return curves

    print(
        "[WARN] 未找到完整 raw/matrix CSV，使用原图转录参考曲线。"
        " 若有实验 CSV，请传 --raw-csv 重新生成。",
        flush=True,
    )
    return {k: REFERENCE_CURVES[k] for k in SELECTED_LABELS}


def plot_results(
    curves: dict[str, list[tuple[float, float]]],
    save_dir: Path,
) -> None:
    plt.rcParams.update({"font.size": 11, "legend.fontsize": 10})
    for no_caption in (False, True):
        fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=180)
        all_y = []
        for label in SELECTED_LABELS:
            if label not in curves:
                continue
            pts = curves[label]
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            all_y.extend(y)
            style = LINE_STYLES[label]
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
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([round(i * 0.1, 1) for i in range(11)])
        if all_y:
            ax.set_ylim(min(all_y) - 1.0, max(all_y) + 1.0)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", frameon=False)
        if not no_caption:
            ax.set_title(
                f"CIFAR-100 {ARCH} wd vs mne_l2 vs mne_l2+wd "
                f"(seed={SEED}, L={LVAL}, T={TVAL}, {IF_MODE}, {SCHEME_TAG})"
            )
        fig.tight_layout()
        suffix = "_no_caption" if no_caption else ""
        out_png = save_dir / f"cifar100_{ARCH}_mne_l2_wd_combo_three_methods_noise_sweep{suffix}.png"
        fig.savefig(out_png)
        plt.close(fig)
        print(f"[PLOT] saved {out_png}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot 3-method subset for CIFAR-100 VGG16 combo")
    p.add_argument(
        "--raw-csv",
        type=Path,
        default=DEFAULT_RAW,
        help="汇总 raw CSV 路径",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出 PNG 目录（默认：raw CSV 所在目录；若无则用仓库根目录）",
    )
    p.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="实验结果目录（用于读取各 method 的 matrix CSV）",
    )
    p.add_argument(
        "--use-reference",
        action="store_true",
        help="强制使用内置参考曲线（原 6 曲线图转录值）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    curves = load_curves(args.raw_csv, args.result_dir, args.use_reference)
    missing = [label for label in SELECTED_LABELS if label not in curves]
    if missing:
        raise SystemExit(
            "缺少以下 method 的数据：\n  "
            + "\n  ".join(missing)
            + f"\n请确认 {args.raw_csv} 或 {args.result_dir} 下 matrix CSV 存在。"
        )

    save_dir = args.out_dir
    if save_dir is None:
        save_dir = args.raw_csv.parent if args.raw_csv.exists() else ROOT.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_results(curves, save_dir)


if __name__ == "__main__":
    main()
