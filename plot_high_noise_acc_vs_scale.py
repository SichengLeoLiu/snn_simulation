"""
High-noise SNN accuracy A_m(sigma=1.0) vs model scale — FC3 and CNN2 line plots.

Data: strict-seed three-regs noise sweep, L=16, T=16, rate_uniform, seeds 40–44.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "plots" / "high_noise_acc_vs_scale"

FC3_DATA_ROOT = (
    ROOT
    / "QCFS_simulation"
    / "noise3_exp"
    / "ablation_mne_l2_vs_weight_decay_l16_fc3_h4_h8_h16_h32_h64_h128"
    / "strict_seed_train_rate_uniform_L16_T16"
)
CNN_DATA_ROOT_OLD = ROOT / "cnn_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16"
CNN_DATA_ROOT_NEW = (
    ROOT
    / "QCFS_simulation"
    / "noise3_exp"
    / "cnn_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16"
)
# Architectures re-run into CNN_DATA_ROOT_NEW override the legacy root.
CNN_RERUN_ARCHS = {"cnn2_c8_c16"}


def cnn_data_root_for_arch(arch: str) -> Path:
    if arch in CNN_RERUN_ARCHS:
        new_dir = CNN_DATA_ROOT_NEW / arch
        if new_dir.exists() and any(new_dir.glob("*/seed_*/noise_sweep_matrix_*.csv")):
            return CNN_DATA_ROOT_NEW
    if CNN_DATA_ROOT_OLD.exists():
        return CNN_DATA_ROOT_OLD
    if CNN_DATA_ROOT_NEW.exists():
        return CNN_DATA_ROOT_NEW
    return CNN_DATA_ROOT_OLD


def collect_cnn_rows(target_sigma: float = 1.0) -> list[dict]:
    rows = []
    for arch, (x_num, x_label) in CNN_SCALES.items():
        data_root = cnn_data_root_for_arch(arch)
        source = "rerun" if data_root is CNN_DATA_ROOT_NEW and arch in CNN_RERUN_ARCHS else "legacy"
        for method in METHODS:
            mean, sem, std, n = aggregate_scale(data_root, arch, method, target_sigma)
            rows.append(
                {
                    "arch": arch,
                    "scale_x": x_num,
                    "scale_label": x_label,
                    "regularizer": method,
                    "acc_mean": mean,
                    "acc_sem": sem,
                    "acc_std": std,
                    "n_seeds": n,
                    "data_source": source,
                }
            )
        print(f"[CNN] {arch}: {source} ({data_root})", flush=True)
    return rows

METHODS = ["weight_decay", "mne_l2", "no_regularization"]
METHOD_LABELS = {
    "weight_decay": "L2",
    "mne_l2": "MNE L2",
    "no_regularization": "No reg",
}
METHOD_STYLES = {
    "weight_decay": {"color": "#ff7f0e", "marker": "o"},
    "mne_l2": {"color": "#1f77b4", "marker": "s"},
    "no_regularization": {"color": "#2ca02c", "marker": "^"},
}

FC3_SCALES = {
    "fc3_h4": (4, "h4"),
    "fc3_h8": (8, "h8"),
    "fc3_h16": (16, "h16"),
    "fc3_h32": (32, "h32"),
    "fc3_h64": (64, "h64"),
    "fc3_h128": (128, "h128"),
}

CNN_SCALES = {
    "cnn2_c2_c4": (4, "c2→c4"),
    "cnn2_c4_c8": (8, "c4→c8"),
    "cnn2_c8_c16": (16, "c8→c16"),
    "cnn2_c16_c32": (32, "c16→c32"),
}


def read_matrix_csv(path: Path) -> tuple[list[float], list[float]]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)
    start = 2 if header[:2] == ["L", "T"] else 1
    sigmas = [float(x) for x in header[start:]]
    accs = [float(x) for x in row[start:]]
    return sigmas, accs


def acc_at_sigma(sigmas: list[float], accs: list[float], target: float = 1.0) -> float:
    for sigma, acc in zip(sigmas, accs):
        if abs(sigma - target) < 1e-6:
            return acc
    if abs(sigmas[-1] - target) < 1e-3:
        return accs[-1]
    raise ValueError(f"sigma={target} not found in {sigmas}")


def aggregate_scale(
    data_root: Path,
    arch_dir: str,
    method: str,
    target_sigma: float = 1.0,
) -> tuple[float, float, float, int]:
    method_dir = data_root / arch_dir / method
    seed_files = sorted(method_dir.glob("seed_*/noise_sweep_matrix_*.csv"))
    if not seed_files:
        raise FileNotFoundError(f"no matrix csv under {method_dir}")

    vals = []
    for fp in seed_files:
        sigmas, accs = read_matrix_csv(fp)
        vals.append(acc_at_sigma(sigmas, accs, target_sigma))

    n = len(vals)
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if n > 1 else 0.0
    sem = std / (n**0.5) if n > 0 else 0.0
    return mean, sem, std, n


def collect_rows(
    scales: dict[str, tuple[int, str]],
    data_root: Path,
    target_sigma: float = 1.0,
) -> list[dict]:
    rows = []
    for arch, (x_num, x_label) in scales.items():
        for method in METHODS:
            mean, sem, std, n = aggregate_scale(data_root, arch, method, target_sigma)
            rows.append(
                {
                    "arch": arch,
                    "scale_x": x_num,
                    "scale_label": x_label,
                    "regularizer": method,
                    "acc_mean": mean,
                    "acc_sem": sem,
                    "acc_std": std,
                    "n_seeds": n,
                }
            )
    return rows


def save_summary_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "arch",
        "scale_label",
        "regularizer",
        "acc_mean",
        "acc_sem",
        "acc_std",
        "n_seeds",
        "data_source",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {k: row[k] for k in fields if k in row}
            if "data_source" not in out:
                out["data_source"] = ""
            writer.writerow(out)


def setup_style(font_size: float, legend_font_size: float) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": font_size,
            "axes.labelsize": font_size + 2,
            "axes.titlesize": font_size,
            "legend.fontsize": legend_font_size,
            "xtick.labelsize": font_size - 2,
            "ytick.labelsize": font_size - 2,
        }
    )


def plot_line_chart(
    rows: list[dict],
    scales: dict[str, tuple[int, str]],
    title: str | None,
    xlabel: str,
    out_stem: str,
    font_size: float = 18.0,
    legend_font_size: float = 16.0,
) -> None:
    setup_style(font_size, legend_font_size)
    fig, ax = plt.subplots(figsize=(10.6, 7.2), dpi=220)

    ordered_archs = list(scales.keys())
    x_nums = [scales[a][0] for a in ordered_archs]
    x_labels = [scales[a][1] for a in ordered_archs]

    all_y = []
    for method in METHODS:
        style = METHOD_STYLES[method]
        ys, yerrs = [], []
        for arch in ordered_archs:
            match = next(r for r in rows if r["arch"] == arch and r["regularizer"] == method)
            ys.append(match["acc_mean"])
            yerrs.append(match["acc_sem"])
        all_y.extend(ys)

        ax.errorbar(
            x_nums,
            ys,
            yerr=yerrs,
            label=METHOD_LABELS[method],
            color=style["color"],
            marker=style["marker"],
            markersize=9,
            linewidth=2.2,
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
        )

    ax.set_xticks(x_nums)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy (%)")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", frameon=True)
    ymin = max(0.0, min(all_y) - 12.0)
    ymax = min(100.0, max(all_y) + 8.0)
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = OUT_DIR / f"{out_stem}_with_sem.{ext}"
        fig.savefig(out_path, bbox_inches="tight")
        print(f"[PLOT] {out_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot high-noise SNN accuracy vs model scale (FC3 / CNN2 line charts)"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Noise level for y-axis (default: 1.0)",
    )
    parser.add_argument(
        "--fc3-scales",
        default="h8_h16_h32",
        choices=["all", "h8_h16_h32"],
        help="FC3 hidden sizes to include",
    )
    parser.add_argument("--font-size", type=float, default=18.0)
    parser.add_argument("--legend-font-size", type=float, default=16.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.fc3_scales == "h8_h16_h32":
        fc3_scales = {k: v for k, v in FC3_SCALES.items() if v[0] in (8, 16, 32)}
        fc3_stem = "fc3_three_regs_high_noise_acc_vs_hidden_h8_h16_h32"
    else:
        fc3_scales = FC3_SCALES
        fc3_stem = "fc3_three_regs_high_noise_acc_vs_hidden"

    fc3_rows = collect_rows(fc3_scales, FC3_DATA_ROOT, args.sigma)
    save_summary_csv(fc3_rows, OUT_DIR / f"{fc3_stem}.csv")
    plot_line_chart(
        fc3_rows,
        fc3_scales,
        title=None,
        xlabel="FC3 hidden size $h$",
        out_stem=fc3_stem,
        font_size=args.font_size,
        legend_font_size=args.legend_font_size,
    )

    cnn_rows = collect_cnn_rows(args.sigma)
    cnn_stem = "cnn2_three_regs_high_noise_acc_vs_scale"
    save_summary_csv(cnn_rows, OUT_DIR / f"{cnn_stem}.csv")
    plot_line_chart(
        cnn_rows,
        CNN_SCALES,
        title=None,
        xlabel=r"CNN2 channel scale ($C_1 \rightarrow C_2$)",
        out_stem=cnn_stem,
        font_size=args.font_size,
        legend_font_size=args.legend_font_size,
    )


if __name__ == "__main__":
    main()
