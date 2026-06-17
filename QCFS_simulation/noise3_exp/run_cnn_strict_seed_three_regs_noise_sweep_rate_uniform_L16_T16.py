"""
MNIST CNN2 多架构 strict-seed 三路正则 + rate_uniform 噪声扫描 + mean±std 折线图。

模型：cnn2_c2_c4 / cnn2_c4_c8 / cnn2_c8_c16 / cnn2_c16_c32
方法：mne_l2 (rc=5e-2) / weight_decay (wd=5e-4) / no_regularization
训练：L=16, T=0, spike_schedule=normal, 100 epochs, seeds=40..44
测试：L=16, T=16, IF mode=rate_uniform, sigma=0~1 step=0.1

用法：
  python noise3_exp/run_cnn_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16.py
  python noise3_exp/run_cnn_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16.py \
    --arch-list c2c4 c8c16 --reg mne_l2 --seed 42
  python noise3_exp/run_cnn_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16.py \
    --plot-only --copy-important --font-size 18 --legend-font-size 16
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "noise3_exp" / "cnn_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16"
IMPORTANT_RESULTS = ROOT.parent / "important results"

DEFAULT_SEEDS = [40, 41, 42, 43, 44]
CNN_VARIANTS = [(2, 4), (4, 8), (8, 16), (16, 32)]
REGS = ["mne_l2", "weight_decay", "no_regularization"]
LVAL = 16
TVAL = 16
IF_MODE = "rate_uniform"
EPOCHS = int(os.environ.get("CNN_EPOCHS", "100"))
BATCH = int(os.environ.get("CNN_BATCH", "128"))
NUM_WORKERS = int(os.environ.get("CNN_NUM_WORKERS", "8"))

LINE_STYLES = {
    "mne_l2": {"color": "#1f77b4", "label": "mne_l2 (mean)"},
    "weight_decay": {"color": "#ff7f0e", "label": "weight_decay (mean)"},
    "no_regularization": {"color": "#2ca02c", "label": "no regularization (mean)"},
}

RAW_CSV = OUT / "cnn_strict_seed_three_regs_noise_sweep_raw.csv"
AGG_CSV = OUT / "cnn_strict_seed_three_regs_noise_sweep_mean_std.csv"

RAW_FIELDS = [
    "arch", "c1", "c2", "size_label", "regularizer", "seed", "L", "T",
    "if_mode", "sigma", "acc", "checkpoint", "matrix_csv",
]
AGG_FIELDS = [
    "arch", "c1", "c2", "size_label", "regularizer", "sigma",
    "acc_mean", "acc_std", "n_seeds",
]

ARCH_ALIASES = {
    "c2c4": (2, 4),
    "c4c8": (4, 8),
    "c8c16": (8, 16),
    "c16c32": (16, 32),
}


def coeff_tag(v: float) -> str:
    return f"{v:.0e}".replace("-", "m").replace("+", "p")


def arch_name(c1: int, c2: int) -> str:
    return f"cnn2_c{c1}_c{c2}"


def size_label(c1: int, c2: int) -> str:
    return f"c{c1}c{c2}"


def resolve_arch_list(names: list[str] | None) -> list[tuple[int, int]]:
    if not names:
        return list(CNN_VARIANTS)
    out: list[tuple[int, int]] = []
    for name in names:
        key = name.strip().lower()
        if key in ARCH_ALIASES:
            pair = ARCH_ALIASES[key]
        elif key.startswith("cnn2_c") and "_c" in key:
            rest = key.replace("cnn2_c", "")
            c1_s, c2_s = rest.split("_c", 1)
            pair = (int(c1_s), int(c2_s))
        else:
            raise ValueError(f"未知 arch: {name!r}，可用 c2c4/c4c8/c8c16/c16c32")
        if pair not in out:
            out.append(pair)
    return out


def build_suffix(arch: str, reg: str, seed: int) -> str:
    if reg == "weight_decay":
        return f"strict_seed{seed}_ablation_wd_l{LVAL}_{arch}"
    if reg == "no_regularization":
        return f"strict_seed{seed}_ablation_none_l{LVAL}_{arch}"
    return f"strict_seed{seed}_ablation_mne_l2_l{LVAL}_{arch}_rc{coeff_tag(5e-2)}"


def ckpt_path(arch: str, reg: str, seed: int) -> Path:
    suffix = build_suffix(arch, reg, seed)
    return ROOT / "mnist-checkpoints" / f"{arch}_L[{LVAL}]_{suffix}.pth"


def test_out_dir(arch: str, reg: str, seed: int) -> Path:
    return OUT / arch / reg / f"seed_{seed}"


def matrix_path(arch: str, reg: str, seed: int) -> Path:
    return (
        test_out_dir(arch, reg, seed)
        / f"noise_sweep_matrix_mnist_{arch}_T{TVAL}_mode_{IF_MODE}_schedule_normal_seed_{seed}.csv"
    )


def important_plot_name(arch: str) -> str:
    return f"strict_seed_train_{arch}_rate_uniform_noise_sweep_mean_std_lineplot_no_caption.png"


def clear_test_artifacts(arch: str, reg: str, seed: int) -> None:
    out_dir = test_out_dir(arch, reg, seed)
    if not out_dir.exists():
        return
    for p in out_dir.glob("noise_sweep_matrix_*.csv"):
        p.unlink()
        print(f"[CLEAR] {p}", flush=True)
    for p in out_dir.glob("noise_sweep_combined_L_T.csv"):
        p.unlink()
        print(f"[CLEAR] {p}", flush=True)


def train_one(arch: str, reg: str, seed: int, retrain: bool) -> Path:
    ckpt = ckpt_path(arch, reg, seed)
    if retrain and ckpt.exists():
        ckpt.unlink()
        print(f"[RETRAIN] removed {ckpt.name}", flush=True)
    if ckpt.exists():
        print(f"[SKIP TRAIN] {ckpt.name}", flush=True)
        return ckpt

    if reg == "mne_l2":
        regularizer, wd, rc = "mne_l2", 0.0, 5e-2
    elif reg == "weight_decay":
        regularizer, wd, rc = "weight_decay", 5e-4, 1.0
    else:
        regularizer, wd, rc = "weight_decay", 0.0, 1.0

    cmd = [
        sys.executable,
        str(ROOT / "main_train.py"),
        "-data", "mnist",
        "-arch", arch,
        "-L", str(LVAL),
        "--epochs", str(EPOCHS),
        "-j", str(NUM_WORKERS),
        "-b", str(BATCH),
        "--seed", str(seed),
        "--device", "auto",
        "--time", "0",
        "--spike_schedule", "normal",
        "--regularizer", regularizer,
        "--weight_decay", str(wd),
        "--reg_coeff", str(rc),
        "--suffix", build_suffix(arch, reg, seed),
    ]
    print(f"[TRAIN] {arch} {reg} seed={seed}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
    print(f"[TRAIN DONE] {ckpt.name}", flush=True)
    return ckpt


def test_noise_sweep(
    arch: str, reg: str, seed: int, ckpt: Path, force_test: bool
) -> Path:
    if force_test:
        clear_test_artifacts(arch, reg, seed)
    matrix = matrix_path(arch, reg, seed)
    if matrix.exists():
        print(f"[SKIP TEST] {arch} {reg} seed={seed}", flush=True)
        return matrix

    out_dir = test_out_dir(arch, reg, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "main_test.py"),
        "-data", "mnist",
        "-arch", arch,
        "-L", str(LVAL),
        "-T", str(TVAL),
        "-j", str(NUM_WORKERS),
        "-b", str(BATCH),
        "--seed", str(seed),
        "--device", "auto",
        "--mode", IF_MODE,
        "--spike_schedule", "normal",
        "--weights", str(ckpt),
        "--noise_sweep",
        "--noise_sigma_start", "0.0",
        "--noise_sigma_end", "1.0",
        "--noise_sigma_step", "0.1",
        "--noise_output_dir", str(out_dir),
    ]
    print(f"[TEST] {arch} {reg} seed={seed} mode={IF_MODE}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    if not matrix.exists():
        cands = sorted(
            out_dir.glob(
                f"noise_sweep_matrix_*_T{TVAL}_mode_{IF_MODE}_schedule_normal_seed_{seed}.csv"
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


def load_raw_rows() -> list[dict]:
    if not RAW_CSV.exists():
        return []
    with RAW_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def upsert_run_rows(
    new_rows: list[dict],
    archs: set[str],
    regs: list[str],
    seeds: list[int],
) -> None:
    kept = [
        r
        for r in load_raw_rows()
        if not (
            r["arch"] in archs
            and r["regularizer"] in regs
            and int(r["seed"]) in seeds
        )
    ]
    kept.extend(new_rows)
    kept.sort(
        key=lambda r: (
            int(r["c1"]),
            int(r["c2"]),
            REGS.index(r["regularizer"]) if r["regularizer"] in REGS else 99,
            int(r["seed"]),
            float(r["sigma"]),
        )
    )
    OUT.mkdir(parents=True, exist_ok=True)
    with RAW_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writeheader()
        w.writerows(kept)


def aggregate_rows(raw_rows: list[dict]) -> list[dict]:
    bucket: dict[tuple[str, int, int, str, float], list[float]] = defaultdict(list)
    for row in raw_rows:
        bucket[
            (
                row["arch"],
                int(row["c1"]),
                int(row["c2"]),
                row["regularizer"],
                float(row["sigma"]),
            )
        ].append(float(row["acc"]))

    agg_rows = []
    for (arch, c1, c2, reg, sigma), vals in sorted(
        bucket.items(),
        key=lambda x: (
            x[0][1],
            x[0][2],
            REGS.index(x[0][3]) if x[0][3] in REGS else 99,
            x[0][4],
        ),
    ):
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        agg_rows.append(
            {
                "arch": arch,
                "c1": c1,
                "c2": c2,
                "size_label": size_label(c1, c2),
                "regularizer": reg,
                "sigma": f"{sigma:.1f}",
                "acc_mean": f"{mean:.6f}",
                "acc_std": f"{std:.6f}",
                "n_seeds": len(vals),
            }
        )
    return agg_rows


def plot_results(
    agg_rows: list[dict],
    variants: list[tuple[int, int]],
    copy_important: bool,
    font_size: float,
    legend_font_size: float,
) -> None:
    plt.rcParams.update({"font.size": font_size, "legend.fontsize": legend_font_size})
    for c1, c2 in variants:
        arch = arch_name(c1, c2)
        rows_arch = [r for r in agg_rows if r["arch"] == arch]
        if not rows_arch:
            print(f"[PLOT] skip {arch}: no data", flush=True)
            continue

        fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=180)
        all_y: list[float] = []
        for reg in REGS:
            rr = [r for r in rows_arch if r["regularizer"] == reg]
            if not rr:
                continue
            rr.sort(key=lambda x: float(x["sigma"]))
            x = [float(r["sigma"]) for r in rr]
            y = [float(r["acc_mean"]) for r in rr]
            s = [float(r["acc_std"]) for r in rr]
            all_y.extend([yy - ss for yy, ss in zip(y, s)])
            all_y.extend([yy + ss for yy, ss in zip(y, s)])
            style = LINE_STYLES[reg]
            ax.plot(
                x, y, marker="o", linewidth=2.4, markersize=6,
                color=style["color"], label=style["label"],
            )
            if any(ss > 0 for ss in s):
                ax.fill_between(
                    x,
                    [yy - ss for yy, ss in zip(y, s)],
                    [yy + ss for yy, ss in zip(y, s)],
                    color=style["color"], alpha=0.18, linewidth=0,
                )
        ax.set_xlabel("Gaussian noise sigma")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([round(i * 0.1, 1) for i in range(11)])
        if all_y:
            ax.set_ylim(min(all_y) - 0.8, max(all_y) + 0.8)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", frameon=False)
        fig.tight_layout()
        out_png = OUT / f"strict_seed_train_{arch}_noise_sweep_mean_std_lineplot_no_caption.png"
        fig.savefig(out_png)
        plt.close(fig)
        print(f"[PLOT] saved {out_png}", flush=True)
        if copy_important:
            IMPORTANT_RESULTS.mkdir(parents=True, exist_ok=True)
            dest = IMPORTANT_RESULTS / important_plot_name(arch)
            shutil.copy2(out_png, dest)
            print(f"[PLOT] copied {dest}", flush=True)


def finalize_tables_and_plots(
    variants: list[tuple[int, int]],
    plot_variants: list[tuple[int, int]],
    copy_important: bool,
    font_size: float,
    legend_font_size: float,
    replot: bool,
) -> None:
    raw_rows = load_raw_rows()
    agg_rows = aggregate_rows(raw_rows)
    with AGG_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AGG_FIELDS)
        w.writeheader()
        w.writerows(agg_rows)
    print(f"[TABLE] raw: {RAW_CSV}", flush=True)
    print(f"[TABLE] agg: {AGG_CSV}", flush=True)
    if replot:
        plot_results(
            agg_rows, plot_variants, copy_important, font_size, legend_font_size
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MNIST CNN2 strict-seed 三路正则 + rate_uniform 噪声实验"
    )
    p.add_argument(
        "--arch-list",
        nargs="+",
        default=None,
        help="c2c4 c4c8 c8c16 c16c32 或 cnn2_c2_c4 等（默认全部）",
    )
    p.add_argument(
        "--plot-arch-list",
        nargs="+",
        default=None,
        help="重画图时的架构列表（默认与 --arch-list 相同）",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--reg",
        choices=REGS + ["all"],
        default="all",
        help="只跑一种正则（默认 all）",
    )
    p.add_argument("--retrain", action="store_true", help="删除并重新训练 checkpoint")
    p.add_argument("--force-test", action="store_true", help="删除并重新跑噪声扫描")
    p.add_argument("--replot", action="store_true", help="结束后重算表并出图")
    p.add_argument("--copy-important", action="store_true")
    p.add_argument("--font-size", type=float, default=18.0)
    p.add_argument("--legend-font-size", type=float, default=16.0)
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="仅从 raw CSV 重算 agg 并重画图（不训练/测试）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    variants = resolve_arch_list(args.arch_list)
    plot_variants = resolve_arch_list(
        args.plot_arch_list if args.plot_arch_list is not None else args.arch_list
    )
    if args.plot_arch_list is None and args.arch_list is None:
        plot_variants = list(CNN_VARIANTS)
    seeds = [args.seed] if args.seed is not None else args.seeds
    regs = REGS if args.reg == "all" else [args.reg]
    archs = {arch_name(c1, c2) for c1, c2 in variants}

    if args.plot_only:
        finalize_tables_and_plots(
            variants, plot_variants, args.copy_important,
            args.font_size, args.legend_font_size, replot=True,
        )
        return

    new_rows: list[dict] = []
    for c1, c2 in variants:
        arch = arch_name(c1, c2)
        label = size_label(c1, c2)
        for reg in regs:
            for seed in seeds:
                ckpt = train_one(arch, reg, seed, args.retrain)
                mat = test_noise_sweep(arch, reg, seed, ckpt, args.force_test)
                for sigma, acc in read_matrix(mat):
                    new_rows.append(
                        {
                            "arch": arch,
                            "c1": c1,
                            "c2": c2,
                            "size_label": label,
                            "regularizer": reg,
                            "seed": seed,
                            "L": LVAL,
                            "T": TVAL,
                            "if_mode": IF_MODE,
                            "sigma": sigma,
                            "acc": acc,
                            "checkpoint": str(ckpt.relative_to(ROOT)),
                            "matrix_csv": str(mat.relative_to(ROOT)),
                        }
                    )

    upsert_run_rows(new_rows, archs, regs, seeds)
    finalize_tables_and_plots(
        variants, plot_variants, args.copy_important,
        args.font_size, args.legend_font_size, replot=True,
    )


if __name__ == "__main__":
    main()
