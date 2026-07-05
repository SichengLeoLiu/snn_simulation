"""
MNIST fc3 strict-seed 三路正则 + rate_uniform 噪声扫描 + mean±std 折线图。

方法：mne_l2 / weight_decay / no_regularization（与 normal 版共用 checkpoint）
测试：L=16, T=16, IF mode=rate_uniform, sigma=0~1 step=0.05, seeds=40..44

用法：
  # 全量（跳过已有 checkpoint / matrix）
  python noise3_exp/run_fc3_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16.py

  # 重做 h32/h64：重训 + 重测 + 重画图
  python noise3_exp/run_fc3_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16.py \
    --h-list 32 64 --retrain --force-test --replot --copy-important

  # 仅重跑噪声扫描（checkpoint 保留）
  python noise3_exp/run_fc3_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16.py \
    --h-list 32 64 --force-test --replot --copy-important
"""
from __future__ import annotations

import argparse
import csv
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = (
    ROOT
    / "noise3_exp"
    / "ablation_mne_l2_vs_weight_decay_l16_fc3_h4_h8_h16_h32_h64_h128"
    / "strict_seed_train_rate_uniform_L16_T16"
)
IMPORTANT_RESULTS = ROOT.parent / "important results"
DERIVATIVE_RESULTS = ROOT.parent / "derivative results"

DEFAULT_SEEDS = [40, 41, 42, 43, 44]
ALL_H_LIST = [4, 8, 16, 32, 64, 128]
REGS = ["mne_l2", "weight_decay", "no_regularization"]
LVAL = 16
TVAL = 16
IF_MODE = "rate_uniform"

LINE_STYLES = {
    "mne_l2": {"color": "#1f77b4", "label": "MNE-L2"},
    "weight_decay": {"color": "#ff7f0e", "label": "L2"},
    "no_regularization": {"color": "#2ca02c", "label": "No Reg"},
}

RAW_CSV = OUT / "strict_seed_train_noise_sweep_fc3_h4_h8_h16_h32_h64_h128_raw.csv"
AGG_CSV = OUT / "strict_seed_train_noise_sweep_fc3_h4_h8_h16_h32_h64_h128_mean_std.csv"
BACKUP_ROOT = OUT.parent / f"{OUT.name}_backups"


def _backup_path(category: str, name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_ROOT / category / ts / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def _backup_file(path: Path, category: str) -> None:
    if not path.exists():
        return
    dest = _backup_path(category, path.name)
    shutil.move(str(path), str(dest))
    print(f"[BACKUP] {path} -> {dest}", flush=True)

RAW_FIELDS = [
    "arch", "hidden_size", "regularizer", "seed", "L", "T",
    "if_mode", "sigma", "acc", "checkpoint", "matrix_csv",
]
AGG_FIELDS = [
    "arch", "hidden_size", "regularizer", "sigma",
    "acc_mean", "acc_std", "n_seeds",
]


def coeff_tag(v: float) -> str:
    return f"{v:.0e}".replace("-", "m").replace("+", "p")


def arch_for(h: int) -> str:
    return f"fc3_h{h}"


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


def clear_test_artifacts(arch: str, reg: str, seed: int) -> None:
    out_dir = test_out_dir(arch, reg, seed)
    if not out_dir.exists():
        return
    tag = f"{arch}_{reg}_seed{seed}"
    for p in out_dir.glob("noise_sweep_matrix_*.csv"):
        _backup_file(p, f"noise_sweep/{tag}")
    for p in out_dir.glob("noise_sweep_combined_L_T.csv"):
        _backup_file(p, f"noise_sweep/{tag}")


def train_one(arch: str, reg: str, seed: int, retrain: bool, epochs: int) -> Path:
    ckpt = ckpt_path(arch, reg, seed)
    if retrain and ckpt.exists():
        _backup_file(ckpt, f"checkpoints/{arch}_{reg}_seed{seed}")
        print(f"[RETRAIN] will retrain {ckpt.name}", flush=True)
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
        "--epochs", str(epochs),
        "-j", "0",
        "-b", "128",
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
        "-j", "0",
        "-b", "128",
        "--seed", str(seed),
        "--device", "auto",
        "--mode", IF_MODE,
        "--spike_schedule", "normal",
        "--weights", str(ckpt),
        "--noise_sweep",
        "--noise_sigma_start", "0.0",
        "--noise_sigma_end", "1.0",
        "--noise_sigma_step", "0.05",
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
    h_list: list[int],
    regs: list[str],
    seeds: list[int],
) -> None:
    archs = {arch_for(h) for h in h_list}
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
            int(r["hidden_size"]),
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
    bucket: dict[tuple[str, int, str, float], list[float]] = defaultdict(list)
    for row in raw_rows:
        sigma = round(float(row["sigma"]), 6)
        bucket[(row["arch"], int(row["hidden_size"]), row["regularizer"], sigma)].append(
            float(row["acc"])
        )
    agg_rows = []
    for (arch, h, reg, sigma), vals in sorted(
        bucket.items(),
        key=lambda x: (
            x[0][1],
            REGS.index(x[0][2]) if x[0][2] in REGS else 99,
            x[0][3],
        ),
    ):
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        agg_rows.append(
            {
                "arch": arch,
                "hidden_size": h,
                "regularizer": reg,
                "sigma": f"{sigma:.2f}",
                "acc_mean": f"{mean:.6f}",
                "acc_std": f"{std:.6f}",
                "n_seeds": len(vals),
            }
        )
    return agg_rows


def important_plot_name(arch: str) -> str:
    return f"strict_seed_train_{arch}_rate_uniform_noise_sweep_mean_std_lineplot_no_caption.png"


def plot_results(
    agg_rows: list[dict],
    h_list: list[int],
    copy_important: bool,
    important_subdir: str | None,
    font_size: float,
    legend_font_size: float,
) -> None:
    plt.rcParams.update({"font.size": font_size, "legend.fontsize": legend_font_size})
    important_dir: Path | None = None
    if copy_important:
        subdir = important_subdir or f"fc3_rate_uniform_noise_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        important_dir = IMPORTANT_RESULTS / subdir
        important_dir.mkdir(parents=True, exist_ok=True)
        print(f"[PLOT] copy dir: {important_dir}", flush=True)
    for h in h_list:
        arch = arch_for(h)
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
        if copy_important and important_dir is not None:
            dest = important_dir / important_plot_name(arch)
            shutil.copy2(out_png, dest)
            print(f"[PLOT] copied {dest}", flush=True)


def replot_derivatives(h_list: list[int], font_size: float, legend_font_size: float) -> None:
    script = ROOT / "noise3_exp" / "plot_fc3_strict_seed_rate_uniform_acc_derivative.py"
    cmd = [
        sys.executable,
        str(script),
        "--h-list",
        *[str(h) for h in h_list],
        "--font-size",
        str(font_size),
        "--legend-font-size",
        str(legend_font_size),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def finalize_tables_and_plots(
    h_list: list[int],
    plot_h_list: list[int],
    copy_important: bool,
    important_subdir: str | None,
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
            agg_rows, plot_h_list, copy_important, important_subdir, font_size, legend_font_size
        )
        replot_derivatives(plot_h_list, font_size, legend_font_size)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="fc3 strict-seed 三路 rate_uniform 噪声实验")
    p.add_argument("--h-list", type=int, nargs="+", default=ALL_H_LIST)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--reg",
        choices=REGS + ["all"],
        default="all",
        help="只跑一种正则（默认 all）",
    )
    p.add_argument(
        "--retrain",
        action="store_true",
        help="备份旧 checkpoint 到 ..._backups/ 后重新训练",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="训练轮数（MNIST 默认 100，可按需求下调）",
    )
    p.add_argument(
        "--force-test",
        action="store_true",
        help="备份旧噪声 CSV 到 ..._backups/ 后重新跑噪声扫描",
    )
    p.add_argument("--replot", action="store_true", help="结束后重算表并出图")
    p.add_argument(
        "--plot-h-list",
        type=int,
        nargs="+",
        default=None,
        help="重画图时的 h 列表（默认与 --h-list 相同；可设 4 8 16 32 64 128 更新全部图）",
    )
    p.add_argument("--copy-important", action="store_true")
    p.add_argument(
        "--important-subdir",
        type=str,
        default=None,
        help="当 --copy-important 开启时，复制到 important results 下该子目录（默认自动时间戳目录）",
    )
    p.add_argument("--font-size", type=float, default=14.0)
    p.add_argument("--legend-font-size", type=float, default=12.0)
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="仅从 raw CSV 重算 agg 并重画图（不训练/测试）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [args.seed] if args.seed is not None else args.seeds
    regs = REGS if args.reg == "all" else [args.reg]
    plot_h_list = args.plot_h_list if args.plot_h_list is not None else args.h_list

    if args.plot_only:
        finalize_tables_and_plots(
            args.h_list, plot_h_list, args.copy_important, args.important_subdir,
            args.font_size, args.legend_font_size, replot=True,
        )
        return

    new_rows: list[dict] = []
    for h in args.h_list:
        arch = arch_for(h)
        for reg in regs:
            for seed in seeds:
                ckpt = train_one(arch, reg, seed, args.retrain, args.epochs)
                mat = test_noise_sweep(arch, reg, seed, ckpt, args.force_test)
                for sigma, acc in read_matrix(mat):
                    new_rows.append(
                        {
                            "arch": arch,
                            "hidden_size": h,
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

    upsert_run_rows(new_rows, args.h_list, regs, seeds)
    if args.replot:
        finalize_tables_and_plots(
            args.h_list, plot_h_list, args.copy_important, args.important_subdir,
            args.font_size, args.legend_font_size, replot=True,
        )


if __name__ == "__main__":
    main()
