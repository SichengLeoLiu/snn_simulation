"""
CIFAR-10 / CIFAR-100 VGG16：三路正则 strict-seed 训练 + 噪声扫描 + mean±std 折线图。

方法（同一 pipeline）：
  - weight_decay / L2 (wd=5e-4)
  - mne_l2 / MNE L2 (rc=1e-4)
  - no_regularization / No reg (wd=0)

默认 5 seeds: 40,41,42,43,44；L=16, T=16, rate_uniform, sigma=0~1.0 step=0.05。
方案 C：输出层不参与 MNE-L2。

用法：
  python noise3_exp/run_cifar_vgg16_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16.py --dataset cifar10
  python noise3_exp/run_cifar_vgg16_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16.py --dataset cifar100
  python noise3_exp/run_cifar_vgg16_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16.py --dataset cifar10 --method no_regularization
  python noise3_exp/run_cifar_vgg16_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16.py --dataset cifar100 --plot-only

  # 强制重跑噪声扫描（不重新训练）
  python noise3_exp/run_cifar_vgg16_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16.py \\
      --dataset cifar10 --force-test

  # 强制重训 + 重测（新 run-tag，不覆盖旧 checkpoint/结果）
  python noise3_exp/run_cifar_vgg16_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16.py \\
      --dataset cifar10 --retrain --force-test \\
      --run-tag drs_rerun \\
      --out-dir ../important_results/cifar10_vgg16_three_regs_drs_rerun

或：
  bash noise3_exp/RUN_cifar10_vgg16_three_regs_rerun.sh
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _path_for_csv(path: Path) -> str:
    p = path.resolve()
    try:
        return str(p.relative_to(ROOT.resolve()))
    except ValueError:
        return str(p)


ARCH = "vgg16"
LVAL = 16
TVAL = 16
IF_MODE = "rate_uniform"
EPOCHS = int(os.environ.get("CIFAR_EPOCHS", "300"))
LR = 0.1
BATCH = int(os.environ.get("CIFAR_BATCH", "128"))
NUM_WORKERS = int(os.environ.get("CIFAR_NUM_WORKERS", "8"))
SCHEME_TAG = "schemeC_noout"
DEFAULT_SEEDS = [40, 41, 42, 43, 44]

MNE_RC = 1e-4
WD_BASE = 5e-4

METHOD_KEYS = ["weight_decay", "mne_l2", "no_regularization"]

METHOD_ALIASES = {
    "weight_decay": "weight_decay",
    "wd": "weight_decay",
    "l2": "weight_decay",
    "mne_l2": "mne_l2",
    "mnel2": "mne_l2",
    "no_regularization": "no_regularization",
    "no_reg": "no_regularization",
    "none": "no_regularization",
    "all": "all",
}

METHOD_CONFIG = {
    "weight_decay": {
        "label": "weight_decay",
        "reg_coeff": None,
        "wd": WD_BASE,
    },
    "mne_l2": {
        "label": "mne_l2 rc=1e-4",
        "reg_coeff": MNE_RC,
        "wd": 0.0,
    },
    "no_regularization": {
        "label": "no regularization",
        "reg_coeff": None,
        "wd": 0.0,
    },
}

PLOT_ORDER = [
    "weight_decay",
    "mne_l2 rc=1e-4",
    "no regularization",
]

LINE_STYLES = {
    "weight_decay": {"color": "#ff7f0e", "label": "L2"},
    "mne_l2 rc=1e-4": {"color": "#1f77b4", "label": "MNE L2"},
    "no regularization": {"color": "#2ca02c", "label": "No reg"},
}

RAW_FIELDS = [
    "dataset",
    "arch",
    "method",
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

AGG_FIELDS = ["method", "label", "sigma", "acc_mean", "acc_std", "n_seeds"]


def coeff_tag(v: float) -> str:
    return f"{v:.0e}".replace("-", "m").replace("+", "p")


def out_dir_for(dataset: str, run_tag: Optional[str] = None, out_dir: Optional[Path] = None) -> Path:
    if out_dir is not None:
        out = Path(out_dir)
    elif run_tag:
        out = (
            ROOT
            / "noise3_exp"
            / f"{dataset}_{ARCH}_{run_tag}_three_regs_noise_sweep_rate_uniform_L16_T16"
        )
    else:
        out = (
            ROOT
            / "noise3_exp"
            / f"{dataset}_{ARCH}_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16"
        )
    out.mkdir(parents=True, exist_ok=True)
    return out


def scheme_mid(run_tag: Optional[str]) -> str:
    if run_tag:
        return f"{run_tag}_{SCHEME_TAG}"
    return SCHEME_TAG


def raw_csv_path(out: Path, dataset: str) -> Path:
    return out / f"{dataset}_{ARCH}_strict_seed_three_regs_noise_sweep_raw.csv"


def agg_csv_path(out: Path, dataset: str) -> Path:
    return out / f"{dataset}_{ARCH}_strict_seed_three_regs_noise_sweep_mean_std.csv"


def build_suffix(method_key: str, seed: int, run_tag: Optional[str] = None) -> str:
    mid = scheme_mid(run_tag)
    if method_key == "no_regularization":
        return f"strict_seed{seed}_{mid}_none_l{LVAL}_{ARCH}"
    cfg = METHOD_CONFIG[method_key]
    reg_coeff = cfg["reg_coeff"]
    wd = cfg["wd"]
    if reg_coeff is None:
        return f"strict_seed{seed}_{mid}_wd_l{LVAL}_{ARCH}"
    if wd > 0:
        return (
            f"strict_seed{seed}_{mid}_mne_l2_wd_l{LVAL}_{ARCH}"
            f"_rc{coeff_tag(reg_coeff)}_wd{coeff_tag(wd)}"
        )
    return (
        f"strict_seed{seed}_{mid}_mne_l2_l{LVAL}_{ARCH}"
        f"_rc{coeff_tag(reg_coeff)}"
    )


def ckpt_path(dataset: str, method_key: str, seed: int, run_tag: Optional[str] = None) -> Path:
    suffix = build_suffix(method_key, seed, run_tag)
    return ROOT / f"{dataset}-checkpoints" / f"{ARCH}_L[{LVAL}]_{suffix}.pth"


def resolve_method(name: str) -> str:
    key = METHOD_ALIASES.get(name.strip().lower())
    if key is None or key == "all":
        raise ValueError(f"未知 method: {name!r}")
    return key


def test_out_dir(out: Path, method_key: str, seed: int) -> Path:
    safe_label = METHOD_CONFIG[method_key]["label"].replace(" ", "_").replace("=", "")
    return out / safe_label / f"seed_{seed}"


def clear_test_artifacts(out: Path, method_key: str, seed: int) -> None:
    test_dir = test_out_dir(out, method_key, seed)
    if not test_dir.exists():
        return
    for p in test_dir.glob("noise_sweep_matrix_*.csv"):
        p.unlink()
        print(f"[CLEAR] {p}", flush=True)
    for p in test_dir.glob("noise_sweep_combined_L_T.csv"):
        p.unlink()
        print(f"[CLEAR] {p}", flush=True)


def train_one(
    dataset: str,
    method_key: str,
    seed: int,
    run_tag: Optional[str] = None,
    retrain: bool = False,
    ckpt_save_mode: str = "best",
) -> Path:
    cfg = METHOD_CONFIG[method_key]
    ckpt = ckpt_path(dataset, method_key, seed, run_tag)
    if retrain and ckpt.exists():
        ckpt.unlink()
        print(f"[RETRAIN] removed {ckpt.name}", flush=True)
    if ckpt.exists():
        print(f"[SKIP TRAIN] {ckpt.name}", flush=True)
        return ckpt

    reg_coeff = cfg["reg_coeff"]
    wd = cfg["wd"]
    if method_key == "no_regularization":
        regularizer, rc = "weight_decay", 1.0
    elif reg_coeff is None:
        regularizer, rc = "weight_decay", 1.0
    else:
        regularizer, rc = "mne_l2", reg_coeff

    cmd = [
        sys.executable,
        str(ROOT / "main_train.py"),
        "-data",
        dataset,
        "-arch",
        ARCH,
        "-L",
        str(LVAL),
        "--epochs",
        str(EPOCHS),
        "-lr",
        str(LR),
        "-j",
        str(NUM_WORKERS),
        "-b",
        str(BATCH),
        "--seed",
        str(seed),
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
        build_suffix(method_key, seed, run_tag),
        "--ckpt-save-mode",
        ckpt_save_mode,
    ]
    if regularizer == "mne_l2":
        cmd.append("--mne_detach_lambda")

    print(
        f"[TRAIN] {dataset} {method_key} seed={seed} epochs={EPOCHS}",
        flush=True,
    )
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
    print(f"[TRAIN DONE] {ckpt.name}", flush=True)
    return ckpt


def test_noise_sweep(
    dataset: str,
    method_key: str,
    seed: int,
    ckpt: Path,
    out: Path,
    force_test: bool = False,
    first_layer_noise_position: str = "post_input_if",
) -> Path:
    if force_test:
        clear_test_artifacts(out, method_key, seed)
    test_dir = test_out_dir(out, method_key, seed)
    test_dir.mkdir(parents=True, exist_ok=True)
    matrix = (
        test_dir
        / f"noise_sweep_matrix_{dataset}_{ARCH}_T{TVAL}_mode_{IF_MODE}_schedule_normal_seed_{seed}.csv"
    )
    if matrix.exists():
        print(f"[SKIP TEST] {method_key} seed={seed}", flush=True)
        return matrix

    cmd = [
        sys.executable,
        str(ROOT / "main_test.py"),
        "-data",
        dataset,
        "-arch",
        ARCH,
        "-L",
        str(LVAL),
        "-T",
        str(TVAL),
        "-j",
        str(NUM_WORKERS),
        "-b",
        str(BATCH),
        "--seed",
        str(seed),
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
        "0.05",
        "--first_layer_noise_position",
        first_layer_noise_position,
        "--noise_output_dir",
        str(test_dir),
    ]
    print(f"[TEST] {method_key} seed={seed} mode={IF_MODE}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    if not matrix.exists():
        cands = sorted(
            test_dir.glob(
                f"noise_sweep_matrix_*_T{TVAL}_mode_{IF_MODE}_schedule_normal_seed_{seed}.csv"
            )
        )
        if not cands:
            raise FileNotFoundError(f"matrix missing: {test_dir}")
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


def load_raw_rows(raw_csv: Path) -> list[dict]:
    if not raw_csv.exists():
        return []
    with raw_csv.open(newline="") as f:
        return list(csv.DictReader(f))


def upsert_run_rows(
    raw_csv: Path,
    method_key: str,
    seed: int,
    new_rows: list[dict],
) -> list[dict]:
    kept = [
        r
        for r in load_raw_rows(raw_csv)
        if not (r["method"] == method_key and int(r["seed"]) == seed)
    ]
    kept.extend(new_rows)
    kept.sort(
        key=lambda r: (
            PLOT_ORDER.index(r["label"]) if r["label"] in PLOT_ORDER else 99,
            int(r["seed"]),
            float(r["sigma"]),
        )
    )
    with raw_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(kept)
    return kept


def aggregate_rows(raw_rows: list[dict]) -> list[dict]:
    bucket: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    for row in raw_rows:
        if row["method"] not in METHOD_KEYS:
            continue
        bucket[(row["method"], row["label"], float(row["sigma"]))].append(
            float(row["acc"])
        )

    def _plot_idx(label: str) -> int:
        try:
            return PLOT_ORDER.index(label)
        except ValueError:
            return 99

    agg_rows = []
    for (method_key, label, sigma), vals in sorted(
        bucket.items(), key=lambda x: (_plot_idx(x[0][1]), x[0][2])
    ):
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        agg_rows.append(
            {
                "method": method_key,
                "label": label,
                "sigma": f"{sigma:.1f}",
                "acc_mean": f"{mean:.6f}",
                "acc_std": f"{std:.6f}",
                "n_seeds": len(vals),
            }
        )
    return agg_rows


def plot_results(
    dataset: str,
    agg_rows: list[dict],
    out: Path,
    run_tag: Optional[str] = None,
    font_size: float = 20.0,
    legend_font_size: float = 18.0,
) -> None:
    if not agg_rows:
        print("[PLOT] 无汇总数据，跳过", flush=True)
        return

    multi_seed = any(int(r["n_seeds"]) > 1 for r in agg_rows)
    plt.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": legend_font_size,
    })
    for no_caption in (False, True):
        fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=180)
        all_y = []
        for label in PLOT_ORDER:
            rr = [r for r in agg_rows if r["label"] == label]
            if not rr:
                continue
            rr.sort(key=lambda x: float(x["sigma"]))
            x = [float(r["sigma"]) for r in rr]
            y = [float(r["acc_mean"]) for r in rr]
            s = [float(r["acc_std"]) for r in rr]
            all_y.extend([yy - ss for yy, ss in zip(y, s)])
            all_y.extend([yy + ss for yy, ss in zip(y, s)])
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
            if multi_seed:
                ax.fill_between(
                    x,
                    [yy - ss for yy, ss in zip(y, s)],
                    [yy + ss for yy, ss in zip(y, s)],
                    color=style["color"],
                    alpha=0.18,
                    linewidth=0,
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
            n_seeds = max(int(r["n_seeds"]) for r in agg_rows)
            ax.set_title(
                f"{dataset.upper()} {ARCH} strict-seed three-regs noise sweep "
                f"(seeds={n_seeds}, L={LVAL}, T={TVAL}, {IF_MODE}, {scheme_mid(run_tag)})"
            )
        fig.tight_layout()
        suffix = "_no_caption" if no_caption else ""
        out_png = (
            out
            / f"{dataset}_{ARCH}_strict_seed_three_regs_noise_sweep_mean_std_lineplot{suffix}.png"
        )
        fig.savefig(out_png)
        plt.close(fig)
        print(f"[PLOT] saved {out_png}", flush=True)


def run_one(
    dataset: str,
    method_key: str,
    seed: int,
    out: Path,
    run_tag: Optional[str] = None,
    retrain: bool = False,
    force_test: bool = False,
    ckpt_save_mode: str = "best",
    first_layer_noise_position: str = "post_input_if",
) -> list[dict]:
    cfg = METHOD_CONFIG[method_key]
    ckpt = train_one(
        dataset,
        method_key,
        seed,
        run_tag=run_tag,
        retrain=retrain,
        ckpt_save_mode=ckpt_save_mode,
    )
    matrix = test_noise_sweep(
        dataset,
        method_key,
        seed,
        ckpt,
        out,
        force_test=force_test,
        first_layer_noise_position=first_layer_noise_position,
    )
    curve = read_matrix(matrix)
    rows = []
    for sigma, acc in curve:
        rows.append(
            {
                "dataset": dataset,
                "arch": ARCH,
                "method": method_key,
                "label": cfg["label"],
                "reg_coeff": "" if cfg["reg_coeff"] is None else cfg["reg_coeff"],
                "weight_decay": cfg["wd"],
                "seed": seed,
                "L": LVAL,
                "T": TVAL,
                "if_mode": IF_MODE,
                "sigma": sigma,
                "acc": acc,
                "checkpoint": _path_for_csv(ckpt),
                "matrix_csv": _path_for_csv(matrix),
            }
        )
    return rows


def finalize_outputs(
    dataset: str,
    out: Path,
    run_tag: Optional[str] = None,
    font_size: float = 18.0,
    legend_font_size: float = 16.0,
) -> None:
    raw_csv = raw_csv_path(out, dataset)
    raw_rows = load_raw_rows(raw_csv)
    agg_rows = aggregate_rows(raw_rows)
    agg_csv = agg_csv_path(out, dataset)
    with agg_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AGG_FIELDS)
        writer.writeheader()
        writer.writerows(agg_rows)
    plot_results(dataset, agg_rows, out, run_tag, font_size, legend_font_size)
    print(f"[TABLE] raw: {raw_csv}", flush=True)
    print(f"[TABLE] agg: {agg_csv}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CIFAR-10/100 VGG16 strict-seed 三路正则 (L2/MNE L2/No reg) + 噪声 mean±std"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["cifar10", "cifar100"],
        help="数据集",
    )
    parser.add_argument(
        "--method",
        default="all",
        help="weight_decay | mne_l2 | no_regularization | all（默认 all=三路）",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=f"随机种子列表（默认 {' '.join(map(str, DEFAULT_SEEDS))}）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="只跑单个 seed（覆盖 --seeds）",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="仅从已有 raw CSV 重算 mean±std 并出图（不训练/测试）",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="删除已有 checkpoint 后重新训练",
    )
    parser.add_argument(
        "--ckpt-save-mode",
        choices=["best", "last"],
        default="best",
        help="训练时 checkpoint 保存策略：best=验证最优（默认），last=最后一个 epoch",
    )
    parser.add_argument(
        "--force-test",
        action="store_true",
        help="删除已有 noise_sweep matrix 后重新测试",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="写入 checkpoint suffix 与默认输出目录名，避免覆盖旧实验（如 drs_rerun）",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="结果输出目录（raw/agg CSV、噪声 matrix、折线图）；默认随 --run-tag 或旧路径",
    )
    parser.add_argument("--font-size", type=float, default=20.0)
    parser.add_argument("--legend-font-size", type=float, default=18.0)
    parser.add_argument(
        "--first-layer-noise-position",
        choices=["post_input_if", "pre_input_if"],
        default="post_input_if",
        help="噪声注入位置：post_input_if(默认) 或 pre_input_if",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset
    run_tag = args.run_tag
    out = out_dir_for(
        dataset,
        run_tag=run_tag,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
    raw_csv = raw_csv_path(out, dataset)

    if args.plot_only:
        finalize_outputs(
            dataset, out, run_tag, args.font_size, args.legend_font_size
        )
        return

    seeds = [args.seed] if args.seed is not None else args.seeds
    if args.method.strip().lower() == "all":
        method_keys = list(METHOD_KEYS)
    else:
        method_keys = [resolve_method(args.method)]

    print(
        f"\n=== {dataset.upper()} VGG16 strict-seed three-regs (L2 / MNE L2 / No reg) ===",
        flush=True,
    )
    print(f"methods={method_keys} seeds={seeds} run_tag={run_tag!r} out={out}", flush=True)

    for method_key in method_keys:
        for seed in seeds:
            rows = run_one(
                dataset,
                method_key,
                seed,
                out,
                run_tag=run_tag,
                retrain=args.retrain,
                force_test=args.force_test,
                ckpt_save_mode=args.ckpt_save_mode,
                first_layer_noise_position=args.first_layer_noise_position,
            )
            upsert_run_rows(raw_csv, method_key, seed, rows)

    finalize_outputs(dataset, out, run_tag, args.font_size, args.legend_font_size)


if __name__ == "__main__":
    main()
