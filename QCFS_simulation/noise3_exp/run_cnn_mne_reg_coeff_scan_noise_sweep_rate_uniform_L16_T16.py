"""
MNIST CNN2：mne_l2 reg_coeff 扫描 + weight_decay 基线 + rate_uniform 噪声注入。

模型：cnn2_c2_c4 / cnn2_c4_c8 / cnn2_c8_c16 / cnn2_c16_c32（可 --arch-list 指定）
默认 rc ∈ {1e-4, 5e-4, 1e-3, 1e-2, 5e-2}；训练 L=16 T=0，测试 L=16 T=16 rate_uniform。

checkpoint 后缀含 rcscan，与 strict-seed 实验区分，不会覆盖已有权重。

用法：
  python noise3_exp/run_cnn_mne_reg_coeff_scan_noise_sweep_rate_uniform_L16_T16.py
  python noise3_exp/run_cnn_mne_reg_coeff_scan_noise_sweep_rate_uniform_L16_T16.py \\
      --arch-list c2c4 c4c8 --seed 42
  python noise3_exp/run_cnn_mne_reg_coeff_scan_noise_sweep_rate_uniform_L16_T16.py \\
      --rc-list 1e-4 1e-3 5e-2 --plot-only
  bash noise3_exp/RUN_cnn_mne_reg_coeff_scan.sh
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "noise3_exp" / "cnn_mne_reg_coeff_scan_noise_sweep_rate_uniform_L16_T16"
REPO_ROOT = ROOT.parent

DEFAULT_SEED = 42
DEFAULT_RC_LIST = [1e-4, 5e-4, 1e-3, 1e-2, 5e-2]
CNN_VARIANTS = [(2, 4), (4, 8), (8, 16), (16, 32)]
LVAL = 16
TVAL = 16
IF_MODE = "rate_uniform"
WD = 5e-4
EPOCHS = int(os.environ.get("CNN_EPOCHS", "100"))
BATCH = int(os.environ.get("CNN_BATCH", "128"))
NUM_WORKERS = int(os.environ.get("CNN_NUM_WORKERS", "8"))
SCAN_TAG = "rcscan"

ARCH_ALIASES = {
    "c2c4": (2, 4),
    "c4c8": (4, 8),
    "c8c16": (8, 16),
    "c16c32": (16, 32),
}

RC_COLORS = {
    "1em04": "#1f77b4",
    "5em04": "#17becf",
    "1em03": "#2ca02c",
    "1em02": "#9467bd",
    "5em02": "#d62728",
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


def method_key(reg: str, reg_coeff: Optional[float] = None) -> str:
    if reg == "weight_decay":
        return "weight_decay"
    return f"mne_l2:{coeff_tag(reg_coeff)}"


def build_suffix(arch: str, reg: str, seed: int, reg_coeff: Optional[float] = None) -> str:
    if reg == "weight_decay":
        return f"seed{seed}_{SCAN_TAG}_wd_l{LVAL}_{arch}"
    return f"seed{seed}_{SCAN_TAG}_mne_l2_l{LVAL}_{arch}_rc{coeff_tag(reg_coeff)}"


def ckpt_path(arch: str, reg: str, seed: int, reg_coeff: Optional[float] = None) -> Path:
    suffix = build_suffix(arch, reg, seed, reg_coeff)
    return ROOT / "mnist-checkpoints" / f"{arch}_L[{LVAL}]_{suffix}.pth"


def test_out_dir(arch: str, tag: str, seed: int) -> Path:
    safe = tag.replace(":", "_")
    return OUT / arch / safe / f"seed_{seed}"


def train_one(
    arch: str, reg: str, seed: int, reg_coeff: Optional[float] = None, retrain: bool = False
) -> Path:
    ckpt = ckpt_path(arch, reg, seed, reg_coeff)
    if retrain and ckpt.exists():
        ckpt.unlink()
        print(f"[RETRAIN] removed {ckpt.name}", flush=True)
    if ckpt.exists():
        print(f"[SKIP TRAIN] {ckpt.name}", flush=True)
        return ckpt

    if reg == "mne_l2":
        regularizer, wd, rc = "mne_l2", 0.0, reg_coeff
    else:
        regularizer, wd, rc = "weight_decay", WD, 1.0

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
        "--suffix", build_suffix(arch, reg, seed, reg_coeff),
    ]
    label = method_key(reg, reg_coeff)
    print(f"[TRAIN] {arch} {label} seed={seed}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
    print(f"[TRAIN DONE] {ckpt.name}", flush=True)
    return ckpt


def test_noise_sweep(
    arch: str,
    tag: str,
    seed: int,
    ckpt: Path,
    force_test: bool = False,
) -> Path:
    out_dir = test_out_dir(arch, tag, seed)
    matrix = (
        out_dir
        / f"noise_sweep_matrix_mnist_{arch}_T{TVAL}_mode_{IF_MODE}_schedule_normal_seed_{seed}.csv"
    )
    if force_test and matrix.exists():
        matrix.unlink()
        combined = out_dir / "noise_sweep_combined_L_T.csv"
        if combined.exists():
            combined.unlink()
    if matrix.exists():
        print(f"[SKIP TEST] {arch} {tag} seed={seed}", flush=True)
        return matrix

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
    print(f"[TEST] {arch} {tag} seed={seed} mode={IF_MODE}", flush=True)
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


def curve_acc(curve: list[tuple[float, float]], sigma: float) -> float:
    d = dict(curve)
    return float(d.get(sigma, d.get(int(sigma) if sigma == int(sigma) else sigma)))


def line_style_for(key: str, rc_list: list[float]) -> dict:
    if key == "weight_decay":
        return {"color": "#ff7f0e", "label": "weight_decay"}
    rc_tag = key.split(":", 1)[1]
    for coeff in rc_list:
        if coeff_tag(coeff) == rc_tag:
            return {
                "color": RC_COLORS.get(rc_tag, "#333333"),
                "label": f"mne_l2 rc={coeff:.0e}",
            }
    return {"color": "#333333", "label": key}


def plot_arch(
    arch: str,
    curves: dict[str, list[tuple[float, float]]],
    rc_list: list[float],
    seed: int,
    font_size: float,
    legend_font_size: float,
) -> None:
    plt.rcParams.update({"font.size": font_size, "legend.fontsize": legend_font_size})
    order = ["weight_decay"] + [method_key("mne_l2", c) for c in rc_list]
    for no_caption in (False, True):
        fig, ax = plt.subplots(figsize=(10.0, 6.2), dpi=180)
        all_y: list[float] = []
        for key in order:
            if key not in curves:
                continue
            pts = sorted(curves[key], key=lambda x: x[0])
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            all_y.extend(y)
            style = line_style_for(key, rc_list)
            ax.plot(
                x, y, marker="o", linewidth=2.2, markersize=5,
                color=style["color"], label=style["label"],
            )
        ax.set_xlabel("Gaussian noise sigma")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([round(i * 0.1, 1) for i in range(11)])
        if all_y:
            ax.set_ylim(min(all_y) - 1.0, max(all_y) + 1.0)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", frameon=False, ncol=2)
        if not no_caption:
            ax.set_title(
                f"MNIST {arch} mne_l2 rc scan vs wd "
                f"(seed={seed}, L={LVAL}, T={TVAL}, {IF_MODE})"
            )
        fig.tight_layout()
        suffix = "_no_caption" if no_caption else ""
        out_png = OUT / f"{arch}_mne_reg_coeff_scan_noise_sweep{suffix}.png"
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] saved {out_png}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MNIST CNN mne_l2 rc scan + noise sweep")
    p.add_argument(
        "--arch-list",
        nargs="+",
        default=None,
        help="c2c4 c4c8 c8c16 c16c32（默认四个全跑）",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--rc-list",
        type=float,
        nargs="+",
        default=None,
        help=f"reg_coeff 列表（默认 {' '.join(f'{x:.0e}' for x in DEFAULT_RC_LIST)}）",
    )
    p.add_argument("--skip-wd", action="store_true", help="跳过 weight_decay 基线")
    p.add_argument("--retrain", action="store_true", help="删除已有 checkpoint 后重训")
    p.add_argument("--force-test", action="store_true", help="删除已有 matrix 后重测")
    p.add_argument("--plot-only", action="store_true", help="仅从已有 matrix 汇总出图")
    p.add_argument("--font-size", type=float, default=11.0)
    p.add_argument("--legend-font-size", type=float, default=9.0)
    p.add_argument(
        "--copy-root",
        action="store_true",
        help="复制各 arch 的 no_caption 图到仓库根目录",
    )
    return p.parse_args()


def load_existing_records(
    arch: str, seed: int, rc_list: list[float], include_wd: bool
) -> list[dict]:
    records: list[dict] = []
    if include_wd:
        tag = "weight_decay"
        mat = test_out_dir(arch, tag, seed) / (
            f"noise_sweep_matrix_mnist_{arch}_T{TVAL}_mode_{IF_MODE}_schedule_normal_seed_{seed}.csv"
        )
        if mat.exists():
            records.append(
                {
                    "arch": arch,
                    "method": tag,
                    "regularizer": "weight_decay",
                    "reg_coeff": "",
                    "checkpoint": "",
                    "matrix_csv": str(mat.relative_to(ROOT)),
                    "curve": read_matrix(mat),
                }
            )
    for coeff in rc_list:
        tag = method_key("mne_l2", coeff)
        mat = test_out_dir(arch, tag, seed) / (
            f"noise_sweep_matrix_mnist_{arch}_T{TVAL}_mode_{IF_MODE}_schedule_normal_seed_{seed}.csv"
        )
        if mat.exists():
            records.append(
                {
                    "arch": arch,
                    "method": tag,
                    "regularizer": "mne_l2",
                    "reg_coeff": coeff,
                    "checkpoint": "",
                    "matrix_csv": str(mat.relative_to(ROOT)),
                    "curve": read_matrix(mat),
                }
            )
    return records


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    arch_pairs = resolve_arch_list(args.arch_list)
    rc_list = args.rc_list if args.rc_list is not None else list(DEFAULT_RC_LIST)
    seed = args.seed
    include_wd = not args.skip_wd

    all_records: list[dict] = []

    for c1, c2 in arch_pairs:
        arch = arch_name(c1, c2)
        print(f"\n=== {arch} rc scan seed={seed} ===", flush=True)

        if args.plot_only:
            records = load_existing_records(arch, seed, rc_list, include_wd)
            if not records:
                print(f"[WARN] {arch}: 无 matrix，跳过", flush=True)
                continue
        else:
            records = []
            if include_wd:
                wd_ckpt = train_one(arch, "weight_decay", seed, retrain=args.retrain)
                wd_mat = test_noise_sweep(
                    arch, "weight_decay", seed, wd_ckpt, force_test=args.force_test
                )
                records.append(
                    {
                        "arch": arch,
                        "method": "weight_decay",
                        "regularizer": "weight_decay",
                        "reg_coeff": "",
                        "checkpoint": str(wd_ckpt.relative_to(ROOT)),
                        "matrix_csv": str(wd_mat.relative_to(ROOT)),
                        "curve": read_matrix(wd_mat),
                    }
                )
            for coeff in rc_list:
                ckpt = train_one(
                    arch, "mne_l2", seed, reg_coeff=coeff, retrain=args.retrain
                )
                tag = method_key("mne_l2", coeff)
                mat = test_noise_sweep(
                    arch, tag, seed, ckpt, force_test=args.force_test
                )
                records.append(
                    {
                        "arch": arch,
                        "method": tag,
                        "regularizer": "mne_l2",
                        "reg_coeff": coeff,
                        "checkpoint": str(ckpt.relative_to(ROOT)),
                        "matrix_csv": str(mat.relative_to(ROOT)),
                        "curve": read_matrix(mat),
                    }
                )

        all_records.extend(records)
        curves = {r["method"]: r["curve"] for r in records}
        if curves:
            plot_arch(arch, curves, rc_list, seed, args.font_size, args.legend_font_size)

        print(f"\n--- {arch} summary (seed={seed}) ---", flush=True)
        print(f"{'method':<22} {'acc@0':>8} {'acc@1':>8} {'drop':>8}", flush=True)
        for rec in records:
            a0 = curve_acc(rec["curve"], 0.0)
            a1 = curve_acc(rec["curve"], 1.0)
            rc_str = f"{rec['reg_coeff']:.0e}" if rec["reg_coeff"] != "" else "wd"
            print(
                f"{rec['method']:<22} {a0:>8.3f} {a1:>8.3f} {a0 - a1:>8.3f}  ({rc_str})",
                flush=True,
            )

    if not all_records:
        raise SystemExit("无可用结果，请先运行训练+测试或检查 OUT 目录")

    raw_rows = []
    summary_rows = []
    for rec in all_records:
        a0 = curve_acc(rec["curve"], 0.0)
        a1 = curve_acc(rec["curve"], 1.0)
        rc_str = f"{rec['reg_coeff']:.0e}" if rec["reg_coeff"] != "" else ""
        summary_rows.append(
            {
                "arch": rec["arch"],
                "method": rec["method"],
                "regularizer": rec["regularizer"],
                "reg_coeff": rc_str,
                "acc_sigma0": f"{a0:.6f}",
                "acc_sigma1": f"{a1:.6f}",
                "acc_drop_sigma0_to_1": f"{a0 - a1:.6f}",
                "checkpoint": rec["checkpoint"],
                "matrix_csv": rec["matrix_csv"],
            }
        )
        for sigma, acc in rec["curve"]:
            raw_rows.append(
                {
                    "arch": rec["arch"],
                    "method": rec["method"],
                    "regularizer": rec["regularizer"],
                    "reg_coeff": rc_str,
                    "seed": seed,
                    "L": LVAL,
                    "T": TVAL,
                    "if_mode": IF_MODE,
                    "sigma": sigma,
                    "acc": acc,
                    "checkpoint": rec["checkpoint"],
                    "matrix_csv": rec["matrix_csv"],
                }
            )

    raw_csv = OUT / "cnn_mne_reg_coeff_scan_noise_sweep_raw.csv"
    summary_csv = OUT / "cnn_mne_reg_coeff_scan_best_test_summary.csv"
    with raw_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "arch", "method", "regularizer", "reg_coeff", "seed", "L", "T",
                "if_mode", "sigma", "acc", "checkpoint", "matrix_csv",
            ],
        )
        w.writeheader()
        w.writerows(raw_rows)
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "arch", "method", "regularizer", "reg_coeff",
                "acc_sigma0", "acc_sigma1", "acc_drop_sigma0_to_1",
                "checkpoint", "matrix_csv",
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n[DONE] raw: {raw_csv}", flush=True)
    print(f"[DONE] summary: {summary_csv}", flush=True)

    if args.copy_root:
        for c1, c2 in arch_pairs:
            arch = arch_name(c1, c2)
            src = OUT / f"{arch}_mne_reg_coeff_scan_noise_sweep_no_caption.png"
            if src.exists():
                dest = REPO_ROOT / src.name
                if dest.resolve() != src.resolve():
                    shutil.copy2(src, dest)
                    print(f"[PLOT] copied {dest}", flush=True)


if __name__ == "__main__":
    main()
