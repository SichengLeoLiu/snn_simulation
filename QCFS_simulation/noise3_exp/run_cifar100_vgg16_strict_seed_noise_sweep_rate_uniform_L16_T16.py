"""
CIFAR-100 VGG16 严格多 seed 高斯噪声注入：mne_l2 vs weight_decay。
设置：L=16, T=16, IF mode=rate_uniform, sigma=0.0~1.0 step=0.1。
每个 seed 独立训练，再用对应 checkpoint 做噪声扫描。
"""
import csv
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "noise3_exp" / "cifar100_vgg16_strict_seed_noise_sweep_rate_uniform_L16_T16"
OUT.mkdir(parents=True, exist_ok=True)

ARCH = "vgg16"
DATASET = "cifar100"
SEEDS = [40, 41, 42, 43, 44]
REGS = ["mne_l2", "weight_decay"]
LVAL = 16
TVAL = 16
IF_MODE = "rate_uniform"
EPOCHS = 300
LR = 0.1
BATCH = 128


def coeff_tag(v: float) -> str:
    return f"{v:.0e}".replace("-", "m").replace("+", "p")


def build_suffix(reg: str, seed: int) -> str:
    if reg == "weight_decay":
        return f"strict_seed{seed}_ablation_wd_l{LVAL}_{ARCH}"
    return f"strict_seed{seed}_ablation_mne_l2_l{LVAL}_{ARCH}_rc{coeff_tag(5e-2)}"


def ckpt_path(reg: str, seed: int) -> Path:
    suffix = build_suffix(reg, seed)
    return ROOT / f"{DATASET}-checkpoints" / f"{ARCH}_L[{LVAL}]_{suffix}.pth"


def train_one(reg: str, seed: int) -> Path:
    ckpt = ckpt_path(reg, seed)
    if ckpt.exists():
        print(f"[SKIP TRAIN] {ckpt.name}", flush=True)
        return ckpt

    if reg == "mne_l2":
        regularizer, wd, rc = "mne_l2", 0.0, 5e-2
    else:
        regularizer, wd, rc = "weight_decay", 5e-4, 1.0

    cmd = [
        sys.executable, str(ROOT / "main_train.py"),
        "-data", DATASET,
        "-arch", ARCH,
        "-L", str(LVAL),
        "--epochs", str(EPOCHS),
        "-lr", str(LR),
        "-j", "8",
        "-b", str(BATCH),
        "--seed", str(seed),
        "--device", "auto",
        "--time", "0",
        "--spike_schedule", "normal",
        "--regularizer", regularizer,
        "--weight_decay", str(wd),
        "--reg_coeff", str(rc),
        "--suffix", build_suffix(reg, seed),
    ]
    print(f"[TRAIN] {DATASET} {ARCH} {reg} seed={seed}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
    print(f"[TRAIN DONE] {ckpt.name}", flush=True)
    return ckpt


def test_noise_sweep(reg: str, seed: int, ckpt: Path) -> Path:
    out_dir = OUT / reg / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = (
        out_dir
        / f"noise_sweep_matrix_{DATASET}_{ARCH}_T{TVAL}_mode_{IF_MODE}_schedule_normal_seed_{seed}.csv"
    )
    if matrix.exists():
        print(f"[SKIP TEST] {reg} seed={seed}", flush=True)
        return matrix

    cmd = [
        sys.executable, str(ROOT / "main_test.py"),
        "-data", DATASET,
        "-arch", ARCH,
        "-L", str(LVAL),
        "-T", str(TVAL),
        "-j", "8",
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
    print(f"[TEST] {reg} seed={seed} mode={IF_MODE}", flush=True)
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


def plot_results(agg_rows: list[dict]) -> None:
    colors = {"mne_l2": "#1f77b4", "weight_decay": "#ff7f0e"}
    labels = {"mne_l2": "mne_l2 (mean)", "weight_decay": "weight_decay (mean)"}
    plt.rcParams.update({"font.size": 11, "legend.fontsize": 10})
    for no_caption in (False, True):
        fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=180)
        all_y = []
        for reg in REGS:
            rr = [r for r in agg_rows if r["regularizer"] == reg]
            rr.sort(key=lambda x: float(x["sigma"]))
            x = [float(r["sigma"]) for r in rr]
            y = [float(r["acc_mean"]) for r in rr]
            s = [float(r["acc_std"]) for r in rr]
            all_y.extend([yy - ss for yy, ss in zip(y, s)])
            all_y.extend([yy + ss for yy, ss in zip(y, s)])
            ax.plot(x, y, marker="o", linewidth=2.2, markersize=5, color=colors[reg], label=labels[reg])
            ax.fill_between(
                x,
                [yy - ss for yy, ss in zip(y, s)],
                [yy + ss for yy, ss in zip(y, s)],
                color=colors[reg],
                alpha=0.18,
                linewidth=0,
            )
        ax.set_xlabel("Gaussian noise sigma")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([round(i * 0.1, 1) for i in range(11)])
        ax.set_ylim(min(all_y) - 1.0, max(all_y) + 1.0)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", frameon=False)
        if not no_caption:
            ax.set_title(
                f"CIFAR-100 {ARCH} strict-seed noise sweep (L={LVAL},T={TVAL},{IF_MODE})"
            )
        fig.tight_layout()
        suffix = "_no_caption" if no_caption else ""
        fig.savefig(OUT / f"cifar100_{ARCH}_noise_sweep_mean_std_lineplot{suffix}.png")
        plt.close(fig)


def main() -> None:
    records = []
    for reg in REGS:
        for seed in SEEDS:
            ckpt = train_one(reg, seed)
            mat = test_noise_sweep(reg, seed, ckpt)
            records.append((reg, seed, ckpt, mat))

    raw_rows = []
    for reg, seed, ckpt, mat in records:
        with mat.open() as f:
            reader = csv.reader(f)
            header = next(reader)
            row = next(reader)
        start = 2 if header[:2] == ["L", "T"] else 1
        sigmas = [float(x) for x in header[start:]]
        accs = [float(x) for x in row[start:]]
        for sg, ac in zip(sigmas, accs):
            raw_rows.append({
                "dataset": DATASET,
                "arch": ARCH,
                "regularizer": reg,
                "seed": seed,
                "L": LVAL,
                "T": TVAL,
                "if_mode": IF_MODE,
                "sigma": sg,
                "acc": ac,
                "checkpoint": str(ckpt.relative_to(ROOT)),
                "matrix_csv": str(mat.relative_to(ROOT)),
            })

    raw_csv = OUT / "cifar100_vgg16_strict_seed_noise_sweep_raw.csv"
    with raw_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "dataset", "arch", "regularizer", "seed", "L", "T",
                "if_mode", "sigma", "acc", "checkpoint", "matrix_csv",
            ],
        )
        w.writeheader()
        w.writerows(raw_rows)

    bucket = defaultdict(list)
    for r in raw_rows:
        bucket[(r["regularizer"], r["sigma"])].append(float(r["acc"]))

    agg_rows = []
    for (reg, sigma), vals in sorted(bucket.items(), key=lambda x: (x[0][0], x[0][1])):
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        agg_rows.append({
            "regularizer": reg,
            "sigma": f"{sigma:.1f}",
            "acc_mean": f"{mean:.6f}",
            "acc_std": f"{std:.6f}",
            "n_seeds": len(vals),
        })

    agg_csv = OUT / "cifar100_vgg16_strict_seed_noise_sweep_mean_std.csv"
    with agg_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["regularizer", "sigma", "acc_mean", "acc_std", "n_seeds"]
        )
        w.writeheader()
        w.writerows(agg_rows)

    plot_results(agg_rows)
    print(f"[DONE] raw: {raw_csv}", flush=True)
    print(f"[DONE] agg: {agg_csv}", flush=True)
    print(f"[DONE] plot: {OUT / 'cifar100_vgg16_noise_sweep_mean_std_lineplot_no_caption.png'}", flush=True)


if __name__ == "__main__":
    main()
