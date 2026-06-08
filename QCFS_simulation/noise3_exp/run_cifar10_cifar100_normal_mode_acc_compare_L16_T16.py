"""
CIFAR-10 / CIFAR-100 VGG16：normal 模式下三路方法精度对比。

复用 rate_uniform 实验已训练的 checkpoint（不重新训练）：
  - weight_decay (wd=5e-4)
  - mne_l2 (rc=1e-4)
  - mne_l2+wd (rc=1e-4, wd=1e-4)

在 normal 模式下做噪声扫描 (sigma=0~1.0)，输出：
  - 各数据集折线图
  - CIFAR-10 vs CIFAR-100 acc 对比表与柱状图
"""
import csv
import subprocess
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "noise3_exp" / "cifar10_cifar100_vgg16_normal_mode_acc_compare_L16_T16"
OUT.mkdir(parents=True, exist_ok=True)

ARCH = "vgg16"
DATASETS = ["cifar10", "cifar100"]
SEED = 42
LVAL = 16
TVAL = 16
IF_MODE = "normal"
BATCH = 128
SCHEME_TAG = "schemeC_noout"

MNE_RC = 1e-4
WD_BASE = 5e-4
WD_COMBO = 1e-4

EXPERIMENTS = [
    ("weight_decay", None, WD_BASE),
    ("mne_l2 rc=1e-4", MNE_RC, 0.0),
    ("mne_l2+wd rc=1e-4 wd=1e-4", MNE_RC, WD_COMBO),
]

LINE_STYLES = {
    "weight_decay": {"color": "#ff7f0e", "label": "weight_decay (wd=5e-4)"},
    "mne_l2 rc=1e-4": {"color": "#1f77b4", "label": "mne_l2 (rc=1e-4)"},
    "mne_l2+wd rc=1e-4 wd=1e-4": {
        "color": "#2ca02c",
        "label": "mne_l2+wd (rc=1e-4, wd=1e-4)",
    },
}


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


def ckpt_path(dataset: str, reg_coeff: Optional[float], wd: float) -> Path:
    suffix = build_suffix(reg_coeff, wd)
    return ROOT / f"{dataset}-checkpoints" / f"{ARCH}_L[{LVAL}]_{suffix}.pth"


def test_noise_sweep(
    dataset: str, label: str, ckpt: Path, reg_coeff: Optional[float], wd: float
) -> Path:
    safe = f"{dataset}_{label}".replace(" ", "_").replace("=", "")
    out_dir = OUT / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = (
        out_dir
        / f"noise_sweep_matrix_{dataset}_{ARCH}_T{TVAL}_mode_{IF_MODE}_schedule_normal_seed_{SEED}.csv"
    )
    if matrix.exists():
        print(f"[SKIP TEST] {dataset} {label}", flush=True)
        return matrix

    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")

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
    print(f"[TEST] {dataset} {label} mode={IF_MODE}", flush=True)
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


def plot_per_dataset(
    dataset: str, curves: dict[str, list[tuple[float, float]]]
) -> None:
    plt.rcParams.update({"font.size": 11, "legend.fontsize": 10})
    order = [exp[0] for exp in EXPERIMENTS]

    for no_caption in (False, True):
        fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=180)
        all_y = []
        for label in order:
            if label not in curves:
                continue
            pts = sorted(curves[label], key=lambda x: x[0])
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            all_y.extend(y)
            style = LINE_STYLES.get(label, {"color": "#333333", "label": label})
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
        ax.legend(loc="lower left", frameon=False)
        if not no_caption:
            ax.set_title(
                f"{dataset} {ARCH} normal mode (L={LVAL}, T={TVAL}, seed={SEED})"
            )
        fig.tight_layout()
        suffix = "_no_caption" if no_caption else ""
        fig.savefig(OUT / f"{dataset}_{ARCH}_normal_mode_noise_sweep{suffix}.png")
        plt.close(fig)


def plot_acc_compare_bar(summary_rows: list[dict]) -> None:
    """CIFAR-10 vs CIFAR-100 acc@sigma=0 分组柱状图。"""
    labels = [exp[0] for exp in EXPERIMENTS]
    short = {
        "weight_decay": "wd",
        "mne_l2 rc=1e-4": "mne_l2",
        "mne_l2+wd rc=1e-4 wd=1e-4": "mne_l2+wd",
    }
    x = np.arange(len(labels))
    width = 0.35

    c10 = []
    c100 = []
    for label in labels:
        r10 = next(r for r in summary_rows if r["dataset"] == "cifar10" and r["label"] == label)
        r100 = next(r for r in summary_rows if r["dataset"] == "cifar100" and r["label"] == label)
        c10.append(float(r10["acc_sigma0"]))
        c100.append(float(r100["acc_sigma0"]))

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=180)
    ax.bar(x - width / 2, c10, width, label="CIFAR-10", color="#4c72b0")
    ax.bar(x + width / 2, c100, width, label="CIFAR-100", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels([short.get(l, l) for l in labels])
    ax.set_ylabel("Accuracy @ sigma=0 (%)")
    ax.set_title(f"normal mode acc compare (L={LVAL}, T={TVAL}, seed={SEED})")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    for i, (a10, a100) in enumerate(zip(c10, c100)):
        ax.text(i - width / 2, a10 + 0.3, f"{a10:.1f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, a100 + 0.3, f"{a100:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "cifar10_cifar100_normal_mode_acc_sigma0_bar.png")
    plt.close(fig)


def main() -> None:
    all_records: list[dict] = []

    for dataset in DATASETS:
        ds_curves: dict[str, list[tuple[float, float]]] = {}
        for label, reg_coeff, wd in EXPERIMENTS:
            ckpt = ckpt_path(dataset, reg_coeff, wd)
            mat = test_noise_sweep(dataset, label, ckpt, reg_coeff, wd)
            curve = read_matrix(mat)
            ds_curves[label] = curve
            all_records.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "reg_coeff": reg_coeff if reg_coeff is not None else "",
                    "weight_decay": wd,
                    "checkpoint": str(ckpt.relative_to(ROOT)),
                    "matrix_csv": str(mat.relative_to(ROOT)),
                    "curve": curve,
                }
            )
        plot_per_dataset(dataset, ds_curves)

    raw_rows = []
    for rec in all_records:
        for sigma, acc in rec["curve"]:
            raw_rows.append(
                {
                    "dataset": rec["dataset"],
                    "label": rec["label"],
                    "reg_coeff": rec["reg_coeff"],
                    "weight_decay": rec["weight_decay"],
                    "seed": SEED,
                    "L": LVAL,
                    "T": TVAL,
                    "if_mode": IF_MODE,
                    "sigma": sigma,
                    "acc": acc,
                    "checkpoint": rec["checkpoint"],
                    "matrix_csv": rec["matrix_csv"],
                }
            )

    raw_csv = OUT / "cifar10_cifar100_normal_mode_noise_sweep_raw.csv"
    with raw_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "dataset", "label", "reg_coeff", "weight_decay", "seed",
                "L", "T", "if_mode", "sigma", "acc", "checkpoint", "matrix_csv",
            ],
        )
        w.writeheader()
        w.writerows(raw_rows)

    summary_rows = []
    print(f"\n========== normal mode acc @ sigma=0 ==========", flush=True)
    print(f"{'dataset':<10} {'label':<32} {'acc@sigma=0':>12}", flush=True)
    print("-" * 56, flush=True)
    for rec in all_records:
        curve = dict(rec["curve"])
        acc0 = curve.get(0.0, curve.get(0))
        print(f"{rec['dataset']:<10} {rec['label']:<32} {acc0:>12.3f}", flush=True)
        rc_str = f"{rec['reg_coeff']:.0e}" if rec["reg_coeff"] != "" else ""
        wd_str = f"{rec['weight_decay']:.0e}" if rec["weight_decay"] > 0 else "0"
        summary_rows.append(
            {
                "dataset": rec["dataset"],
                "label": rec["label"],
                "reg_coeff": rc_str,
                "weight_decay": wd_str,
                "acc_sigma0": f"{acc0:.6f}",
                "checkpoint": rec["checkpoint"],
            }
        )

    compare_csv = OUT / "cifar10_cifar100_normal_mode_acc_compare.csv"
    with compare_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "dataset", "label", "reg_coeff", "weight_decay",
                "acc_sigma0", "checkpoint",
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)

    # pivot table for easy reading
    pivot_csv = OUT / "cifar10_cifar100_normal_mode_acc_pivot.csv"
    with pivot_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "cifar10_acc_sigma0", "cifar100_acc_sigma0", "delta_c10_minus_c100"])
        for label, _, _ in EXPERIMENTS:
            a10 = float(next(r["acc_sigma0"] for r in summary_rows if r["dataset"] == "cifar10" and r["label"] == label))
            a100 = float(next(r["acc_sigma0"] for r in summary_rows if r["dataset"] == "cifar100" and r["label"] == label))
            w.writerow([label, f"{a10:.6f}", f"{a100:.6f}", f"{a10 - a100:.6f}"])

    plot_acc_compare_bar(summary_rows)

    print(f"\n[DONE] raw: {raw_csv}", flush=True)
    print(f"[DONE] compare: {compare_csv}", flush=True)
    print(f"[DONE] pivot: {pivot_csv}", flush=True)
    print(f"[DONE] bar: {OUT / 'cifar10_cifar100_normal_mode_acc_sigma0_bar.png'}", flush=True)
    for ds in DATASETS:
        print(
            f"[DONE] {ds} plot: {OUT / f'{ds}_{ARCH}_normal_mode_noise_sweep_no_caption.png'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
