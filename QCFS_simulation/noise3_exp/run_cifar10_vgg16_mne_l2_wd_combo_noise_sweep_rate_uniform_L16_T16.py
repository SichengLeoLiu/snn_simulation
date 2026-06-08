"""
CIFAR-10 VGG16：mne_l2 / weight_decay / mne_l2+wd 三路对比 + 噪声注入。

参数来自 CIFAR-100 最优组合：mne_l2 rc=1e-4 + wd=1e-4。
对比：
  - weight_decay only (wd=5e-4)
  - mne_l2 only (rc=1e-4)
  - mne_l2+wd (rc=1e-4, wd=1e-4)
方案 C：输出层（无 IF）不参与 MNE-L2。
"""
import csv
import subprocess
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "noise3_exp"
    / "cifar10_vgg16_mne_l2_wd_combo_noise_sweep_rate_uniform_L16_T16"
)
OUT.mkdir(parents=True, exist_ok=True)

ARCH = "vgg16"
DATASET = "cifar10"
SEED = 42
LVAL = 16
TVAL = 16
IF_MODE = "rate_uniform"
EPOCHS = 300
LR = 0.1
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


def ckpt_path(reg_coeff: Optional[float], wd: float) -> Path:
    suffix = build_suffix(reg_coeff, wd)
    return ROOT / f"{DATASET}-checkpoints" / f"{ARCH}_L[{LVAL}]_{suffix}.pth"


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

    print(f"[TRAIN] {label}", flush=True)
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


def plot_results(curves: dict[str, list[tuple[float, float]]]) -> None:
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
                f"CIFAR-10 {ARCH} wd vs mne_l2 vs mne_l2+wd "
                f"(seed={SEED}, L={LVAL}, T={TVAL}, {IF_MODE}, {SCHEME_TAG})"
            )
        fig.tight_layout()
        suffix = "_no_caption" if no_caption else ""
        fig.savefig(OUT / f"cifar10_{ARCH}_mne_l2_wd_combo_noise_sweep{suffix}.png")
        plt.close(fig)


def main() -> None:
    records: list[dict] = []

    for label, reg_coeff, wd in EXPERIMENTS:
        ckpt = train_one(label, reg_coeff, wd)
        mat = test_noise_sweep(label, ckpt)
        records.append(
            {
                "label": label,
                "reg_coeff": reg_coeff if reg_coeff is not None else "",
                "weight_decay": wd,
                "checkpoint": str(ckpt.relative_to(ROOT)),
                "matrix_csv": str(mat.relative_to(ROOT)),
                "curve": read_matrix(mat),
            }
        )

    raw_rows = []
    for rec in records:
        for sigma, acc in rec["curve"]:
            raw_rows.append(
                {
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

    raw_csv = OUT / "cifar10_vgg16_mne_l2_wd_combo_noise_sweep_raw.csv"
    with raw_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        w.writeheader()
        w.writerows(raw_rows)

    curves = {rec["label"]: rec["curve"] for rec in records}
    plot_results(curves)

    summary_rows = []
    print("\n========== Test results (sigma=0, rate_uniform) ==========", flush=True)
    print(f"{'label':<32} {'acc@sigma=0':>12}", flush=True)
    print("-" * 46, flush=True)
    for rec in records:
        curve = dict(rec["curve"])
        acc0 = curve.get(0.0, curve.get(0))
        print(f"{rec['label']:<32} {acc0:>12.3f}", flush=True)
        rc_str = f"{rec['reg_coeff']:.0e}" if rec["reg_coeff"] != "" else ""
        wd_str = f"{rec['weight_decay']:.0e}" if rec["weight_decay"] > 0 else "0"
        summary_rows.append(
            {
                "label": rec["label"],
                "reg_coeff": rc_str,
                "weight_decay": wd_str,
                "acc_sigma0": f"{acc0:.6f}",
                "checkpoint": rec["checkpoint"],
            }
        )

    best_clean = max(summary_rows, key=lambda r: float(r["acc_sigma0"]))
    print("-" * 46, flush=True)
    print(
        f"Best sigma=0: {best_clean['label']}  acc={best_clean['acc_sigma0']}",
        flush=True,
    )

    summary_csv = OUT / "cifar10_vgg16_mne_l2_wd_combo_best_test_summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["label", "reg_coeff", "weight_decay", "acc_sigma0", "checkpoint"],
        )
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n[DONE] raw: {raw_csv}", flush=True)
    print(f"[DONE] summary: {summary_csv}", flush=True)
    print(
        f"[DONE] plot: {OUT / 'cifar10_vgg16_mne_l2_wd_combo_noise_sweep_no_caption.png'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
