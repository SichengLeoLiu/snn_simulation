"""
CIFAR-100 VGG16：mne_l2 reg_coeff 扫描 + weight_decay 对比 + 噪声注入。

方案 C：输出层（无 IF）不参与 MNE-L2（见 utils.compute_mne_l2_regularization）。
扫描 reg_coeff ∈ {1e-4, 1e-3, 3e-3, 3e-2}，每个训练后做 rate_uniform 噪声扫描，
并与 weight_decay 基线画在同一张折线图；结尾输出各模型 sigma=0 最佳测试结果。
"""
import csv
import subprocess
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "noise3_exp" / "cifar100_vgg16_mne_reg_coeff_scan_noise_sweep_rate_uniform_L16_T16"
OUT.mkdir(parents=True, exist_ok=True)

ARCH = "vgg16"
DATASET = "cifar100"
SEED = 42
MNE_REG_COEFFS = [1e-4, 1e-3, 3e-3, 3e-2]
LVAL = 16
TVAL = 16
IF_MODE = "rate_uniform"
EPOCHS = 300
LR = 0.1
BATCH = 128
WD = 5e-4

# 与方案 C 后的 checkpoint 命名区分，避免误用旧权重
SCHEME_TAG = "schemeC_noout"

LINE_STYLES = {
    "weight_decay": {"color": "#ff7f0e", "label": "weight_decay"},
    "mne_l2:1em04": {"color": "#1f77b4", "label": "mne_l2 rc=1e-4"},
    "mne_l2:1em03": {"color": "#2ca02c", "label": "mne_l2 rc=1e-3"},
    "mne_l2:3em03": {"color": "#9467bd", "label": "mne_l2 rc=3e-3"},
    "mne_l2:3em02": {"color": "#d62728", "label": "mne_l2 rc=3e-2"},
}


def coeff_tag(v: float) -> str:
    return f"{v:.0e}".replace("-", "m").replace("+", "p")


def method_key(reg: str, reg_coeff: Optional[float] = None) -> str:
    if reg == "weight_decay":
        return "weight_decay"
    return f"mne_l2:{coeff_tag(reg_coeff)}"


def build_suffix(reg: str, reg_coeff: Optional[float] = None) -> str:
    if reg == "weight_decay":
        return f"seed{SEED}_{SCHEME_TAG}_wd_l{LVAL}_{ARCH}"
    return (
        f"seed{SEED}_{SCHEME_TAG}_mne_l2_l{LVAL}_{ARCH}_rc{coeff_tag(reg_coeff)}"
    )


def ckpt_path(reg: str, reg_coeff: Optional[float] = None) -> Path:
    suffix = build_suffix(reg, reg_coeff)
    return ROOT / f"{DATASET}-checkpoints" / f"{ARCH}_L[{LVAL}]_{suffix}.pth"


def train_one(reg: str, reg_coeff: Optional[float] = None) -> Path:
    ckpt = ckpt_path(reg, reg_coeff)
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
        build_suffix(reg, reg_coeff),
    ]
    if reg == "mne_l2":
        cmd.append("--mne_detach_lambda")

    tag = method_key(reg, reg_coeff)
    print(f"[TRAIN] {tag}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
    print(f"[TRAIN DONE] {ckpt.name}", flush=True)
    return ckpt


def test_noise_sweep(reg: str, ckpt: Path, reg_coeff: Optional[float] = None) -> Path:
    tag = method_key(reg, reg_coeff)
    safe_tag = tag.replace(":", "_")
    out_dir = OUT / safe_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = (
        out_dir
        / f"noise_sweep_matrix_{DATASET}_{ARCH}_T{TVAL}_mode_{IF_MODE}_schedule_normal_seed_{SEED}.csv"
    )
    if matrix.exists():
        print(f"[SKIP TEST] {tag}", flush=True)
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
    print(f"[TEST] {tag} mode={IF_MODE}", flush=True)
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
    plt.rcParams.update({"font.size": 11, "legend.fontsize": 9})
    order = ["weight_decay"] + [method_key("mne_l2", c) for c in MNE_REG_COEFFS]

    for no_caption in (False, True):
        fig, ax = plt.subplots(figsize=(10.0, 6.0), dpi=180)
        all_y = []
        for key in order:
            if key not in curves:
                continue
            pts = sorted(curves[key], key=lambda x: x[0])
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            all_y.extend(y)
            style = LINE_STYLES.get(key, {"color": "#333333", "label": key})
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.0,
                markersize=4,
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
        ax.legend(loc="lower left", frameon=False, ncol=2)
        if not no_caption:
            ax.set_title(
                f"CIFAR-100 {ARCH} mne_l2 reg_coeff scan vs wd "
                f"(seed={SEED}, L={LVAL}, T={TVAL}, {IF_MODE}, {SCHEME_TAG})"
            )
        fig.tight_layout()
        suffix = "_no_caption" if no_caption else ""
        fig.savefig(OUT / f"cifar100_{ARCH}_mne_reg_coeff_scan_noise_sweep{suffix}.png")
        plt.close(fig)


def main() -> None:
    records: list[dict] = []

    # weight_decay baseline
    wd_ckpt = train_one("weight_decay")
    wd_mat = test_noise_sweep("weight_decay", wd_ckpt)
    records.append(
        {
            "method": "weight_decay",
            "regularizer": "weight_decay",
            "reg_coeff": "",
            "checkpoint": str(wd_ckpt.relative_to(ROOT)),
            "matrix_csv": str(wd_mat.relative_to(ROOT)),
            "curve": read_matrix(wd_mat),
        }
    )

    # mne_l2 scan
    for coeff in MNE_REG_COEFFS:
        ckpt = train_one("mne_l2", coeff)
        mat = test_noise_sweep("mne_l2", ckpt, coeff)
        records.append(
            {
                "method": method_key("mne_l2", coeff),
                "regularizer": "mne_l2",
                "reg_coeff": coeff,
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
                    "method": rec["method"],
                    "regularizer": rec["regularizer"],
                    "reg_coeff": rec["reg_coeff"],
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

    raw_csv = OUT / "cifar100_vgg16_mne_reg_coeff_scan_noise_sweep_raw.csv"
    with raw_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "regularizer",
                "reg_coeff",
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

    curves = {rec["method"]: rec["curve"] for rec in records}
    plot_results(curves)

    summary_rows = []
    print("\n========== Best test results (sigma=0, rate_uniform) ==========", flush=True)
    print(f"{'method':<22} {'reg_coeff':<10} {'acc@sigma=0':>12}", flush=True)
    print("-" * 48, flush=True)
    for rec in records:
        curve = dict(rec["curve"])
        best_acc = curve.get(0.0, curve.get(0))
        rc_str = f"{rec['reg_coeff']:.0e}" if rec["reg_coeff"] != "" else "5e-4(wd)"
        print(f"{rec['method']:<22} {rc_str:<10} {best_acc:>12.3f}", flush=True)
        summary_rows.append(
            {
                "method": rec["method"],
                "regularizer": rec["regularizer"],
                "reg_coeff": rc_str,
                "acc_sigma0": f"{best_acc:.6f}",
                "checkpoint": rec["checkpoint"],
            }
        )

    best_overall = max(summary_rows, key=lambda r: float(r["acc_sigma0"]))
    print("-" * 48, flush=True)
    print(
        f"Best overall: {best_overall['method']}  acc@sigma=0={best_overall['acc_sigma0']}",
        flush=True,
    )

    summary_csv = OUT / "cifar100_vgg16_mne_reg_coeff_scan_best_test_summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["method", "regularizer", "reg_coeff", "acc_sigma0", "checkpoint"],
        )
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n[DONE] raw: {raw_csv}", flush=True)
    print(f"[DONE] summary: {summary_csv}", flush=True)
    print(
        f"[DONE] plot: {OUT / 'cifar100_vgg16_mne_reg_coeff_scan_noise_sweep_no_caption.png'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
