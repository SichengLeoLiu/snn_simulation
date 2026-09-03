from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT.parent / "important_results"
MODELS = [
    ("cnn2_c2_c4", "c2-c4", "c2c4"),
    ("cnn2_c4_c8", "c4-c8", "c4c8"),
    ("cnn2_c8_c16", "c8-c16", "c8c16"),
    ("cnn2_c16_c32", "c16-c32", "c16c32"),
]


def _checkpoint_stem(model: str, method: str) -> str:
    return f"{model}_L[16]_fashion_calibrated_mne_{model}_{method}_seed42_ep30_L16"


def _ann_accuracy(model: str, method: str) -> float:
    log_path = ROOT / "fashion_mnist-checkpoints" / f"{_checkpoint_stem(model, method)}.log"
    matches = re.findall(r"Best Test acc=([0-9.]+)", log_path.read_text())
    if not matches:
        raise ValueError(f"No best test accuracy in {log_path}")
    return float(matches[-1])


def _curve(model: str, short: str, method: str) -> dict[float, float]:
    path = (
        RESULT_ROOT
        / f"fashion_calibrated_mne_{short}_seed42_ep30"
        / method
        / "full_sweep"
        / (
            f"noise_sweep_matrix_fashion_mnist_{model}_T16_"
            "mode_rate_uniform_schedule_normal_seed_42.csv"
        )
    )
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    return {
        float(key): float(value)
        for key, value in row.items()
        if key not in {"L", "T"} and value not in {None, ""}
    }


def main() -> None:
    out_dir = RESULT_ROOT / "fashion_calibrated_mne_width_summary_seed42_ep30"
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoint_rows = []
    curve_rows = []
    curves = {}
    for model, label, short in MODELS:
        l2 = _curve(model, short, "l2_wo")
        mne = _curve(model, short, "cmne_a0p1")
        curves[label] = (l2, mne)
        sigmas = sorted(set(l2) & set(mne))
        ann_l2 = _ann_accuracy(model, "l2_wo")
        ann_mne = _ann_accuracy(model, "cmne_a0p1")
        endpoint_rows.append(
            {
                "model": label,
                "ann_l2": ann_l2,
                "ann_mne": ann_mne,
                "ann_delta": ann_mne - ann_l2,
                "snn_clean_l2": l2[0.0],
                "snn_clean_mne": mne[0.0],
                "snn_clean_delta": mne[0.0] - l2[0.0],
                "snn_sigma1_l2": l2[1.0],
                "snn_sigma1_mne": mne[1.0],
                "snn_sigma1_delta": mne[1.0] - l2[1.0],
                "drop_l2": l2[0.0] - l2[1.0],
                "drop_mne": mne[0.0] - mne[1.0],
                "mean_curve_delta": sum(mne[s] - l2[s] for s in sigmas)
                / len(sigmas),
            }
        )
        for sigma in sigmas:
            curve_rows.append(
                {
                    "model": label,
                    "sigma": sigma,
                    "l2_weights_only": l2[sigma],
                    "calibrated_mne_alpha_0p1": mne[sigma],
                    "mne_minus_l2": mne[sigma] - l2[sigma],
                }
            )

    endpoint_path = out_dir / "width_endpoint_summary.csv"
    with endpoint_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(endpoint_rows[0]))
        writer.writeheader()
        writer.writerows(endpoint_rows)

    curve_path = out_dir / "width_full_sweep.csv"
    with curve_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=False)
    for ax, (_, label, _), row in zip(axes.flat, MODELS, endpoint_rows):
        l2, mne = curves[label]
        sigmas = sorted(set(l2) & set(mne))
        ax.plot(sigmas, [l2[s] for s in sigmas], marker="o", label="L2 weights-only")
        ax.plot(
            sigmas,
            [mne[s] for s in sigmas],
            marker="o",
            label="Calibrated MNE (alpha=0.1)",
        )
        ax.set_title(f"{label}: delta@1={row['snn_sigma1_delta']:+.2f} pp")
        ax.set_xlabel("Gaussian noise sigma")
        ax.set_ylabel("SNN accuracy (%)")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Fashion-MNIST CNN width sweep, seed 42, L=T=16")
    fig.tight_layout()
    curve_figure = out_dir / "full_noise_curves_by_width.png"
    fig.savefig(curve_figure, dpi=200, bbox_inches="tight")
    plt.close(fig)

    labels = [row["model"] for row in endpoint_rows]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.plot(x, [row["ann_delta"] for row in endpoint_rows], marker="o", label="ANN clean")
    ax.plot(
        x,
        [row["snn_clean_delta"] for row in endpoint_rows],
        marker="o",
        label="SNN clean",
    )
    ax.plot(
        x,
        [row["snn_sigma1_delta"] for row in endpoint_rows],
        marker="o",
        label="SNN sigma=1",
    )
    ax.set_xticks(x, labels)
    ax.set_xlabel("CNN width")
    ax.set_ylabel("Calibrated MNE - L2 (percentage points)")
    ax.set_title("Effect of calibrated MNE changes with width")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    delta_figure = out_dir / "accuracy_delta_vs_width.png"
    fig.savefig(delta_figure, dpi=200, bbox_inches="tight")
    plt.close(fig)

    c16_controls = []
    c16_curves = {}
    for method, label, alpha in [
        ("l2_wo", "L2 weights-only", 0.0),
        ("cmne_a0p05", "Calibrated MNE alpha=0.05", 0.05),
        ("cmne_a0p1", "Calibrated MNE alpha=0.1", 0.1),
    ]:
        curve = _curve("cnn2_c16_c32", "c16c32", method)
        c16_curves[label] = curve
        c16_controls.append(
            {
                "method": label,
                "alpha": alpha,
                "ann_acc": _ann_accuracy("cnn2_c16_c32", method),
                "snn_clean": curve[0.0],
                "snn_sigma1": curve[1.0],
                "drop": curve[0.0] - curve[1.0],
            }
        )
    c16_control_path = out_dir / "c16c32_alpha_control.csv"
    with c16_control_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(c16_controls[0]))
        writer.writeheader()
        writer.writerows(c16_controls)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for label, curve in c16_curves.items():
        sigmas = sorted(curve)
        ax.plot(sigmas, [curve[s] for s in sigmas], marker="o", label=label)
    ax.set_xlabel("Gaussian noise sigma")
    ax.set_ylabel("SNN accuracy (%)")
    ax.set_title("Fashion-MNIST c16-c32 alpha control")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    c16_control_figure = out_dir / "c16c32_alpha_control.png"
    fig.savefig(c16_control_figure, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] {endpoint_path}")
    print(f"[DONE] {curve_path}")
    print(f"[DONE] {curve_figure}")
    print(f"[DONE] {delta_figure}")
    print(f"[DONE] {c16_control_path}")
    print(f"[DONE] {c16_control_figure}")


if __name__ == "__main__":
    main()
