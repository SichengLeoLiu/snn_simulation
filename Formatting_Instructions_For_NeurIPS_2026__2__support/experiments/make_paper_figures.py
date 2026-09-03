#!/usr/bin/env python3
"""Generate publication figures from the repository's experiment summaries."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUPPORT_DIR = Path(__file__).resolve().parents[1]
PAPER_DIR = SUPPORT_DIR.with_name(SUPPORT_DIR.name.removesuffix("_support"))
REPO_DIR = PAPER_DIR.parent
FIGURE_DIR = PAPER_DIR / "images"
FIGURE_DIR.mkdir(exist_ok=True)

MECHANISM_DIR = REPO_DIR / "important_results" / "recent_cifar10_mechanism_summary"
BN_SCOPE_CSV = (
    REPO_DIR
    / "important_results"
    / "cifar10_bn_gamma_beta_l2_scope_sigma0_5_seed42"
    / "mne_stability_ablation_mean_std.csv"
)
CIFAR_SWEEP_CSV = {
    "cifar10": (
        REPO_DIR
        / "important_results"
        / "cifar10_vgg16_five_regs_sigma0_5_5seed"
        / "mne_stability_ablation_mean_std.csv"
    ),
    "cifar100": (
        REPO_DIR
        / "important_results"
        / "cifar100_vgg16_five_regs_sigma0_5_5seed"
        / "mne_stability_ablation_mean_std.csv"
    ),
}
DEPTH_SWEEP_CSV = {
    ("cifar10", 13): (
        REPO_DIR
        / "important_results"
        / "cifar10_vgg13_five_regs_sigma0_5_5seed"
        / "mne_stability_ablation_mean_std.csv"
    ),
    ("cifar10", 19): (
        REPO_DIR
        / "important_results"
        / "cifar10_vgg19_five_regs_sigma0_5_5seed"
        / "mne_stability_ablation_mean_std.csv"
    ),
    ("cifar100", 13): (
        REPO_DIR
        / "important_results"
        / "cifar100_vgg13_five_regs_sigma0_5_5seed"
        / "mne_stability_ablation_mean_std.csv"
    ),
    ("cifar100", 19): (
        REPO_DIR
        / "important_results"
        / "cifar100_vgg19_five_regs_sigma0_5_5seed"
        / "mne_stability_ablation_mean_std.csv"
    ),
}
IMAGENET_SWEEP_CSV = (
    REPO_DIR
    / "important_results"
    / "imagenet_resnet18_five_regs_sigma0_5_seed42"
    / "mne_stability_ablation_mean_std.csv"
)
LOW_LATENCY_CSV = (
    REPO_DIR
    / "important_results"
    / "mne_adv_5seed"
    / "c2c4_fashion_mnist_mnist_T4_T8_T16_summary.csv"
)
EFFICIENCY_CSV = SUPPORT_DIR / "experiments" / "results" / "event_efficiency_summary.csv"

COLORS = {
    "MNE-standard": "#D62728",
    "Weights-only": "#009E73",
    "L1-all": "#E69F00",
    "MNE-all": "#CC79A7",
    "L2-all": "#6F3FA0",
    "Weights+BN-gamma": "#D55E00",
    "MNE-L2": "#D62728",
    "Orthogonal": "#009E73",
    "No Reg": "#E69F00",
    "L2": "#0072B2",
}

MARKERS = ["o"] * 6
DISPLAY_NAMES = {
    "MNE-standard": "MNE-L2 (ours)",
    "MNE-L2": "MNE-L2 (ours)",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.5,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURE_DIR / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def plot_bn_scope() -> None:
    df = pd.read_csv(BN_SCOPE_CSV)
    methods = [
        ("weight_decay_weights_only", "Weights\nonly", "#009E73"),
        ("manual_l2_w_bn_beta", "Weights +\nBN $\\beta$", "#0072B2"),
        ("manual_l2_w_bn_gamma", "Weights +\nBN $\\gamma$", "#D55E00"),
        ("manual_l2_w_bn", "Weights +\nBN $\\gamma+\\beta$", "#6F3FA0"),
    ]
    sigma_one = df[np.isclose(df["sigma"], 1.0)]
    values = []
    for variant, _, _ in methods:
        row = sigma_one[sigma_one["variant"] == variant]
        if len(row) != 1:
            raise ValueError(f"Expected one sigma=1 row for {variant}, found {len(row)}")
        values.append(float(row.iloc[0]["acc_mean"]))

    fig, ax = plt.subplots(figsize=(5.1, 2.15))
    x = np.arange(len(methods))
    bars = ax.bar(
        x,
        values,
        width=0.68,
        color=[color for _, _, color in methods],
        edgecolor="white",
        linewidth=0.7,
    )
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9.2)
    ax.set_ylabel("Top-1 accuracy (%)", fontsize=10)
    ax.set_xticks(x, [label for _, label, _ in methods])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=9.2, length=0)
    ax.tick_params(axis="y", labelsize=9.0)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.995, top=0.93, bottom=0.27)
    save_figure(fig, "bn_affine_scope")


def plot_bn_scope_sweep() -> None:
    df = pd.read_csv(BN_SCOPE_CSV)
    methods = [
        ("weight_decay_weights_only", "Weights only", "#009E73", "-"),
        ("manual_l2_w_bn_beta", r"+ BN $\beta$", "#0072B2", "-"),
        ("manual_l2_w_bn_gamma", r"+ BN $\gamma$", "#D55E00", "-"),
        ("manual_l2_w_bn", r"+ BN $\gamma+\beta$", "#6F3FA0", "--"),
    ]

    fig, ax = plt.subplots(figsize=(5.4, 1.7))
    for variant, label, color, linestyle in methods:
        group = df[df["variant"] == variant].sort_values("sigma")
        ax.plot(
            group["sigma"],
            group["acc_mean"],
            label=label,
            color=color,
            linestyle=linestyle,
            marker="o",
            linewidth=1.7,
            markersize=3.2,
        )

    ax.set_xlim(-0.05, 5.05)
    ax.set_ylim(5, 95)
    ax.set_xlabel(r"Per-timestep Gaussian noise $\sigma$", fontsize=8.5)
    ax.set_ylabel("Top-1 accuracy (%)", fontsize=8.5)
    ax.tick_params(labelsize=7.3)
    ax.grid(True)
    ax.legend(
        loc="upper right",
        ncol=2,
        fontsize=6.6,
        frameon=True,
        borderpad=0.35,
        handlelength=2.2,
        columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.105, right=0.995, top=0.97, bottom=0.26)
    save_figure(fig, "bn_affine_scope_sweep")


def plot_cifar_multiseed() -> None:
    methods = [
        ("weight_decay", "L2-all", "#0072B2", "--"),
        ("weight_decay_weights_only", "L2-wo", "#009E73", "-"),
        ("l1", "L1-wo", "#E69F00", "-"),
        ("old_detach", "MNE-L2 (ours)", "#D62728", "-"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    dataset_titles = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}
    for panel, (ax, dataset) in enumerate(zip(axes[:2], ("cifar10", "cifar100"))):
        df = pd.read_csv(CIFAR_SWEEP_CSV[dataset])
        for idx, (variant, label, color, linestyle) in enumerate(methods):
            group = df[df["variant"] == variant].sort_values("sigma")
            sigma = group["sigma"].to_numpy()
            mean = group["acc_mean"].to_numpy()
            std = group["acc_std"].to_numpy()
            ax.plot(
                sigma,
                mean,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=MARKERS[idx],
                linewidth=1.8,
                markersize=3.8,
                zorder=4 if variant == "old_detach" else 2,
            )
            ax.fill_between(sigma, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
        ax.set_title(f"({chr(97 + panel)}) {dataset_titles[dataset]} VGG16", fontsize=8.3)
        ax.set_xlim(-0.05, 5.05)
        ax.set_ylim((5, 94) if dataset == "cifar10" else (0, 68))
        ax.tick_params(labelsize=7.2)
        ax.grid(True)

    imagenet = pd.read_csv(IMAGENET_SWEEP_CSV)
    imagenet = imagenet[imagenet["sigma"] <= 2.0]
    for idx, (variant, label, color, linestyle) in enumerate(methods):
        group = imagenet[imagenet["variant"] == variant].sort_values("sigma")
        axes[2].plot(
            group["sigma"],
            group["acc_mean"],
            label=label,
            color=color,
            linestyle=linestyle,
            marker="o",
            linewidth=1.8,
            markersize=3.8,
            zorder=4 if variant == "old_detach" else 2,
        )
    axes[2].set_title("(c) ImageNet ResNet-18", fontsize=8.3)
    axes[2].set_xlim(-0.03, 2.03)
    axes[2].set_ylim(0, 66)
    axes[2].set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    axes[2].tick_params(labelsize=7.2)
    axes[2].grid(True)

    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.text(0.5, 0.14, r"Per-timestep Gaussian noise $\sigma$", ha="center", fontsize=8.3)
    fig.text(0.008, 0.57, "Top-1 accuracy (%)", va="center", rotation="vertical", fontsize=8.3)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=6.7,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.91, bottom=0.29, wspace=0.27)
    save_figure(fig, "cifar_imagenet_absolute_noise")


def plot_vgg_depth_transfer() -> None:
    methods = [
        ("weight_decay", "L2-all", "#0072B2", "--"),
        ("weight_decay_weights_only", "L2-wo", "#009E73", "-"),
        ("l1", r"L1-wo ($10^{-5}$)", "#E69F00", "-"),
        ("old_detach", "MNE-L2 (ours)", "#D62728", "-"),
        ("calibrated_mne_a0p1", r"Calibrated MNE ($\alpha=0.1$)", "#6F3FA0", "-"),
    ]
    panels = [
        ("cifar10", 13, "CIFAR-10 VGG13"),
        ("cifar10", 19, "CIFAR-10 VGG19"),
        ("cifar100", 13, "CIFAR-100 VGG13"),
        ("cifar100", 19, "CIFAR-100 VGG19"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.25), sharex=True)
    for panel, (ax, (dataset, depth, title)) in enumerate(zip(axes.flat, panels)):
        df = pd.read_csv(DEPTH_SWEEP_CSV[(dataset, depth)])
        for variant, label, color, linestyle in methods:
            group = df[df["variant"] == variant].sort_values("sigma")
            sigma = group["sigma"].to_numpy()
            mean = group["acc_mean"].to_numpy()
            std = group["acc_std"].to_numpy()
            ax.plot(
                sigma,
                mean,
                label=label,
                color=color,
                linestyle=linestyle,
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                zorder=5 if variant == "old_detach" else 2,
            )
            ax.fill_between(
                sigma,
                mean - std,
                mean + std,
                color=color,
                alpha=0.13,
                linewidth=0,
            )
        ax.set_title(f"({chr(97 + panel)}) {title}", fontsize=8.4)
        ax.set_xlim(-0.05, 5.05)
        ax.set_ylim((5, 94) if dataset == "cifar10" else (0, 68))
        ax.tick_params(labelsize=7.2)
        ax.grid(True)

    fig.text(0.5, 0.12, r"Per-timestep Gaussian noise $\sigma$", ha="center", fontsize=8.3)
    fig.text(0.012, 0.55, "Top-1 accuracy (%)", va="center", rotation="vertical", fontsize=8.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=6.5,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.94, bottom=0.22, hspace=0.25, wspace=0.18)
    save_figure(fig, "vgg_depth_transfer")


def plot_latency_reliability() -> None:
    robustness = pd.read_csv(LOW_LATENCY_CSV)
    efficiency = pd.read_csv(EFFICIENCY_CSV)
    methods = ["MNE-L2", "Orthogonal", "No Reg", "L2"]
    dataset_specs = [
        ("fashion_mnist", "Fashion-MNIST"),
        ("mnist", "MNIST"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.35), sharex="col")
    for col, (dataset, title) in enumerate(dataset_specs):
        robust_data = robustness[robustness["dataset"] == dataset]
        eff_data = efficiency[efficiency["dataset"] == dataset]
        for idx, method in enumerate(methods):
            robust_group = robust_data[robust_data["method_label"] == method].sort_values("T")
            efficiency_label = "No regularization" if method == "No Reg" else method
            eff_group = eff_data[eff_data["method_label"] == efficiency_label].sort_values("T")
            axes[0, col].errorbar(
                robust_group["T"],
                robust_group["sigma_0p5_acc"],
                yerr=robust_group["sigma_0p5_std"],
                color=COLORS[method],
                marker=MARKERS[idx],
                capsize=2,
                label=DISPLAY_NAMES.get(method, method),
            )
            axes[1, col].errorbar(
                eff_group["T"],
                eff_group["event_synops_per_ann_mac_mean"],
                yerr=eff_group["event_synops_per_ann_mac_std"],
                color=COLORS[method],
                marker=MARKERS[idx],
                capsize=2,
                label=DISPLAY_NAMES.get(method, method),
            )

        axes[0, col].set_title(title)
        axes[0, col].grid(True)
        axes[1, col].grid(True)
        axes[1, col].axhline(1.0, color="#555555", linestyle="--", linewidth=0.9)
        axes[1, col].set_xticks([4, 8, 16])
        axes[1, col].set_xlabel("SNN timesteps $T$")

    axes[0, 0].set_ylabel(r"Accuracy at $\sigma=0.5$ (%)")
    axes[1, 0].set_ylabel("Event SynOps / dense ANN MACs")
    axes[0, 0].set_ylim(20, 85)
    axes[0, 1].set_ylim(20, 85)
    axes[1, 0].set_ylim(0, 3.8)
    axes[1, 1].set_ylim(0, 3.8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.15, hspace=0.22, wspace=0.22)
    save_figure(fig, "latency_reliability_efficiency")


def plot_layerwise_diagnostics() -> None:
    df = pd.read_csv(MECHANISM_DIR / "layerwise_metrics.csv")
    methods = ["MNE-standard", "Weights-only", "L2-all", "Weights+BN-gamma"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))

    for idx, method in enumerate(methods):
        group = df[df["method"] == method].sort_values("layer_index")
        downstream = group[group["layer_index"] > 0]
        axes[0].plot(
            downstream["layer_index"],
            downstream["rho"],
            color=COLORS[method],
            marker=MARKERS[idx],
            label=DISPLAY_NAMES.get(method, method),
        )
        axes[1].plot(
            group["layer_index"],
            group["p_cross"],
            color=COLORS[method],
            marker=MARKERS[idx],
            label=DISPLAY_NAMES.get(method, method),
        )

    axes[0].set_yscale("log")
    axes[0].set_title(r"(a) Margin-to-noise ratio $\rho_l$")
    axes[0].set_xlabel("IF layer index")
    axes[0].set_ylabel(r"$d_{50,l}/\sigma_{\mathrm{eff},l}$")
    axes[0].grid(True, which="both")
    axes[1].set_title(r"(b) Quantization-bin crossing $P_{\mathrm{cross},l}$")
    axes[1].set_xlabel("IF layer index")
    axes[1].set_ylabel("Crossing probability")
    axes[1].set_ylim(-0.03, 0.82)
    axes[1].grid(True)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.09))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.87, bottom=0.29, wspace=0.28)
    save_figure(fig, "layerwise_diagnostics")


def main() -> None:
    set_style()
    plot_bn_scope()
    plot_bn_scope_sweep()
    plot_cifar_multiseed()
    plot_vgg_depth_transfer()
    plot_latency_reliability()
    plot_layerwise_diagnostics()
    print(f"Wrote paper figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
