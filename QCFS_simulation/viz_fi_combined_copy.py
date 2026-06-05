import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 18})


def plot_single_csv(csv_path: str, title_tag: str) -> None:
    if not os.path.exists(csv_path):
        print(f"Skip missing file: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    t_cols = [c for c in df.columns if c.startswith("T")]
    if not t_cols:
        print(f"Skip invalid file (no T columns): {csv_path}")
        return

    x = list(range(1, len(t_cols) + 1))
    fig, ax = plt.subplots(figsize=(14, 8))

    for _, row in df.iterrows():
        l_val = int(row["L"])
        y = [float(row[c]) for c in t_cols]
        ax.plot(x, y, marker="o", linewidth=2.3, markersize=4, label=f"L={l_val}")

    ax.set_title(f"FI Curves ({title_tag})", fontsize=22, fontweight="bold")
    ax.set_xlabel("Timestep (T)", fontsize=18)
    ax.set_ylabel("FI Trace (log scale)", fontsize=18)
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=14)
    plt.tight_layout()

    out_path = csv_path.replace(".csv", "_lineplot.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fi_dir = os.path.join(base_dir, "Fisher Information")

    files = [
        ("c2_c4", "fi_summary_results_variable_T_with_L32_c2_c4 copy.csv"),
        ("c4_c8", "fi_summary_results_variable_T_with_L32_c4_c8 copy.csv"),
        ("c16_c32", "fi_summary_results_variable_T_with_L32_c16_c32 copy.csv"),
    ]

    for tag, name in files:
        plot_single_csv(os.path.join(fi_dir, name), tag)


if __name__ == "__main__":
    main()
