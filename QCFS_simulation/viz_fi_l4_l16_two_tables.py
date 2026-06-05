import os
import pandas as pd
import matplotlib.pyplot as plt


def _extract_curve(csv_path: str, l_value: int):
    df = pd.read_csv(csv_path)
    row = df[df["L"] == l_value]
    if row.empty:
        return None, None
    t_cols = [c for c in df.columns if c.startswith("T")]
    y = [float(row.iloc[0][c]) for c in t_cols]
    x = list(range(1, len(t_cols) + 1))
    return x, y


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fi_dir = os.path.join(base_dir, "Fisher Information")

    csv_a = os.path.join(fi_dir, "fi_summary_results_variable_T_with_L32_c2_c4 copy.csv")
    csv_b = os.path.join(fi_dir, "fi_summary_results_variable_T_with_L32_c16_c32 copy.csv")

    # 颜色按 L 区分，线型按表（模型组）区分
    color_map = {4: "#1f77b4", 16: "#d62728"}  # blue / red
    style_map = {
        "c2_c4": "-",
        "c16_c32": "--",
    }

    plt.figure(figsize=(14, 8))

    for l_value in [4, 16]:
        x_a, y_a = _extract_curve(csv_a, l_value)
        if x_a is not None:
            plt.plot(
                x_a,
                y_a,
                linestyle=style_map["c2_c4"],
                color=color_map[l_value],
                linewidth=2.4,
                marker="o",
                markersize=4,
                label=f"L={l_value}, c2_c4",
            )

        x_b, y_b = _extract_curve(csv_b, l_value)
        if x_b is not None:
            plt.plot(
                x_b,
                y_b,
                linestyle=style_map["c16_c32"],
                color=color_map[l_value],
                linewidth=2.4,
                marker="o",
                markersize=4,
                label=f"L={l_value}, c16_c32",
            )

    plt.title("FI Curves for L=4 and L=16 (Two Tables)", fontsize=22, fontweight="bold")
    plt.xlabel("Timestep (T)", fontsize=18)
    plt.ylabel("FI Trace (log scale)", fontsize=18)
    plt.yscale("log")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(fi_dir, "fi_l4_l16_c2c4_vs_c16c32_lineplot.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
