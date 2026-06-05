import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 18})


def plot_noise_matrix(csv_path: str) -> None:
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if "L" not in df.columns:
        print(f"Skip (missing L column): {csv_path}")
        return

    sigma_cols = [c for c in df.columns if c != "L"]
    if not sigma_cols:
        print(f"Skip (no sigma columns): {csv_path}")
        return

    sigma_vals = [float(c) for c in sigma_cols]

    plt.figure(figsize=(14, 8))
    for _, row in df.iterrows():
        l_val = int(row["L"])
        acc_vals = [float(row[c]) for c in sigma_cols]
        plt.plot(sigma_vals, acc_vals, marker="o", linewidth=2.6, markersize=5, label=f"L={l_val}")

    plt.title(f"Noise Sweep Accuracy: {os.path.basename(csv_path)}", fontsize=22, fontweight="bold")
    plt.xlabel("Sigma", fontsize=18)
    plt.ylabel("Accuracy (%)", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=14)
    plt.tight_layout()

    out_path = csv_path.replace(".csv", "_lineplot.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    noise_dir = os.path.join(base_dir, "Noise_exp")
    if not os.path.isdir(noise_dir):
        print(f"Directory not found: {noise_dir}")
        return

    files = sorted(
        f for f in os.listdir(noise_dir)
        if f.endswith(".csv") and f.startswith("noise_sweep_matrix_")
    )
    if not files:
        print(f"No matrix CSV found in: {noise_dir}")
        return

    for name in files:
        plot_noise_matrix(os.path.join(noise_dir, name))


if __name__ == "__main__":
    main()
