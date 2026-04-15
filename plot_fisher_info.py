import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Plot Fisher Information against Timestep")
    parser.add_argument("-f", "--files", nargs="+", required=True, help="Result files to plot")
    parser.add_argument("-o", "--output", default="fisher_info_plot.png", type=str, help="Output image path")
    parser.add_argument("-l", "--labels", nargs="+", help="Labels for each file")
    args = parser.parse_args()

    plt.figure(figsize=(8, 6))

    for i, file_path in enumerate(args.files):
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        with open(file_path, "r") as f:
            data = list(map(float, f.read().strip().split(",")))
        
        timesteps = list(range(1, len(data) + 1))
        label = args.labels[i] if args.labels and i < len(args.labels) else os.path.basename(file_path)
        
        plt.plot(timesteps, data, marker='o', label=label)

    plt.xlabel("Timestep")
    plt.ylabel("Fisher Information Trace")
    plt.title("Fisher Information Dynamics in SNN")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
