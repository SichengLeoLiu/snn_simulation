import subprocess
import os
import csv
import sys

def main():
    # 实验配置：增加 L=32
    L_values = [2, 4, 8, 16, 32]
    dataset = "mnist"
    arch = "cnn2"
    # 统一固定 SNN 时间步为 32，使每个 L 都输出 T1~T32
    fixed_T = 32
    num_steps = 100
    device = "mps" # 用户指定的设备 (Mac MPS)
    modes_to_run = [
        "normal",
        "uniform",
        "weight_sign_pos_front",
        "weight_sign_neg_front",
    ]
    
    # 获取脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, "mnist-checkpoints")
    results_dir = os.path.join(base_dir, "Fisher Information")
    os.makedirs(results_dir, exist_ok=True)
    
    # 存储所有结果
    all_results = {}
    
    max_t = fixed_T # 用于确定 CSV 列数

    for L in L_values:
        T_val = fixed_T
        weights = os.path.join(checkpoint_dir, f"cnn2_L[{L}].pth")
        if not os.path.exists(weights):
            print(f"Warning: Weights not found for L={L} at {weights}")
            continue
            
        print(f"\n========================================")
        print(f"Running FI for L={L}, T={T_val}, Weights: {os.path.basename(weights)}")
        print(f"========================================\n")
        
        try:
            for mode in modes_to_run:
                # 使用 sys.executable 确保使用当前环境的 Python
                cmd = [
                    sys.executable, os.path.join(base_dir, "calculate_fisher_info.py"),
                    "-w", weights,
                    "-data", dataset,
                    "-arch", arch,
                    "-T", str(T_val),
                    "--num_steps", str(num_steps),
                    "-dev", device,
                    "--spike_schedule", mode
                ]
                subprocess.run(cmd, check=True)

                # 读取当前模式输出
                output_file = os.path.join(
                    base_dir, f"fi_{dataset}_{arch}_T{T_val}_{mode}.txt"
                )
                if not os.path.exists(output_file):
                    print(f"Warning: Result file not found for mode={mode}: {output_file}")
                    continue
                with open(output_file, "r") as f:
                    data = list(map(float, f.read().strip().split(",")))
                    all_results[(L, mode)] = data
                    
        except subprocess.CalledProcessError as e:
            print(f"Error running command for L={L}: {e}")

    # 写入汇总表格 (CSV)
    output_csv = os.path.join(results_dir, "fi_summary_results_variable_T_with_L32.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        
        # 表头: L, Mode, T1, T2, ..., T32
        header = ["L", "Mode"] + [f"T{t+1}" for t in range(max_t)]
        writer.writerow(header)
        
        # 排序并写入数据
        sorted_keys = sorted(all_results.keys())
        for L, mode in sorted_keys:
            if (L, mode) in all_results:
                fi_data = all_results[(L, mode)]
                # 填充空位以匹配表头长度
                row = [L, mode] + fi_data + [""] * (max_t - len(fi_data))
                writer.writerow(row)

    print(f"\nSuccessfully finished all tests including L=32!")
    print(f"Summary results saved to: {output_csv}")

if __name__ == "__main__":
    main()
