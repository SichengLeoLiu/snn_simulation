import subprocess
import os
import csv
import sys

def main():
    # 实验配置：增加 L=32
    L_values = [2, 4, 8, 16, 32]
    dataset = "mnist"
    arch = "cnn2"
    # T_val 将在循环中动态设置为等于 L
    num_steps = 100
    device = "mps" # 用户指定的设备 (Mac MPS)
    
    # 获取脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, "mnist-checkpoints-c4_c8")
    
    # 存储所有结果
    all_results = {}
    
    max_t = max(L_values) # 用于确定 CSV 列数

    for L in L_values:
        T_val = L # 核心修改：T 等于对应的 L
        weights = os.path.join(checkpoint_dir, f"cnn2_L[{L}].pth")
        if not os.path.exists(weights):
            print(f"Warning: Weights not found for L={L} at {weights}")
            continue
            
        print(f"\n========================================")
        print(f"Running FI for L={L}, T={T_val}, Weights: {os.path.basename(weights)}")
        print(f"========================================\n")
        
        # 使用 sys.executable 确保使用当前环境的 Python
        cmd = [
            sys.executable, os.path.join(base_dir, "calculate_fisher_info.py"),
            "-w", weights,
            "-data", dataset,
            "-arch", arch,
            "-T", str(T_val),
            "--num_steps", str(num_steps),
            "-dev", device,
            "--spike_schedule", "all"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # 计算完成后，搜寻生成的 .txt 文件
            prefix = f"fi_{dataset}_{arch}_T{T_val}_"
            for filename in os.listdir(base_dir):
                if filename.startswith(prefix) and filename.endswith(".txt"):
                    mode = filename.replace(prefix, "").replace(".txt", "")
                    
                    filepath = os.path.join(base_dir, filename)
                    with open(filepath, "r") as f:
                        data = list(map(float, f.read().strip().split(",")))
                        all_results[(L, mode)] = data
                    
        except subprocess.CalledProcessError as e:
            print(f"Error running command for L={L}: {e}")

    # 写入汇总表格 (CSV)
    output_csv = os.path.join(base_dir, "fi_summary_results_variable_T_with_L32.csv")
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
