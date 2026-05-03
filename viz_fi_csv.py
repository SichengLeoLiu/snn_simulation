import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# 全局设置字体
plt.rcParams.update({'font.size': 18})

def plot_fi_csv(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # 读取 CSV
    df = pd.read_csv(file_path)
    
    # 获取所有的 L 值并排序 (确保 2, 4, 8, 16, 32 顺序正确)
    l_values = sorted(df['L'].unique())
    num_l = len(l_values)
    
    if num_l == 0:
        print(f"No data found in {file_path}")
        return

    # 计算需要的行数，固定每行 3 个子图
    cols = 3
    rows = math.ceil(num_l / cols)
    
    # 创建画布
    fig, axes = plt.subplots(rows, cols, figsize=(24, 8 * rows))
    axes_flat = axes.flatten() if rows > 1 or cols > 1 else [axes]
    
    fig.suptitle(f"Fisher Information Dynamics: {os.path.basename(file_path)}", fontsize=34, y=0.99, fontweight='bold')

    for i, L in enumerate(l_values):
        ax = axes_flat[i]
        subset = df[df['L'] == L]
        
        # 遍历该 L 下的所有模式
        for _, row in subset.iterrows():
            mode = row['Mode']
            # 提取 T1, T2... 的数值
            fi_values = row.iloc[2:].dropna().values
            fi_values = [float(x) for x in fi_values if str(x).strip() != '']
            
            if len(fi_values) == 0:
                continue
                
            timesteps = range(1, len(fi_values) + 1)
            ax.plot(timesteps, fi_values, marker='o', label=mode, alpha=0.8, linewidth=2.8, markersize=8)
        
        ax.set_title(f"Quantization Level L = {L}", fontsize=24, fontweight='bold', pad=12)
        ax.set_xlabel("Timestep (T)", fontsize=20)
        ax.set_ylabel("FI Trace (Log Scale)", fontsize=20)
        ax.set_yscale('log')
        
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(True, which="both", ls="-", alpha=0.5)
        
        # 图例放在子图内部
        ax.legend(fontsize=14, loc='best', framealpha=0.6)

    # 隐藏多余的空白子图
    for j in range(num_l, len(axes_flat)):
        axes_flat[j].axis('off')
    
    # 调整整体布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    output_png = file_path.replace('.csv', '_dynamic_grid.png')
    plt.savefig(output_png, bbox_inches='tight', dpi=150)
    print(f"Plot saved with {num_l} L levels to: {output_png}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fi_dir = os.path.join(base_dir, "Fisher Information")
    files = [
        "fi_summary_results_variable_T_with_L32_c2_c4.csv",
        "fi_summary_results_variable_T_with_L32_c4_c8.csv",
        "fi_summary_results_variable_T_with_L32_c16_c32.csv"
    ]
    
    for f in files:
        file_path = os.path.join(fi_dir, f)
        plot_fi_csv(file_path)
