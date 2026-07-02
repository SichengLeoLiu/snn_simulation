"""噪声 sweep 实验统一默认：sigma 从 0 到 1。"""

NOISE_SIGMA_START = 0.0
NOISE_SIGMA_END = 1.0
NOISE_SIGMA_STEP = 0.05


def noise_sigma_step_argv() -> list[str]:
    return [
        "--noise_sigma_start",
        str(NOISE_SIGMA_START),
        "--noise_sigma_end",
        str(NOISE_SIGMA_END),
        "--noise_sigma_step",
        str(NOISE_SIGMA_STEP),
    ]


def sigma_plot_xticks(major_step: float = 0.1) -> list[float]:
    """折线图 x 轴主刻度（默认仍每 0.1 显示，数据点步长为 NOISE_SIGMA_STEP）。"""
    n = int(round(NOISE_SIGMA_END / major_step)) + 1
    return [round(i * major_step, 2) for i in range(n)]
