import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def run_interactive_plot():
    # --- 初始参数 ---
    init_base_scale = 0.1
    init_k_gain = 25.0
    stop_threshold = 0.015  # 15mm
    
    # 生成距离数据 (0 到 50cm)
    dist = np.linspace(0, 0.5, 1000)

    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.25) # 为滑块留出底部空间

    # 定义计算函数
    def get_scale(d, k, base):
        s = base * (1.0 - np.exp(-k * d))
        s[d < stop_threshold] = 0
        return s

    # 绘制初始曲线
    line, = ax.plot(dist * 100, get_scale(dist, init_k_gain, init_base_scale), 
                    lw=2, color='#1f77b4', label=f'k_gain={init_k_gain}')
    
    # 辅助线和样式
    ax.axvline(x=stop_threshold*100, color='r', linestyle='--', alpha=0.4)
    ax.set_xlim(0, 50)
    ax.set_ylim(-0.01, init_base_scale + 0.05)
    ax.set_xlabel('Distance to Target (cm)')
    ax.set_ylabel('Action Scale (η)')
    ax.set_title('Interactive Exponential Scale Tuning')
    ax.grid(True, alpha=0.3)

    # --- 添加滑块 ---
    ax_k = plt.axes([0.2, 0.1, 0.6, 0.03]) # [left, bottom, width, height]
    slider_k = Slider(ax_k, 'k_gain ', 1.0, 100.0, valinit=init_k_gain, valstep=1.0)

    ax_base = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider_base = Slider(ax_base, 'base_scale ', 0.01, 0.5, valinit=init_base_scale)

    # 更新函数
    def update(val):
        k = slider_k.val
        base = slider_base.val
        line.set_ydata(get_scale(dist, k, base))
        line.set_label(f'k_gain={k:.1f}')
        ax.set_ylim(-0.01, base + 0.05)
        ax.legend()
        fig.canvas.draw_idle()

    slider_k.on_changed(update)
    slider_base.on_changed(update)

    plt.show()

if __name__ == "__main__":
    run_interactive_plot()