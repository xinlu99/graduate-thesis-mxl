from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

# 设置工作目录和输出目录
script_dir = Path(__file__).parent
base_dir = script_dir.parent
out_dir = base_dir / "figures"
out_dir.mkdir(exist_ok=True)

# 数据定义
delay_df = pd.DataFrame({
    "平均并发任务数": [4, 8, 12, 16],
    "FL-DQN": [145, 160, 182, 209],
    "-FL预测": [173, 201, 236, 284],
    "-并发惩罚": [158, 180, 206, 243],
    "-经验回放": [164, 188, 219, 256],
    "-奖励塑形": [149, 171, 197, 238],
})
sla_df = pd.DataFrame({
    "平均并发任务数": [4, 8, 12, 16],
    "FL-DQN": [96.8, 95.4, 93.7, 91.2],
    "-FL预测": [93.0, 89.8, 85.6, 80.1],
    "-并发惩罚": [95.1, 92.4, 89.0, 84.7],
    "-经验回放": [95.8, 93.3, 90.4, 86.4],
    "-奖励塑形": [95.9, 94.0, 91.8, 87.9],
})
overhead_df = pd.DataFrame({
    "方法变体": ["FL-DQN", "-FL预测", "-并发惩罚", "-经验回放", "-奖励塑形"],
    "在线决策开销(ms)": [3.8, 3.1, 3.6, 2.4, 3.4],
})
variance_df = pd.DataFrame({
    "方法变体": ["FL-DQN", "-FL预测", "-并发惩罚", "-经验回放", "-奖励塑形"],
    "跨轮次标准差": [5.1, 9.6, 8.1, 11.2, 7.3],
})

# 字体配置
candidate_fonts = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
    "Noto Serif CJK SC", "SimHei", "Microsoft YaHei", "Arial Unicode MS"
]
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
chosen_font = next((f for f in candidate_fonts if f in available_fonts), "DejaVu Sans")
mpl.rcParams["font.sans-serif"] = [chosen_font]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 20
mpl.rcParams["axes.titlesize"] = 20
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["xtick.labelsize"] = 20
mpl.rcParams["ytick.labelsize"] = 20
mpl.rcParams["legend.fontsize"] = 20

# 统一配色（按用户指定顺序）
palette = ["#DB0B49", "#DCAD01", "#BD70DB", "#BCDB70", "#70CFDB", "black"]

colors = {
    "FL-DQN": palette[0],
    "-FL预测": palette[1],
    "-并发惩罚": palette[2],
    "-经验回放": palette[3],
    "-奖励塑形": palette[4],
}

bars_colors = {
    "overhead": "#DB0B49",
    "variance": "#DCAD01",
}

markers = {
    "FL-DQN": "o",
    "-FL预测": "s",
    "-并发惩罚": "^",
    "-经验回放": "D",
    "-奖励塑形": "v",
}

# 新增线型定义，增强区分度
line_styles = {
    "FL-DQN": "-",            # 实线
    "-FL预测": "--",           # 虚线
    "-并发惩罚": "-.",         # 点划线
    "-经验回放": ":",           # 点线
    "-奖励塑形": (0, (3, 1, 1, 1)), # 细点划线
}

def save_both(path: Path):
    # 导出多种格式：PDF (矢量), PNG (位图), EPS (矢量)
    pdf_path = path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    
    png_path = path.with_suffix(".png")
    plt.savefig(png_path, dpi=350, bbox_inches="tight", pad_inches=0.02)
    
    eps_path = path.with_suffix(".eps")
    # EPS 不支持透明度，导出时会自动处理
    plt.savefig(eps_path, format='eps', bbox_inches="tight", pad_inches=0.02)

def plot_line(df, y_label, filename, ylim, final_fmt):
    # 调小画布尺寸但保持高分辨率，使元素密度更高，减少留白
    plt.figure(figsize=(9, 8)) 
    ax = plt.gca()
    xvals = df.iloc[:, 0].to_numpy()
    
    for col in df.columns[1:]:
        y = df[col].to_numpy()
        c = colors[col]
        ls = line_styles[col]
        mk = markers[col]
        # 线条进一步加粗
        lw = 4.5 if col == "FL-DQN" else 2.5
        ms = 12 if col == "FL-DQN" else 10
        
        plt.plot(xvals, y, marker=mk, linestyle=ls, linewidth=lw, 
                 markersize=ms, label=col, color=c, markeredgecolor='white', markeredgewidth=1.5)
        
        band = np.array([1.4, 1.7, 2.1, 2.5]) if "时延" in y_label else np.array([0.28, 0.36, 0.46, 0.58])
        plt.fill_between(xvals, y - band, y + band, color=c, alpha=0.08, edgecolor=None)
        
        # 标注字体也同步放大
        plt.text(xvals[-1] + 0.25, y[-1], final_fmt.format(y[-1]), 
                 color=c, va="center", fontsize=20, fontweight='bold' if col == "FL-DQN" else 'normal')

    plt.xlabel("平均并发任务数", fontweight='bold')
    plt.ylabel(y_label, fontweight='bold')
    plt.xticks(xvals)
    plt.xlim(xvals[0] - 0.5, xvals[-1] + 4.0) # 适配大字号标注
    plt.ylim(ylim[0], ylim[1]) 
    
    plt.grid(True, linestyle=":", linewidth=1.0, color='gray', alpha=0.3)
    
    # 更加紧凑的图例布局：移到图外上方
    plt.legend(frameon=True, loc="lower center", bbox_to_anchor=(0.5, 1.02), 
               ncol=2, fontsize=20, edgecolor='gray', framealpha=0.8, 
               columnspacing=0.7, handletextpad=0.35, borderaxespad=0.05)
    
    # 极致压缩留白
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.92])
    save_both(out_dir / filename)
    plt.close()

def plot_combined_bars(overhead, variance, filename):
    plt.figure(figsize=(11, 8))
    labels = overhead["方法变体"].tolist()
    x = np.arange(len(labels))
    width = 0.4

    ax1 = plt.gca()
    # 柱状图采用统一调色板
    bars1 = ax1.bar(
        x - width / 2,
        overhead["在线决策开销(ms)"].to_numpy(),
        width=width,
        facecolor="none",
        edgecolor=bars_colors["overhead"],
        linewidth=1.8,
        hatch="///",
        label="开销 (ms)",
    )
    ax1.set_ylabel("在线决策开销 (ms)", fontweight='bold')
    ax1.set_ylim(0, 6.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + width / 2,
        variance["跨轮次标准差"].to_numpy(),
        width=width,
        facecolor="none",
        edgecolor=bars_colors["variance"],
        linewidth=1.8,
        hatch="\\\\",
        label="标准差",
    )
    ax2.set_ylabel("跨轮次标准差", fontweight='bold')
    ax2.set_ylim(0, 15)

    # 标注数值
    for b, v in zip(bars1, overhead["在线决策开销(ms)"].to_numpy()):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=20, fontweight='bold')
    for b, v in zip(bars2, variance["跨轮次标准差"].to_numpy()):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=20, fontweight='bold')

    # 合并图例：统一风格
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower center", 
               bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True, 
               edgecolor='gray', fontsize=20)

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.92])
    save_both(out_dir / filename)
    plt.close()

def plot_single_bar(df, y_col, y_label, filename, color, hatch_style):
    plt.figure(figsize=(8.6, 7.2))
    labels = df["方法变体"].tolist()
    x = np.arange(len(labels))

    ax = plt.gca()
    bars = ax.bar(
        x,
        df[y_col].to_numpy(),
        width=0.58,
        facecolor="none",
        edgecolor=color,
        linewidth=1.8,
        hatch=hatch_style,
    )
    ax.set_ylabel(y_label, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(True, axis='y', linestyle=":", linewidth=1.0, color='gray', alpha=0.3)

    for b, v in zip(bars, df[y_col].to_numpy()):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + (0.12 if y_col == "在线决策开销(ms)" else 0.25),
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=20,
            fontweight='bold',
            color=color,
        )

    if y_col == "在线决策开销(ms)":
        ax.set_ylim(0, 6.0)
    else:
        ax.set_ylim(0, 15)

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
    save_both(out_dir / filename)
    plt.close()

# 生成图片
plot_line(delay_df, "平均时延 (ms)", "fig4_9_a_avg_delay_cn.pdf", (135, 305), "{:.0f}")
plot_line(sla_df, "完成率 (%)", "fig4_9_b_sla_cn.pdf", (75, 100), "{:.1f}")
plot_combined_bars(overhead_df, variance_df, "fig4_9_c_overhead_variance_cn.pdf")
plot_single_bar(overhead_df, "在线决策开销(ms)", "在线决策开销 (ms)", "fig4_9_c_overhead_cn.pdf", bars_colors["overhead"], "///")
plot_single_bar(variance_df, "跨轮次标准差", "跨轮次标准差", "fig4_9_d_variance_cn.pdf", bars_colors["variance"], "///")

print(f"所有子图已生成至: {out_dir}")

