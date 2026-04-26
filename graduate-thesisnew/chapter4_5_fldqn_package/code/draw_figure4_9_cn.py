from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

out_dir = Path("figures")
out_dir.mkdir(exist_ok=True)

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

candidate_fonts = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
    "Noto Serif CJK SC", "SimHei", "Microsoft YaHei", "Arial Unicode MS"
]
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
chosen_font = next((f for f in candidate_fonts if f in available_fonts), "DejaVu Sans")
mpl.rcParams["font.sans-serif"] = [chosen_font]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 22
mpl.rcParams["axes.titlesize"] = 24
mpl.rcParams["axes.labelsize"] = 24
mpl.rcParams["xtick.labelsize"] = 22
mpl.rcParams["ytick.labelsize"] = 22
mpl.rcParams["legend.fontsize"] = 20

colors = {
    "FL-DQN": "#2F6DB5",
    "-FL预测": "#E69F00",
    "-并发惩罚": "#2A9D55",
    "-经验回放": "#C44536",
    "-奖励塑形": "#7E57C2",
}
markers = {
    "FL-DQN": "o",
    "-FL预测": "s",
    "-并发惩罚": "D",
    "-经验回放": "^",
    "-奖励塑形": "v",
}


def save_both(path: Path):
    if path.suffix.lower() == ".pdf":
        plt.savefig(path, bbox_inches="tight")
        plt.savefig(path.with_suffix(".png"), dpi=350, bbox_inches="tight")
    else:
        plt.savefig(path, dpi=350, bbox_inches="tight")
        plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def plot_line(df, y_label, filename, ylim, final_fmt):
    plt.figure(figsize=(11.2, 7.8))
    xvals = df.iloc[:, 0].to_numpy()
    for col in df.columns[1:]:
        y = df[col].to_numpy()
        c = colors[col]
        plt.plot(xvals, y, marker=markers[col], linewidth=3.6, markersize=11.0, label=col, color=c)
        band = np.array([1.4, 1.7, 2.1, 2.5]) if "时延" in y_label else np.array([0.28, 0.36, 0.46, 0.58])
        plt.fill_between(xvals, y - band, y + band, color=c, alpha=0.12)
        plt.text(xvals[-1] + 0.24, y[-1], final_fmt.format(y[-1]), color=c, va="center", fontsize=18)

    plt.xlabel("平均并发任务数")
    plt.ylabel(y_label)
    plt.xticks(xvals)
    plt.xlim(xvals[0] - 0.5, xvals[-1] + 2.4)
    plt.ylim(*ylim)
    plt.grid(True, linestyle="--", linewidth=1.2, alpha=0.35)
    plt.legend(frameon=True, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    save_both(out_dir / filename)
    plt.close()


def plot_combined_bars(overhead, variance, filename):
    plt.figure(figsize=(11.2, 7.8))
    labels = overhead["方法变体"].tolist()
    x = np.arange(len(labels))
    width = 0.36

    ax1 = plt.gca()
    vals1 = overhead["在线决策开销(ms)"].to_numpy()
    vals2 = variance["跨轮次标准差"].to_numpy()

    bars1 = ax1.bar(x - width / 2, vals1, width=width,
                    color="#4C78A8", edgecolor="black", label="在线决策开销 (ms)")
    ax1.set_ylabel("在线决策开销 (ms)")
    ax1.set_ylim(0, 5.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=12)
    ax1.grid(True, axis="y", linestyle="--", linewidth=1.2, alpha=0.35)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, vals2, width=width,
                    color="#E76F51", edgecolor="black", label="跨轮次标准差")
    ax2.set_ylabel("跨轮次标准差")
    ax2.set_ylim(0, 13.8)

    for b, v in zip(bars1, vals1):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.10, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=18, color="#1F4E79")
    for b, v in zip(bars2, vals2):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.20, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=18, color="#9C2F2F")

    handles = [bars1, bars2]
    labels_leg = ["在线决策开销 (ms)", "跨轮次标准差"]
    ax1.legend(handles, labels_leg, frameon=True, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    save_both(out_dir / filename)
    plt.close()


plot_line(delay_df, "平均时延 (ms)", "fig4_9_a_avg_delay_cn.pdf", (135, 300), "{:.0f}")
plot_line(sla_df, "完成率 (%)", "fig4_9_b_sla_cn.pdf", (78, 98), "{:.1f}")
plot_combined_bars(overhead_df, variance_df, "fig4_9_c_overhead_variance_cn.pdf")
print("Figure 4-9 subfigures saved to ./figures")
