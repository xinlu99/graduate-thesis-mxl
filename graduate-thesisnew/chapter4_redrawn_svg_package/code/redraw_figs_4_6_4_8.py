"""
Redraw Figures 4-6, 4-7, 4-8 from CSV data.
Outputs: images/4-new-6.pdf, images/4-new-7.pdf, images/4-new-8.pdf
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager
from matplotlib.lines import Line2D
try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── font ──────────────────────────────────────────────────────────────────────
candidate_fonts = [
    "Noto Sans CJK SC", "Noto Serif CJK SC", "SimHei",
    "Microsoft YaHei", "Arial Unicode MS", "WenQuanYi Micro Hei",
]
available = {f.name for f in font_manager.fontManager.ttflist}
chosen = next((f for f in candidate_fonts if f in available), "DejaVu Sans")
plt.rcParams.update({
    "font.sans-serif": [chosen],
    "axes.unicode_minus": False,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "lines.linewidth": 1.5,
    "lines.markersize": 5.0,
    "mathtext.fontset": "stix",
})

DATA = Path(__file__).parent.parent / "data"
OUT  = Path(__file__).parent.parent.parent / "images"
OUT.mkdir(exist_ok=True)

PAPER_PALETTE = ["#DB0B49", "#DCAD01", "#BD70DB", "#BCDB70", "#70CFDB", "black", "#8c564b", "#7f7f7f"]
PAPER_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
PAPER_LINESTYLES = ["-", "-", "-", "-", "-", "-", "-", "-"]
PAPER_LINEWIDTH = 1.5
PAPER_MARKERSIZE = 5.0
PAPER_BAR_ALPHA = 0.86

# ── colour / marker palettes ───────────────────────────────────────────────────
FL_COLORS = {
    "Clustered FL":  PAPER_PALETTE[0],
    "FedAvg":        PAPER_PALETTE[1],
    "Centralized":   PAPER_PALETTE[2],
    "Local-only":    PAPER_PALETTE[3],
}
FL_MARKERS = {"Clustered FL": PAPER_MARKERS[0], "FedAvg": PAPER_MARKERS[1], "Centralized": PAPER_MARKERS[2], "Local-only": PAPER_MARKERS[3]}

E2E_COLORS = {
    "FL-DQN":           PAPER_PALETTE[0],
    "Centralized-DQN":  PAPER_PALETTE[1],
    "DQN-only":         PAPER_PALETTE[2],
    "FL-only":          PAPER_PALETTE[3],
    "Heuristic":        PAPER_PALETTE[4],
}
E2E_MARKERS = {
    "FL-DQN": PAPER_MARKERS[0], "Centralized-DQN": PAPER_MARKERS[1], "DQN-only": PAPER_MARKERS[2],
    "FL-only": PAPER_MARKERS[3], "Heuristic": PAPER_MARKERS[4],
}

ROB_COLORS = {
    "FL-DQN": PAPER_PALETTE[0], "DQN-only": PAPER_PALETTE[2],
    "FL-only": PAPER_PALETTE[3], "Heuristic": PAPER_PALETTE[4],
}
ROB_MARKERS = {"FL-DQN": PAPER_MARKERS[0], "DQN-only": PAPER_MARKERS[2], "FL-only": PAPER_MARKERS[3], "Heuristic": PAPER_MARKERS[4]}


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="out", length=3, width=0.7)
    ax.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.35)


def _legend(ax, **kw):
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="#aaaaaa",
                    borderpad=0.4, handletextpad=0.4, labelspacing=0.3,
                    framealpha=0.92, **kw)
    leg.get_frame().set_linewidth(0.6)
    return leg


def _subcap(ax, text):
    ax.text(0.5, -0.22, text, transform=ax.transAxes,
            ha="center", va="top", fontsize=10.5)


def _save_both(base_path_no_ext: Path, fig):
    fig.savefig(base_path_no_ext.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base_path_no_ext.with_suffix(".png"), bbox_inches="tight", facecolor="white")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4-6  Prediction module performance (2×2)
# ══════════════════════════════════════════════════════════════════════════════
def draw_fig46():
    conv_df = pd.read_csv(DATA / "fig4_6_convergence.csv", index_col=0)
    dens_df = pd.read_csv(DATA / "fig4_6_pred_true_density.csv", index_col=0)
    calib_df = pd.read_csv(DATA / "fig4_6_calibration.csv", index_col=0)
    het_df  = pd.read_csv(DATA / "fig4_6_heterogeneity.csv", index_col=0)

    fig = plt.figure(figsize=(10, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.38)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    # (a) convergence
    ax = axes[0]
    x  = conv_df["通信轮次"].values
    for col in ["Clustered FL", "FedAvg", "Centralized", "Local-only"]:
        ax.plot(x, conv_df[col].values,
                marker=FL_MARKERS[col], color=FL_COLORS[col], label=col,
                markerfacecolor=FL_COLORS[col], markeredgecolor="black",
                markeredgewidth=0.3, markevery=1)
    ax.set_xlabel("通信轮次")
    ax.set_ylabel("验证集 MAE (ms)")
    ax.set_xlim(-2, 55)
    ax.set_ylim(1, 16)
    _style(ax)
    _legend(ax, loc="upper right")
    _subcap(ax, "(a) 联邦训练收敛过程")

    # (b) pred-true scatter density
    ax = axes[1]
    y_true = dens_df["真实并发开销(ms)"].values
    y_pred = dens_df["预测并发开销(ms)"].values
    # clip outlier zeros
    mask = (y_true > 0) & (y_pred > 0)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if HAS_SCIPY:
        xy   = np.vstack([y_true, y_pred])
        kde  = gaussian_kde(xy)
        z    = kde(xy)
        idx  = z.argsort()
        sc = ax.scatter(y_true[idx], y_pred[idx], c=z[idx], s=12,
                        cmap="plasma", alpha=0.75, linewidths=0)
        plt.colorbar(sc, ax=ax, shrink=0.82, label="密度")
    else:
        ax.scatter(y_true, y_pred, s=10, color="#e74c3c", alpha=0.45, linewidths=0)
    lo, hi = 0, max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="理想对角线")
    ax.set_xlabel("真实并发开销 (ms)")
    ax.set_ylabel("预测并发开销 (ms)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    _style(ax)
    _legend(ax, loc="upper left")
    _subcap(ax, "(b) 预测—真实散点密度")

    # (c) calibration
    ax = axes[2]
    unc = calib_df["预测不确定性均值"].values
    err = calib_df["经验误差"].values
    ax.plot(unc, err, "o-", color="#e74c3c", markerfacecolor="#e74c3c",
            markeredgecolor="black", markeredgewidth=0.3)
    ax.fill_between(unc, err * 0.82, err * 1.18, color="#e74c3c", alpha=0.12)
    ax.set_xlabel("预测不确定性均值 (ms)")
    ax.set_ylabel("经验绝对误差 (ms)")
    _style(ax)
    _subcap(ax, "(c) 置信度校准结果")

    # (d) heterogeneity
    ax = axes[3]
    labels = het_df["异构强度"].tolist()
    x_pos  = np.arange(len(labels))
    w = 0.18
    offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]
    for i, col in enumerate(["Clustered FL", "FedAvg", "Centralized", "Local-only"]):
        bars = ax.bar(x_pos + offsets[i], het_df[col].values, width=w,
                      color=FL_COLORS[col], label=col,
                      edgecolor="black", linewidth=0.4, alpha=0.9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("测试集 MAE (ms)")
    ax.set_ylim(0, 6.5)
    _style(ax)
    ax.grid(axis="x", visible=False)
    _legend(ax, loc="upper left", ncol=2)
    _subcap(ax, "(d) 不同异构强度下的 MAE")

    _save_both(OUT / "4-new-6", fig)
    plt.close(fig)
    print("Saved 4-new-6.pdf and 4-new-6.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4-7  End-to-end performance (2×2)
# ══════════════════════════════════════════════════════════════════════════════
def draw_fig47():
    delay_df  = pd.read_csv(DATA / "fig4_7_delay.csv",      index_col=0)
    tp_df     = pd.read_csv(DATA / "fig4_7_throughput.csv", index_col=0)
    sla_df    = pd.read_csv(DATA / "fig4_7_sla.csv",        index_col=0)
    p99_df    = pd.read_csv(DATA / "fig4_7_p99.csv",        index_col=0)

    algs = ["FL-DQN", "Centralized-DQN", "DQN-only", "FL-only", "Heuristic"]

    fig = plt.figure(figsize=(10, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.42)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    x_col = "平均并发任务数"

    specs = [
        (axes[0], delay_df,  "平均时延 (ms)",       "(a) 平均任务时延",      None),
        (axes[1], tp_df,     "吞吐量 (tasks/s)",    "(b) 系统吞吐量",        None),
        (axes[2], sla_df,    "SLA 完成率",           "(c) SLA 完成率",        None),
        (axes[3], p99_df,    "P99 时延 (ms)",        "(d) P99 尾时延",        None),
    ]
    for idx, (ax, df, ylabel, cap, _) in enumerate(specs):
        x = df[x_col].values
        for alg in algs:
            if alg not in df.columns:
                continue
            y = df[alg].values
            ax.plot(x, y, marker=E2E_MARKERS[alg], color=E2E_COLORS[alg],
                    label=alg, markerfacecolor=E2E_COLORS[alg],
                    markeredgecolor="black", markeredgewidth=0.3)
        ax.set_xlabel("平均并发任务数")
        ax.set_ylabel(ylabel)
        _style(ax)
        if idx == 1:
            _legend(
                ax,
                loc="lower left",
                fontsize=8.8,
                markerscale=0.78,
                handlelength=1.2,
            )
        elif idx == 2:
            _legend(ax, loc="lower left")
        else:
            _legend(ax, loc="upper left")
        _subcap(ax, cap)

    _save_both(OUT / "4-new-7", fig)
    plt.close(fig)
    print("Saved 4-new-7.pdf and 4-new-7.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4-8  Robustness / action preference / trade-off
#   Layout: row0 = (a) staleness  |  (b) heatmap
#            row1 =     (c) bubble  (centred)
# ══════════════════════════════════════════════════════════════════════════════
def draw_fig48():
    stale_df  = pd.read_csv(DATA / "fig4_8_staleness.csv", index_col=0)
    heat_df   = pd.read_csv(DATA / "fig4_8_heatmap.csv",   index_col=0)
    trade_df  = pd.read_csv(DATA / "fig4_8_tradeoff.csv",  index_col=0)

    fig = plt.figure(figsize=(11, 9.2))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.62, wspace=0.52, height_ratios=[1.0, 1.28])
    ax1 = fig.add_subplot(gs[0, 0:2])   # (a) top-left, half width
    ax2 = fig.add_subplot(gs[0, 2:4])   # (b) top-right, half width
    ax3 = fig.add_subplot(gs[1, 1:3])   # (c) bottom centre, half width

    # (a) staleness
    x_st = stale_df["链上摘要延时(ms)"].values
    for alg in ["FL-DQN", "DQN-only", "FL-only", "Heuristic"]:
        ax1.plot(x_st, stale_df[alg].values,
                 marker=ROB_MARKERS[alg], color=ROB_COLORS[alg], label=alg,
                 markerfacecolor=ROB_COLORS[alg], markeredgecolor="black",
                 markeredgewidth=0.3)
    ax1.set_xlabel("链上摘要延时 (ms)")
    ax1.set_ylabel("平均任务时延 (ms)")
    _style(ax1)
    _legend(ax1, loc="upper left")
    _subcap(ax1, "(a) 链上陈旧性敏感性")

    # (b) action heatmap
    risk_levels = heat_df.index.tolist()
    actions     = heat_df.columns.tolist()
    data_mat    = heat_df.values.astype(float)
    im = ax2.imshow(data_mat, aspect="auto", cmap="YlOrRd",
                    vmin=0, vmax=data_mat.max())
    ax2.set_xticks(range(len(actions)))
    ax2.set_xticklabels(actions, rotation=25, ha="right", fontsize=8.5)
    ax2.set_yticks(range(len(risk_levels)))
    ax2.set_yticklabels(risk_levels, fontsize=8.5)
    ax2.set_xlabel("卸载动作")
    ax2.set_ylabel("候选节点负载风险区间")
    plt.colorbar(im, ax=ax2, shrink=0.88, label="选择概率")
    for i in range(len(risk_levels)):
        for j in range(len(actions)):
            ax2.text(j, i, f"{data_mat[i, j]:.2f}",
                     ha="center", va="center", fontsize=7.5,
                     color="white" if data_mat[i, j] > 0.35 else "black")
    _subcap(ax2, "(b) 动作选择热力图")

    # (c) trade-off bubble
    edp   = trade_df["EDP"].values
    jain  = trade_df["Jain指数"].values
    comp  = trade_df["完成率(%)"].values
    names = trade_df["方法"].values
    sizes = (comp - comp.min() + 5) * 18

    for name in names:
        mask = names == name
        ax3.scatter(edp[mask], jain[mask], s=sizes[mask],
                    c=[E2E_COLORS.get(name, "#888888")], alpha=0.85,
                    edgecolors="black", linewidths=0.6, label="_")

    custom_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="w",
               markerfacecolor=E2E_COLORS.get(name, "#888888"),
               markeredgecolor="black", markeredgewidth=0.6,
               markersize=6.5, label=name)
        for name in names
    ]
    ax3.legend(
        handles=custom_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="#aaaaaa",
        framealpha=0.92,
        borderpad=0.35,
        handletextpad=0.4,
        labelspacing=0.32,
        fontsize=9.5,
    )
    ax3.set_xlabel("能时积 EDP")
    ax3.set_ylabel("Jain 公平性指数")
    ax3.set_ylim(top=0.950)
    _style(ax3)
    _subcap(ax3, "(c) 效率—公平性折中")

    _save_both(OUT / "4-new-8", fig)
    plt.close(fig)
    print("Saved 4-new-8.pdf and 4-new-8.png")


if __name__ == "__main__":
    draw_fig46()
    draw_fig47()
    draw_fig48()
    print("All done.")
