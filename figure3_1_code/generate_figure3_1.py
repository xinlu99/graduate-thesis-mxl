# -*- coding: utf-8 -*-
"""独立生成图 3.1：局部自利 vs 全局协同。"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 420,
    "font.size": 13,
    "axes.titlesize": 13.5,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
})
plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def paper_box(ax, xy, w, h, title, lines, fc="#ffffff", ec="#2f3b52", title_fc="#eef3fb", title_fs=11.6, body_fs=10.6, title_weight="bold", body_weight="normal"):
    x, y = xy
    outer = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.3, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(outer)
    title_h = min(0.38, h * 0.34)
    header = Rectangle((x, y + h - title_h), w, title_h, linewidth=0, facecolor=title_fc)
    ax.add_patch(header)
    ax.text(x + 0.18, y + h - title_h / 2, title, ha="left", va="center", fontsize=title_fs, weight=title_weight, color="#1f2d3d")
    body = "\n".join(lines) if isinstance(lines, (list, tuple)) else lines
    ax.text(x + w / 2, y + (h - title_h) / 2 + 0.02, body, ha="center", va="center", fontsize=body_fs, color="#25313f", wrap=True, weight=body_weight)


def node(ax, center, r, label, fc="#f6f8fb", ec="#2f3b52", text_color="#1f2d3d", lw=1.2, fs=10.4, weight="bold"):
    c = Circle(center, r, facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(c)
    ax.text(center[0], center[1], label, ha="center", va="center", fontsize=fs, color=text_color, wrap=True, weight=weight)


def label(ax, xy, text, fs=10.0, color="#405066", ha="center", va="center", weight="semibold"):
    ax.text(xy[0], xy[1], text, fontsize=fs, color=color, ha=ha, va=va, wrap=True, weight=weight)


def connect(ax, p0, p1, color="#4a5568", lw=1.5, style="->"):
    arrow = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=12,
        linewidth=lw, color=color,
        connectionstyle="arc3,rad=0.0",
        shrinkA=6, shrinkB=6,
    )
    ax.add_patch(arrow)


def generate_figure():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.4, 9.6))
    for ax in (ax1, ax2):
        ax.set_xlim(0, 10.8)
        ax.set_ylim(0, 6.8)
        ax.axis("off")

    ax1.set_title("(a) 缺乏协调时的局部自利行为", fontsize=16.2, fontweight="bold", pad=12)
    ax2.set_title("(b) 可信共享与激励下的协同分配", fontsize=16.2, fontweight="bold", pad=12)

    task_pos = [(1.7, 5.72), (4.0, 5.72), (6.8, 5.72), (9.1, 5.72)]

    for idx, p in enumerate(task_pos, start=1):
        paper_box(ax1, (p[0] - 0.80, p[1] - 0.44), 1.60, 0.82, f"任务{idx}", ["高收益"], fc="#fbfcfe", title_fc="#f4e8ea", title_fs=13.0, body_fs=11.8, title_weight="bold", body_weight="semibold")
    node(ax1, (2.0, 1.92), 0.82, "节点A\n空闲", fc="#f7fbff", fs=12.4, weight="bold")
    node(ax1, (5.4, 1.92), 1.02, "节点B\n过载", fc="#fdecea", ec="#b94a48", text_color="#8a2f2c", lw=1.9, fs=12.8, weight="bold")
    node(ax1, (8.8, 1.92), 0.82, "节点C\n空闲", fc="#f7fbff", fs=12.4, weight="bold")
    for p in task_pos:
        connect(ax1, (p[0], p[1] - 0.50), (5.4, 3.00), color="#b94a48", lw=2.0)
    ax1.text(
        5.4, 3.78, "大量任务持续涌向同一热点节点",
        ha="center", va="center", fontsize=12.8, fontweight="bold", color="#7a2e2b",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.95),
        zorder=10,
    )
    label(ax1, (5.4, 0.58), "结果：热点拥塞、边缘闲置、整体吞吐下降", fs=12.8, color="#7a2e2b", weight="bold")

    for idx, p in enumerate(task_pos, start=1):
        paper_box(ax2, (p[0] - 0.80, p[1] - 0.44), 1.60, 0.82, f"任务{idx}", ["业务请求"], fc="#fbfcfe", title_fc="#edf2f7", title_fs=13.0, body_fs=11.8, title_weight="bold", body_weight="semibold")
    paper_box(ax2, (2.55, 3.58), 5.7, 1.18, "联盟链公共信息板", ["全局负载摘要", "信誉评分 / 协作记录"], fc="#fffdf9", title_fc="#eef3fb", title_fs=13.2, body_fs=11.8, title_weight="bold", body_weight="semibold")
    node(ax2, (2.0, 1.62), 0.86, "节点A\n分担", fc="#f7fbff", fs=12.5, weight="bold")
    node(ax2, (5.4, 1.62), 0.86, "节点B\n分担", fc="#f7fbff", fs=12.5, weight="bold")
    node(ax2, (8.8, 1.62), 0.86, "节点C\n分担", fc="#f7fbff", fs=12.5, weight="bold")
    for p in task_pos:
        connect(ax2, (p[0], p[1] - 0.50), (5.4, 4.78), color="#7b8794", lw=1.4)
    connect(ax2, (3.45, 3.56), (2.0, 2.66), color="#3b6ea8", lw=2.2)
    connect(ax2, (5.4, 3.56), (5.4, 2.50), color="#3b6ea8", lw=2.2)
    connect(ax2, (7.35, 3.56), (8.8, 2.66), color="#3b6ea8", lw=2.2)
    ax2.text(
        5.4, 2.96, "节点依据可信摘要进行分流协作",
        ha="center", va="center", fontsize=12.8, fontweight="bold", color="#375a7f",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.95),
        zorder=10,
    )
    label(ax2, (5.4, 0.58), "结果：负载更均衡，任务完成率与公平性同步提升", fs=12.8, color="#2f5d50", weight="bold")

    fig.tight_layout(pad=1.3, h_pad=2.1)
    fig.savefig(OUT / "3-new-1.png", bbox_inches="tight")
    plt.close(fig)
    print("已生成:", OUT / "3-new-1.png")


if __name__ == "__main__":
    generate_figure()
