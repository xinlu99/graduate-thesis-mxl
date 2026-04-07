# -*- coding: utf-8 -*-
"""
按第三章正文语义生成示意图与实验风格曲线（matplotlib）。
运行：python figures/generate_chapter3_figures.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "images"
IMG.mkdir(parents=True, exist_ok=True)

SEED = 7
rng = np.random.default_rng(SEED)

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 420,
    "font.size": 13,
    "axes.titlesize": 13.5,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.22,
})
# 中文标签（Windows 常见字体；缺失时 matplotlib 会回退）
plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _box(ax, xy, w, h, text, fc, ec="#333333", fs=10):
    x, y = xy
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, edgecolor=ec, facecolor=fc, mutation_aspect=0.6,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def _arrow(ax, p0, p1, color="#444"):
    ax.annotate(
        "", xy=p1, xytext=p0,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.4, shrinkA=4, shrinkB=4),
    )


def _paper_box(ax, xy, w, h, title, lines, fc="#ffffff", ec="#2f3b52", title_fc="#eef3fb", title_fs=11.6, body_fs=10.6, title_weight="bold", body_weight="normal"):
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


def _node(ax, center, r, label, fc="#f6f8fb", ec="#2f3b52", text_color="#1f2d3d", lw=1.2, fs=10.4, weight="bold"):
    c = Circle(center, r, facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(c)
    ax.text(center[0], center[1], label, ha="center", va="center", fontsize=fs, color=text_color, wrap=True, weight=weight)


def _label(ax, xy, text, fs=10.0, color="#405066", ha="center", va="center", weight="semibold"):
    ax.text(xy[0], xy[1], text, fontsize=fs, color=color, ha=ha, va=va, wrap=True, weight=weight)


def _connect(ax, p0, p1, text=None, color="#4a5568", rad=0.0, lw=1.5, style="->", text_offset=(0, 0)):
    arrow = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=12,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=6, shrinkB=6,
    )
    ax.add_patch(arrow)
    if text:
        mx = (p0[0] + p1[0]) / 2 + text_offset[0]
        my = (p0[1] + p1[1]) / 2 + text_offset[1]
        _label(ax, (mx, my), text, color=color)


def _poly_arrow(ax, pts, color="#4a5568", lw=1.5):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw)
    ax.annotate(
        "", xy=pts[-1], xytext=pts[-2],
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=0, shrinkB=0),
    )


def fig_selfish_vs_coop():
    """图3-1：局部自利（热点） vs 全局协同（均衡）"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.4, 9.6))
    for ax in (ax1, ax2):
        ax.set_xlim(0, 10.8)
        ax.set_ylim(0, 6.8)
        ax.axis("off")

    ax1.set_title("(a) 缺乏协调时的局部自利行为", fontsize=16.2, fontweight="bold", pad=12)
    ax2.set_title("(b) 可信共享与激励下的协同分配", fontsize=16.2, fontweight="bold", pad=12)

    task_pos = [(1.7, 5.72), (4.0, 5.72), (6.8, 5.72), (9.1, 5.72)]

    # 上图：热点汇聚，尽量减少文字密度，突出“拥塞 vs 闲置”
    for idx, p in enumerate(task_pos, start=1):
        _paper_box(ax1, (p[0] - 0.80, p[1] - 0.44), 1.60, 0.82, f"任务{idx}", ["高收益"], fc="#fbfcfe", title_fc="#f4e8ea", title_fs=13.0, body_fs=11.8, title_weight="bold", body_weight="semibold")
    _node(ax1, (2.0, 1.92), 0.82, "节点A\n空闲", fc="#f7fbff", fs=12.4, weight="bold")
    _node(ax1, (5.4, 1.92), 1.02, "节点B\n过载", fc="#fdecea", ec="#b94a48", text_color="#8a2f2c", lw=1.9, fs=12.8, weight="bold")
    _node(ax1, (8.8, 1.92), 0.82, "节点C\n空闲", fc="#f7fbff", fs=12.4, weight="bold")
    for p in task_pos:
        _connect(ax1, (p[0], p[1] - 0.50), (5.4, 3.00), color="#b94a48", lw=2.0)
    ax1.text(
        5.4, 3.78, "大量任务持续涌向同一热点节点",
        ha="center", va="center", fontsize=12.8, fontweight="bold", color="#7a2e2b",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.95),
        zorder=10,
    )
    _label(ax1, (5.4, 0.58), "结果：热点拥塞、边缘闲置、整体吞吐下降", fs=12.8, color="#7a2e2b", weight="bold")

    # 下图：可信共享后的均衡分配
    for idx, p in enumerate(task_pos, start=1):
        _paper_box(ax2, (p[0] - 0.80, p[1] - 0.44), 1.60, 0.82, f"任务{idx}", ["业务请求"], fc="#fbfcfe", title_fc="#edf2f7", title_fs=13.0, body_fs=11.8, title_weight="bold", body_weight="semibold")
    _paper_box(ax2, (2.55, 3.58), 5.7, 1.18, "联盟链公共信息板", ["全局负载摘要", "信誉评分 / 协作记录"], fc="#fffdf9", title_fc="#eef3fb", title_fs=13.2, body_fs=11.8, title_weight="bold", body_weight="semibold")
    _node(ax2, (2.0, 1.62), 0.86, "节点A\n分担", fc="#f7fbff", fs=12.5, weight="bold")
    _node(ax2, (5.4, 1.62), 0.86, "节点B\n分担", fc="#f7fbff", fs=12.5, weight="bold")
    _node(ax2, (8.8, 1.62), 0.86, "节点C\n分担", fc="#f7fbff", fs=12.5, weight="bold")
    for p in task_pos:
        _connect(ax2, (p[0], p[1] - 0.50), (5.4, 4.78), color="#7b8794", lw=1.4)
    _connect(ax2, (3.45, 3.56), (2.0, 2.66), color="#3b6ea8", lw=2.2)
    _connect(ax2, (5.4, 3.56), (5.4, 2.50), color="#3b6ea8", lw=2.2)
    _connect(ax2, (7.35, 3.56), (8.8, 2.66), color="#3b6ea8", lw=2.2)
    ax2.text(
        5.4, 2.96, "节点依据可信摘要进行分流协作",
        ha="center", va="center", fontsize=12.8, fontweight="bold", color="#375a7f",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.95),
        zorder=10,
    )
    _label(ax2, (5.4, 0.58), "结果：负载更均衡，任务完成率与公平性同步提升", fs=12.8, color="#2f5d50", weight="bold")

    fig.tight_layout(pad=1.3, h_pad=2.1)
    fig.savefig(IMG / "3-new-1.png", bbox_inches="tight")
    plt.close(fig)


def fig_three_layer_framework():
    """图3-2：环境层—联盟链—智能体 层次与信息流"""
    fig, ax = plt.subplots(figsize=(11.4, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    _paper_box(ax, (1.0, 6.2), 10.0, 1.2, "任务与资源环境层", ["任务到达、异构算力、队列变化、星间链路状态"], fc="#fbfdff")
    _paper_box(ax, (1.0, 4.2), 10.0, 1.15, "联盟链公共信息板", ["全局负载摘要 g_t   |   信誉向量   |   审计记录   |   链上时延估计"], fc="#fffdf8", title_fc="#f3efe2")
    _paper_box(ax, (1.0, 1.1), 4.1, 1.6, "局部 Actor（节点 i）", ["输入：本地状态 l_t^i + 链上摘要 g_t", "输出：卸载目标 / 资源比例 / 接收意愿"], fc="#fbfcff")
    _paper_box(ax, (7.0, 1.1), 4.0, 1.6, "集中式 Critic（训练阶段）", [r"输入：联合状态 $s_t^{\mathrm{ctr}}$", r"输出：价值估计 $V_\phi(s_t^{\mathrm{ctr}})$"], fc="#fbfcff")

    _connect(ax, (6.0, 6.2), (6.0, 5.35), text="状态摘要上链", color="#3d5873", lw=1.5, text_offset=(0.9, 0.1))
    _connect(ax, (3.3, 4.2), (3.05, 2.7), text="广播 g_t", color="#856404", lw=1.5, text_offset=(-0.5, 0.1))
    _connect(ax, (8.9, 2.7), (8.9, 4.2), text="联合轨迹/奖励", color="#3d5873", lw=1.5, text_offset=(0.8, 0.12))
    _connect(ax, (5.1, 1.9), (7.0, 1.9), text="策略更新信息", color="#5b6b7f", lw=1.3, text_offset=(0, 0.25))

    _label(ax, (6.0, 0.35), "训练阶段利用联合信息稳定估值；执行阶段仅依赖本地观测与可信摘要", fs=9.4, color="#425466")

    fig.tight_layout()
    fig.savefig(IMG / "3-new-4.png", bbox_inches="tight")
    plt.close(fig)


def fig_task_chain_loop():
    """图3-2-flow：五步闭环"""
    fig, ax = plt.subplots(figsize=(11.6, 4.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    step_x = [0.8, 3.2, 5.6, 8.0, 10.4]
    labels = [
        ("步骤1", ["任务到达", "本地状态感知"]),
        ("步骤2", ["读取链上摘要", "获取全局上下文"]),
        ("步骤3", ["局部协同决策", "生成混合动作"]),
        ("步骤4", ["执行反馈", "更新奖励与信誉中间量"]),
        ("步骤5", ["链上记账", "形成下一轮公共状态"]),
    ]
    for x, (title, lines) in zip(step_x, labels):
        _paper_box(ax, (x, 1.5), 1.9, 1.15, title, lines, fc="#fbfcff")
    for i in range(len(step_x) - 1):
        _connect(ax, (step_x[i] + 1.9, 2.075), (step_x[i + 1], 2.075), color="#4b5d73", lw=1.4)
    _poly_arrow(ax, [(11.35, 1.5), (11.35, 0.65), (1.75, 0.65), (1.75, 1.5)], color="#7a8594", lw=1.25)
    _label(ax, (6.6, 0.32), "链上摘要与治理结果在下一决策周期继续生效，形成任务闭环—信息闭环—治理闭环", fs=9.2)

    fig.tight_layout()
    fig.savefig(IMG / "3-new-5.png", bbox_inches="tight")
    plt.close(fig)


def fig_modeling_path():
    """图3-3：集中式—MDP—PO-MAMG 推导路径"""
    fig, ax = plt.subplots(figsize=(12.4, 4.4))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 4)
    ax.axis("off")

    stages = [
        (0.4, "集中式全局优化", ["假设全局状态可实时获取", "目标：统一最优调度"]),
        (3.2, "中心依赖受限", ["去中心化环境下", "全局视图难实时同步"]),
        (6.0, "单智能体 MDP", ["仅把其他节点视作环境", "难表达策略耦合"]),
        (8.8, "问题抽象升级", ["需要显式刻画多主体", "状态—动作—收益联动"]),
        (11.6, "部分可观测 MAMG", ["本地观测 + 链上摘要", "适配 CTDE 多智能体求解"]),
    ]
    for x, title, lines in stages:
        _paper_box(ax, (x, 1.15), 2.3, 1.45, title, lines, fc="#fbfcff")
    for i in range(len(stages) - 1):
        x0 = stages[i][0] + 2.3
        x1 = stages[i + 1][0]
        _connect(ax, (x0, 1.88), (x1, 1.88), color="#4b5d73", lw=1.45)
    _label(ax, (7.25, 3.25), "从系统优化问题到多智能体马尔可夫博弈的建模推导路径", fs=11.5, color="#243447")

    fig.tight_layout()
    fig.savefig(IMG / "3-new-3.png", bbox_inches="tight")
    plt.close(fig)


def fig_mappo_flow():
    """图3-6：BC-CTDE-MAPPO 训练与执行"""
    fig, ax = plt.subplots(figsize=(11.8, 6.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    _paper_box(ax, (0.8, 6.3), 4.2, 1.35, "训练阶段：集中训练", ["联合状态 s_t^{ctr}", "共享 Critic / GAE / PPO 更新"], fc="#fbfcff")
    _paper_box(ax, (7.0, 6.3), 4.0, 1.35, "执行阶段：分散执行", ["各节点仅用 o_t^i=(l_t^i,g_t)", "本地 Actor 独立采样动作"], fc="#fbfcff")
    _paper_box(ax, (2.2, 3.95), 7.6, 1.2, "联盟链公共信息板", ["写入资源摘要、任务摘要、审计记录；广播新的 g_t / g_{t+1}"], fc="#fffdf8", title_fc="#f3efe2")
    _paper_box(ax, (0.8, 1.35), 4.2, 1.35, "环境交互", ["执行联合动作 a_t", "更新队列 / 链路 / 能耗 / 信誉中间量"], fc="#fbfcff")
    _paper_box(ax, (7.0, 1.35), 4.0, 1.35, "Rollout 与参数优化", ["存储轨迹", "计算优势并更新 θ_i, φ"], fc="#fbfcff")

    _connect(ax, (2.9, 6.3), (5.0, 5.15), text="训练使用联合信息", color="#3f5d7d", lw=1.45, text_offset=(0.0, 0.15))
    _connect(ax, (9.0, 5.15), (9.0, 6.3), text="执行仅依赖局部观测", color="#3f5d7d", lw=1.45, text_offset=(1.1, 0.05))
    _connect(ax, (4.9, 3.95), (2.9, 2.7), text="g_t 进入观测", color="#7a6419", lw=1.4, text_offset=(-0.6, 0.05))
    _connect(ax, (7.1, 2.7), (7.1, 3.95), text="轨迹与奖励回传", color="#7a6419", lw=1.4, text_offset=(1.0, 0.05))
    _connect(ax, (5.0, 2.0), (7.0, 2.0), text="轨迹缓冲 / 优势估计", color="#546274", lw=1.3, text_offset=(0, 0.28))

    _label(ax, (6.0, 0.4), "联盟链负责可信共享与审计回写；CTDE 负责稳定训练与去中心化执行", fs=9.4, color="#425466")

    fig.tight_layout()
    fig.savefig(IMG / "3-new-6.png", bbox_inches="tight")
    plt.close(fig)


def fig_blockchain_layers():
    """图3-4（原与3-new-4重复）：分层联盟链四模块职责 — 独立构图"""
    fig, ax = plt.subplots(figsize=(11.4, 6.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    _paper_box(ax, (0.9, 5.9), 4.1, 1.15, "主链 / 资源审计链", ["身份管理", "信誉锚定", "治理规则与跨链监管"], fc="#fbfcff")
    _paper_box(ax, (7.0, 5.9), 4.1, 1.15, "子链 / 任务信令链", ["任务开始/结束事件", "局部资源摘要", "局部审计记录"], fc="#fbfcff")
    _paper_box(ax, (2.2, 3.75), 7.6, 1.1, "公共信息板聚合层", ["生成全局平均负载、负载方差、任务分布、信誉向量 g_t"], fc="#fffdf8", title_fc="#f3efe2")
    _paper_box(ax, (2.2, 1.55), 7.6, 1.1, "智能合约执行层", ["异常识别、信誉奖惩、策略审计、权限更新"], fc="#fbfcff")

    _connect(ax, (3.0, 5.9), (5.0, 4.85), text="慢时标治理信息", color="#556b8a", lw=1.4, text_offset=(-0.2, 0.12))
    _connect(ax, (9.0, 5.9), (7.0, 4.85), text="快时标任务摘要", color="#556b8a", lw=1.4, text_offset=(0.2, 0.12))
    _connect(ax, (6.0, 3.75), (6.0, 2.65), text="审计触发与回写", color="#7a6419", lw=1.45, text_offset=(1.0, 0.02))

    _label(ax, (10.2, 6.45), "快时标", fs=9.0)
    _label(ax, (4.2, 6.45), "慢时标", fs=9.0)
    _label(ax, (10.25, 4.3), "中时标", fs=9.0)
    _label(ax, (10.35, 2.1), "事件驱动", fs=9.0)
    _label(ax, (6.0, 0.45), "分层联盟链分别承担记账、聚合与治理职责，共同支撑可审计的公共信息板", fs=9.3, color="#425466")

    fig.tight_layout()
    fig.savefig(IMG / "3-new-4b.png", bbox_inches="tight")
    plt.close(fig)


def fig_reputation_loop():
    """图3-5：信誉 状态—动作—审计—奖励 闭环（与五步流程图区分）"""
    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    positions = {
        "state": (5.0, 6.6),
        "action": (7.8, 4.9),
        "audit": (6.8, 2.0),
        "rep": (3.2, 2.0),
        "reward": (2.2, 4.9),
    }
    _paper_box(ax, (4.0, 6.0), 2.0, 0.95, "状态 / 观测", ["本地状态 l_t^i", "链上摘要 g_t"], fc="#fbfcff")
    _paper_box(ax, (6.8, 4.35), 2.0, 0.95, "策略动作", ["卸载选择", "资源比例 / 接收意愿"], fc="#fbfcff")
    _paper_box(ax, (5.8, 1.45), 2.0, 0.95, "链上审计", ["事件一致性检查", "异常行为识别"], fc="#fffdf8", title_fc="#f3efe2")
    _paper_box(ax, (2.2, 1.45), 2.0, 0.95, "信誉更新", ["Rep_i 增减", "长期协作记录"], fc="#fbfcff")
    _paper_box(ax, (1.2, 4.35), 2.0, 0.95, "奖励回写", ["可信项 r_c^i", "影响下一轮策略更新"], fc="#fbfcff")

    _connect(ax, (6.0, 6.25), (6.8, 5.0), color="#4b5d73", lw=1.45)
    _connect(ax, (7.8, 4.35), (6.8, 2.4), color="#4b5d73", lw=1.45)
    _connect(ax, (5.8, 1.9), (4.2, 1.9), color="#4b5d73", lw=1.45)
    _connect(ax, (3.2, 2.4), (2.2, 4.35), color="#4b5d73", lw=1.45)
    _connect(ax, (3.2, 5.0), (4.0, 6.25), color="#4b5d73", lw=1.45)

    _label(ax, (5.0, 0.55), "信誉并不直接替代策略，而是通过审计结果回写奖励，逐步改变节点协作偏好", fs=9.3, color="#425466")

    fig.tight_layout()
    fig.savefig(IMG / "3-new-5b.png", bbox_inches="tight")
    plt.close(fig)


def fig_training_curves():
    """图3-7：收敛曲线（示意）"""
    steps = np.linspace(0, 1.5e6, 200)
    base = 1 - np.exp(-steps / 4e5)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    methods = {
        "BC-CTDE-MAPPO": base * 1.0 + rng.normal(0, 0.02, len(steps)).cumsum() * 0.0001,
        "Trusted-Shared-State": base * 0.92 + rng.normal(0, 0.025, len(steps)).cumsum() * 0.0001,
        "MAPPO-NoChain": base * 0.82 + 0.05 * np.sin(steps / 8e4),
        "Independent PPO": base * 0.68 + 0.12 * np.sin(steps / 5e4),
        "Greedy-Local": np.full_like(steps, 0.45),
    }
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#7f7f7f"]
    for (name, y), c in zip(methods.items(), colors):
        ax.plot(steps / 1e6, y, label=name, lw=2, color=c)
    ax.set_xlabel("训练步数 ($\\times 10^6$)")
    ax.set_ylabel("平均回报（归一化示意）")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("不同方法的训练收敛趋势（示意）")
    fig.tight_layout()
    fig.savefig(IMG / "3-new-7.png", bbox_inches="tight")
    plt.close(fig)


def fig_bar_metrics():
    """图3-8：核心指标柱状对比"""
    labels = ["TCR", "JFI", "时延收益", "能耗收益"]
    x = np.arange(len(labels))
    w = 0.14
    series = {
        "本文": [0.94, 0.91, 0.88, 0.82],
        "TSS": [0.90, 0.86, 0.84, 0.80],
        "M-NC": [0.82, 0.74, 0.72, 0.78],
        "Ind-PPO": [0.74, 0.65, 0.62, 0.75],
        "Greedy": [0.58, 0.52, 0.55, 0.88],
    }
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (k, vals) in enumerate(series.items()):
        ax.bar(x + (i - 2) * w, vals, w, label=k)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("相对得分（示意）")
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    ax.set_title("收敛后核心指标对比（示意）")
    fig.tight_layout()
    fig.savefig(IMG / "3-new-8.png", bbox_inches="tight")
    plt.close(fig)


def fig_scale_load():
    """图3-9：规模 / 负载 / 异构 三组曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    nodes = np.array([20, 40, 60, 80])
    ours = 0.92 - (nodes - 20) * 0.0015
    base1 = 0.88 - (nodes - 20) * 0.003
    base2 = 0.78 - (nodes - 20) * 0.0045
    axes[0].plot(nodes, ours, "o-", label="BC-CTDE-MAPPO", lw=2)
    axes[0].plot(nodes, base1, "s--", label="MAPPO-NoChain")
    axes[0].plot(nodes, base2, "^--", label="Greedy-Local")
    axes[0].set_xlabel("节点规模")
    axes[0].set_ylabel("任务完成率")
    axes[0].legend(fontsize=7)
    axes[0].set_title("网络规模")

    lam = np.array([4, 6, 8])
    axes[1].plot(lam, [0.91, 0.87, 0.79], "o-", label="本文", lw=2)
    axes[1].plot(lam, [0.86, 0.78, 0.62], "s--", label="Ind-PPO", lw=1.5)
    axes[1].set_xlabel("基准到达率 $\\lambda_0$")
    axes[1].set_ylabel("任务完成率")
    axes[1].legend(fontsize=7)
    axes[1].set_title("负载强度")

    het = np.array([0.2, 0.5, 0.8])
    axes[2].plot(het, [0.90, 0.85, 0.78], "o-", label="本文", lw=2)
    axes[2].plot(het, [0.82, 0.70, 0.55], "s--", label="Greedy", lw=1.5)
    axes[2].set_xlabel("异构程度（示意）")
    axes[2].set_ylabel("任务完成率")
    axes[2].legend(fontsize=7)
    axes[2].set_title("节点异构性")

    fig.suptitle("多场景下性能对比（示意）", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(IMG / "3-new-9.png", bbox_inches="tight")
    plt.close(fig)


def fig_ablation_sensitivity():
    """图3-10：左消融右敏感性"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    names = ["完整", "无链上摘要", "无公平项", "无信誉", "无图编码"]
    scores = [1.0, 0.72, 0.81, 0.88, 0.85]
    ax1.barh(names, scores, color=plt.cm.Blues(np.linspace(0.35, 0.85, len(names))))
    ax1.set_xlabel("综合得分（示意）")
    ax1.set_title("消融实验")
    ax1.set_xlim(0, 1.05)

    delay = np.linspace(0.5, 3.0, 12)
    ax2.plot(delay, 0.95 - 0.06 * (delay - 0.5) ** 1.2, "o-", label="任务完成率", color="#c0392b")
    ax2.set_xlabel("链上附加时延 (s)")
    ax2.set_ylabel("性能保持（示意）")
    ax2.legend()
    ax2.set_title("链上时延敏感性")
    fig.suptitle("消融与时延敏感性（示意）", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(IMG / "3-new-10.png", bbox_inches="tight")
    plt.close(fig)


def fig_robustness():
    """图3-11：异常节点比例 vs 完成率"""
    ratio = np.linspace(0, 0.3, 10)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(ratio * 100, 0.92 - 0.35 * ratio ** 1.3, "o-", lw=2, label="BC-CTDE-MAPPO")
    ax.plot(ratio * 100, 0.88 - 0.65 * ratio ** 1.1, "s--", lw=1.8, label="MAPPO-NoChain")
    ax.plot(ratio * 100, 0.85 - 0.85 * ratio ** 0.95, "^--", lw=1.8, label="Independent PPO")
    ax.set_xlabel("异常节点比例 (%)")
    ax.set_ylabel("任务完成率")
    ax.legend()
    ax.set_title("自私/欺骗场景下的鲁棒性（示意）")
    fig.tight_layout()
    fig.savefig(IMG / "3-new-11.png", bbox_inches="tight")
    plt.close(fig)


def main():
    fig_selfish_vs_coop()
    fig_three_layer_framework()
    fig_task_chain_loop()
    fig_modeling_path()
    fig_mappo_flow()
    fig_blockchain_layers()
    fig_reputation_loop()
    fig_training_curves()
    fig_bar_metrics()
    fig_scale_load()
    fig_ablation_sensitivity()
    fig_robustness()
    print("已写入:", IMG)


if __name__ == "__main__":
    main()
