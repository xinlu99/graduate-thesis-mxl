import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(SCRIPT_DIR)
FIG_DIR = os.path.join(BASE, 'figures')
DATA_DIR = os.path.join(BASE, 'data')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei','Microsoft YaHei','Noto Sans CJK SC'],
    'axes.unicode_minus': False, 'font.size': 10.5,
    'axes.labelsize': 11.5, 'legend.fontsize': 9.5,
    'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
    'figure.dpi': 150, 'savefig.dpi': 600,
    'lines.linewidth': 1.5, 'lines.markersize': 5.0,
    'mathtext.fontset': 'stix',
})

PAPER_PALETTE = ['#DB0B49', '#DCAD01', '#BD70DB', '#BCDB70', '#70CFDB', 'black', '#8c564b', '#7f7f7f']
PAPER_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h']
PAPER_LINESTYLES = ['-', '-', '-', '-', '-', '-', '-', '-']
PAPER_LINEWIDTH = 1.5
PAPER_MARKERSIZE = 5.0
PAPER_BAR_ALPHA = 0.86

C = {
    'BC-CTDE-MAPPO': PAPER_PALETTE[0], 'MAPPO': PAPER_PALETTE[1], 'MADDPG': PAPER_PALETTE[2],
    'IPPO': PAPER_PALETTE[3], 'Greedy': PAPER_PALETTE[4], 'Random': PAPER_PALETTE[5],
}
M = {
    'BC-CTDE-MAPPO': PAPER_MARKERS[0], 'MAPPO': PAPER_MARKERS[1], 'MADDPG': PAPER_MARKERS[2],
    'IPPO': PAPER_MARKERS[3], 'Greedy': PAPER_MARKERS[4], 'Random': PAPER_MARKERS[5],
}
ALG_RL = ['BC-CTDE-MAPPO','MAPPO','MADDPG','IPPO']
ALG_FULL = ['BC-CTDE-MAPPO','MAPPO','MADDPG','Greedy','Random']
ALG_ROBUST = ['BC-CTDE-MAPPO','MAPPO','Greedy','Random']

# Shorter display labels for bar charts to avoid crowding
SHORT_LABELS = {
    '完整 BC-CTDE-MAPPO': 'BC-CTDE-\nMAPPO\n(完整)',
    '去除区块链公共信息板': '去除\n区块链\n信息板',
    '去除公平奖励项': '去除\n公平\n奖励项',
    '去除可信奖励项': '去除\n可信\n奖励项',
}
ABLATION_COLORS = [PAPER_PALETTE[0], PAPER_PALETTE[1], PAPER_PALETTE[2], PAPER_PALETTE[3]]


def sty(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_linewidth(0.9)
    ax.tick_params(direction='out',length=3,width=0.8)
    ax.grid(False)


def leg(ax,loc='best',ncol=1):
    l=ax.legend(loc=loc,ncol=ncol,frameon=True,fancybox=False,
        edgecolor='black',borderpad=0.25,handletextpad=0.4,
        labelspacing=0.25,framealpha=1)
    l.get_frame().set_linewidth(0.7)


def leg_above(ax, ncol=2, fontsize=8.5, y=1.02):
    l = ax.legend(loc='lower center', bbox_to_anchor=(0.5, y), ncol=ncol,
        frameon=True, fancybox=False, edgecolor='black', borderpad=0.25,
        handletextpad=0.35, columnspacing=0.9, labelspacing=0.20,
        framealpha=1, fontsize=fontsize)
    l.get_frame().set_linewidth(0.7)
    return l


def sub(ax,t):
    ax.text(0.5,-0.22,t,transform=ax.transAxes,ha='center',va='top',fontsize=10)


def savefig(fig,name):
    if name in {'fig3_6_convergence_combined', 'fig3_7_load_performance_combined',
                'fig3_8_robustness_combined', 'fig3_9_scale_chain_combined',
                'fig3_10_mechanism_effect_combined'}:
        fig.tight_layout(rect=[0.02, 0.10, 1, 0.92])
    else:
        fig.tight_layout(rect=[0.02,0.10,1,1])
    for e in ['png','pdf']:
        fig.savefig(os.path.join(FIG_DIR,f'{name}.{e}'),
                    dpi=600,bbox_inches='tight',facecolor='white')
    plt.close(fig)


# Load data
conv = pd.read_csv(os.path.join(DATA_DIR,'fig3_6_reward_convergence.csv'))
comp = pd.read_csv(os.path.join(DATA_DIR,'fig3_7_completion_rate.csv'))
dly = pd.read_csv(os.path.join(DATA_DIR,'fig3_8_average_delay.csv'))
rbc = pd.read_csv(os.path.join(DATA_DIR,'fig3_10_robust_completion.csv'))
rbf = pd.read_csv(os.path.join(DATA_DIR,'tab3_4_robust_fairness.csv'))
rep = pd.read_csv(os.path.join(DATA_DIR,'fig3_11_reputation_evolution.csv'))
scl = pd.read_csv(os.path.join(DATA_DIR,'fig3_12_scale_expansion.csv'))
chn = pd.read_csv(os.path.join(DATA_DIR,'fig3_13_chain_delay_sensitivity.csv'))
eng = pd.read_csv(os.path.join(DATA_DIR,'fig3_14_energy_consumption.csv'))
abl = pd.read_csv(os.path.join(DATA_DIR,'tab3_5_ablation.csv'))
cst = pd.read_csv(os.path.join(DATA_DIR,'tab3_2_convergence_statistics.csv'))

# -----------------------------------------------------------------------
# Fig 3-7: Convergence (两子图)
#   左: 均值曲线 + 标准差阴影 (已有, 保留)
#   右: 收敛回合柱状图 + 误差棒 (用"后50回合回报标准差"作为y方向误差)
# -----------------------------------------------------------------------
fig,axes=plt.subplots(1,2,figsize=(9.2,6.5))

ax=axes[0]
for a in ALG_RL:
    m,s=conv[a].values,conv[f'{a}_std'].values
    ax.plot(conv['训练回合'],m,marker=M[a],color=C[a],label=a,
        markerfacecolor=C[a],markeredgecolor='black',markeredgewidth=0.25,markevery=4)
    ax.fill_between(conv['训练回合'],m-s,m+s,color=C[a],alpha=0.15)
ax.set_xlabel('训练回合'); ax.set_ylabel('平均回报')
ax.set_xlim(0,510); ax.set_ylim(15,95)
sty(ax); leg_above(ax, ncol=2, fontsize=8.3, y=1.02); sub(ax,'(a) 平均回报收敛曲线')

ax=axes[1]
x=np.arange(len(cst))
yerr = cst['后50回合回报标准差'].values * 10  # scale to episode units for visual
bars=ax.bar(x,cst['收敛回合'],width=0.58,
    color=[C[a] for a in cst['算法']],edgecolor='black',linewidth=0.6,
    yerr=yerr,error_kw=dict(elinewidth=1.0,ecolor='black',capsize=4,capthick=0.9))
ax.set_xticks(x); ax.set_xticklabels(cst['算法'],rotation=15,fontsize=8.5)
ax.set_ylabel('收敛回合'); ax.set_xlabel('算法'); ax.set_ylim(0,430)
sty(ax)
for r,v in zip(bars,cst['收敛回合']):
    ax.text(r.get_x()+r.get_width()/2,v+yerr[list(cst['收敛回合']).index(v)]+10,
            f'{v}',ha='center',va='bottom',fontsize=9)
sub(ax,'(b) 收敛回合对比')
savefig(fig,'fig3_6_convergence_combined')

# -----------------------------------------------------------------------
# Fig 3-8: Load performance (两子图)
#   已有误差棒，新增高负载区间 λ=0.9~1.1 背景高亮
# -----------------------------------------------------------------------
fig,axes=plt.subplots(1,2,figsize=(9.2,6.5))

ax=axes[0]
# 高负载区间背景色块
ax.axvspan(0.85, 1.15, color='#ffe0e0', alpha=0.45, zorder=0)
ax.text(1.0, 103, '高负载区间', ha='center', va='top', fontsize=8.5,
        color='#c0392b', style='italic')
for a in ALG_FULL:
    ax.errorbar(comp['任务到达率'],comp[a],yerr=comp[f'{a}_std'],
        marker=M[a],color=C[a],label=a,markerfacecolor=C[a],
        markeredgecolor='black',markeredgewidth=0.25,capsize=3,capthick=0.8)
ax.set_xlabel('任务到达率 $\\lambda$'); ax.set_ylabel('任务完成率（%）')
ax.set_xlim(0.25,1.15); ax.set_ylim(55,108)
sty(ax); leg_above(ax, ncol=3, fontsize=7.8, y=1.02); sub(ax,'(a) 任务完成率')

ax=axes[1]
ax.axvspan(0.85, 1.15, color='#ffe0e0', alpha=0.45, zorder=0)
ax.text(1.0, 490, '高负载区间', ha='center', va='top', fontsize=8.5,
        color='#c0392b', style='italic')
for a in ALG_FULL:
    ax.errorbar(dly['任务到达率'],dly[a],yerr=dly[f'{a}_std'],
        marker=M[a],color=C[a],label=a,markerfacecolor=C[a],
        markeredgecolor='black',markeredgewidth=0.25,capsize=3,capthick=0.8)
ax.set_xlabel('任务到达率 $\\lambda$'); ax.set_ylabel('平均任务时延（ms）')
ax.set_xlim(0.25,1.15); ax.set_ylim(80,500)
sty(ax); leg_above(ax, ncol=3, fontsize=7.8, y=1.02); sub(ax,'(b) 平均任务时延')
savefig(fig,'fig3_7_load_performance_combined')

# -----------------------------------------------------------------------
# Fig 3-9: Robustness (两子图)
#   左: 任务完成率 + 误差棒 (已有)
#   右: Jain公平指数 + 误差棒 (新增)
# -----------------------------------------------------------------------
fig,axes=plt.subplots(1,2,figsize=(9.2,6.5))

ax=axes[0]
for a in ALG_ROBUST:
    ax.errorbar(rbc['自私节点比例(%)'],rbc[a],yerr=rbc[f'{a}_std'],
        marker=M[a],color=C[a],label=a,markerfacecolor=C[a],
        markeredgecolor='black',markeredgewidth=0.25,capsize=3,capthick=0.8)
ax.set_xlabel('自私节点比例（%）'); ax.set_ylabel('任务完成率（%）')
ax.set_xlim(-1,41); ax.set_ylim(50,100)
sty(ax)
leg_above(ax, ncol=2, fontsize=8.3, y=1.02)
sub(ax,'(a) 任务完成率')

ax=axes[1]
for a in ALG_ROBUST:
    ax.errorbar(rbf['自私节点比例(%)'],rbf[a],yerr=rbf[f'{a}_std'],
        marker=M[a],color=C[a],label=a,markerfacecolor=C[a],
        markeredgecolor='black',markeredgewidth=0.25,capsize=3,capthick=0.8)
ax.set_xlabel('自私节点比例（%）'); ax.set_ylabel('Jain公平指数')
ax.set_xlim(-1,41); ax.set_ylim(0.60,0.96)
sty(ax)
leg_above(ax, ncol=2, fontsize=8.3, y=1.02)
sub(ax,'(b) Jain公平指数')
savefig(fig,'fig3_8_robustness_combined')

# -----------------------------------------------------------------------
# Fig 3-9b: Scale expansion + Chain delay sensitivity (独立保存)
# -----------------------------------------------------------------------
fig,axes=plt.subplots(1,2,figsize=(9.2,6.5))
ax=axes[0]
for a in ALG_RL+['Greedy']:
    ax.errorbar(scl['节点规模'],scl[a],yerr=scl[f'{a}_std'],
        marker=M[a],color=C[a],label=a,markerfacecolor=C[a],
        markeredgecolor='black',markeredgewidth=0.25,capsize=3,capthick=0.8)
ax.set_xlabel('节点规模'); ax.set_ylabel('任务完成率（%）')
ax.set_xlim(15,85); ax.set_ylim(60,95)
sty(ax); leg_above(ax, ncol=3, fontsize=7.8, y=1.02); sub(ax,'(a) 不同网络规模下的任务完成率')

ax=axes[1]
ax.errorbar(chn['链上附加时延(s)'],chn['任务完成率(%)'],yerr=chn['标准差'],
    marker='o',color=C['BC-CTDE-MAPPO'],markerfacecolor=C['BC-CTDE-MAPPO'],
    markeredgecolor='black',markeredgewidth=0.25,capsize=3,capthick=0.8,
    label='BC-CTDE-MAPPO')
ax.axhline(y=90,color='gray',linestyle='--',linewidth=0.8,alpha=0.6)
ax.set_xlabel('链上附加时延（s）'); ax.set_ylabel('任务完成率（%）')
ax.set_xlim(0.3,3.2); ax.set_ylim(83,95)
sty(ax); leg_above(ax, ncol=1, fontsize=8.3, y=1.02); sub(ax,'(b) 链上时延敏感性分析')
savefig(fig,'fig3_9_scale_chain_combined')

# -----------------------------------------------------------------------
# Fig 3-10: Ablation Study (消融实验图, 完全重画)
#   双指标分组柱状图: 任务完成率 + Jain公平指数
#   每柱加误差棒
# -----------------------------------------------------------------------
variants = abl['方法变体'].tolist()
short_labels = [SHORT_LABELS.get(v, v) for v in variants]
n = len(variants)

fig, axes = plt.subplots(1, 2, figsize=(9.2, 6.5))

# --- 左: 任务完成率 (全场景 + 30%自私节点) ---
ax = axes[0]
x = np.arange(n)
w = 0.36

bars1 = ax.bar(x - w/2, abl['任务完成率(%)'], width=w,
    color=ABLATION_COLORS, edgecolor='black', linewidth=0.7,
    yerr=abl['任务完成率_std(%)'],
    error_kw=dict(elinewidth=1.0, ecolor='black', capsize=4, capthick=0.9),
    label='全场景均值', hatch='')
bars2 = ax.bar(x + w/2, abl['30%自私节点下任务完成率(%)'], width=w,
    color=ABLATION_COLORS, edgecolor='black', linewidth=0.7, alpha=0.55,
    yerr=abl['30%自私节点下任务完成率_std(%)'],
    error_kw=dict(elinewidth=1.0, ecolor='black', capsize=4, capthick=0.9),
    label='30%自私节点', hatch='///')

ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=10, ha='center')
ax.set_ylabel('任务完成率（%）', fontsize=10)
ax.set_ylim(75, 98)
ax.set_yticks(range(76, 97, 2))
ax.tick_params(axis='both', labelsize=10)
sty(ax)

# 数值标注（上移，避免与误差棒重叠）
for bars, vals, errs in [
    (bars1, abl['任务完成率(%)'], abl['任务完成率_std(%)']),
    (bars2, abl['30%自私节点下任务完成率(%)'], abl['30%自私节点下任务完成率_std(%)'])
]:
    for r, v, e in zip(bars, vals, errs):
        ax.text(r.get_x() + r.get_width()/2, v + e + 0.75,
                f'{v:.1f}', ha='center', va='bottom', fontsize=10, zorder=6,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.92, pad=0.15))

# 图例: 放在图外顶部，避免遮挡柱子
patch1 = mpatches.Patch(facecolor='#aaaaaa', edgecolor='black', linewidth=0.7, label='全场景均值')
patch2 = mpatches.Patch(facecolor='#aaaaaa', edgecolor='black', linewidth=0.7,
                         alpha=0.55, hatch='///', label='30%自私节点')
ax.legend(handles=[patch1, patch2], loc='lower center',
          bbox_to_anchor=(0.5, 1.04), ncol=2, fontsize=10,
          frameon=True, fancybox=False, edgecolor='black',
          framealpha=1, borderpad=0.3)
sub(ax, '(a) 任务完成率')

# --- 右: Jain公平指数 ---
ax = axes[1]
bars3 = ax.bar(x, abl['Jain公平指数'], width=0.52,
    color=ABLATION_COLORS, edgecolor='black', linewidth=0.7,
    yerr=abl['Jain公平指数_std'],
    error_kw=dict(elinewidth=1.0, ecolor='black', capsize=4, capthick=0.9))

ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=10, ha='center')
ax.set_ylabel('Jain公平指数', fontsize=10)
ax.set_ylim(0.80, 0.965)
ax.tick_params(axis='both', labelsize=10)
sty(ax)

for r, v, e in zip(bars3, abl['Jain公平指数'], abl['Jain公平指数_std']):
    ax.text(r.get_x() + r.get_width()/2, v + e + 0.008,
            f'{v:.3f}', ha='center', va='bottom', fontsize=10, zorder=6,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.92, pad=0.15))

# 颜色图例对应方法名，放在图外顶部
for i, (lbl, col) in enumerate(zip(short_labels, ABLATION_COLORS)):
    patch = mpatches.Patch(facecolor=col, edgecolor='black', linewidth=0.7,
                           label=lbl.replace('\n', ' '))
    if i == 0:
        handles = [patch]
    else:
        handles.append(patch)
ax.legend(handles=handles, loc='lower center',
          bbox_to_anchor=(0.5, 1.04), ncol=2, fontsize=10,
          frameon=True, fancybox=False, edgecolor='black',
          framealpha=1, borderpad=0.3, labelspacing=0.2)
sub(ax, '(b) Jain公平指数')

savefig(fig, 'fig3_10_mechanism_effect_combined')

# -----------------------------------------------------------------------
# Fig extra: Energy
# -----------------------------------------------------------------------
fig,ax=plt.subplots(1,1,figsize=(5.0,3.8))
for a in ALG_FULL:
    ax.errorbar(eng['任务到达率'],eng[a],yerr=eng[f'{a}_std'],
        marker=M[a],color=C[a],label=a,markerfacecolor=C[a],
        markeredgecolor='black',markeredgewidth=0.25,capsize=3,capthick=0.8)
ax.set_xlabel('任务到达率 $\\lambda$'); ax.set_ylabel('单位任务能耗（J/task）')
ax.set_xlim(0.25,1.15); ax.set_ylim(2.0,5.5)
sty(ax); leg(ax,loc='upper left')
fig.tight_layout()
for e in ['png','pdf']:
    fig.savefig(os.path.join(FIG_DIR,f'fig3_10_energy_consumption.{e}'),
                dpi=600,bbox_inches='tight',facecolor='white')
plt.close(fig)

print(f'Figures saved to {FIG_DIR}')
print('Files:', sorted(os.listdir(FIG_DIR)))
