# 第三章实验图表与正文成稿（中文博士论文版 v2）

## 主要更新
- 所有数据均包含5次独立运行的均值与标准差
- 收敛曲线添加阴影带，折线图添加误差棒
- 新增规模扩展实验(20到80节点)
- 新增链上时延敏感性数据(0.5-3.0s)
- 新增单位任务能耗数据
- 参数体系与chapter3.tex完全统一

## 文件说明
- figures/ 包含所有PDF和PNG图
- data/ 包含所有CSV原始数据(含_std列)
- scripts/generate_data.py 数据生成
- scripts/generate_figures.py 图表绘制
- scripts/generate_figures_and_tables.py 一键入口

## 参数体系
- 节点数20, 基线MAPPO/MADDPG/IPPO/Greedy/Random
- 到达率0.3-1.1, 自私比例0%-40%, 500回合x200时隙
