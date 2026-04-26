# 第三章实验图表与正文成稿（中文博士论文版）

本资源包包含两部分内容：
1. 更接近参考图风格的中文子图排版版本；
2. 可直接用于论文的“3.5 实验结果与分析”正文初稿。

## 主要文件
- `figures/fig3_6_convergence_combined.pdf`：图3-6，不同训练算法的收敛性对比
- `figures/fig3_7_load_performance_combined.pdf`：图3-7，不同任务到达率下的系统性能对比
- `figures/fig3_8_robustness_combined.pdf`：图3-8，不同自私节点比例下的系统鲁棒性对比
- `figures/fig3_9_mechanism_effect_combined.pdf`：图3-9，信誉机制效果与消融实验对比
- `chapter3_figures_insert.tex`：上述图片的 LaTeX 插入代码
- `chapter3_tables.tex`：表格 LaTeX 代码
- `chapter3_section_3_5.tex`：第三章“实验结果与分析”正文成稿
- `data/`：所有图表对应的原始数据 CSV
- `scripts/generate_figures_and_tables.py`：完整复现脚本

## 说明
当前数据为示例性仿真结果，适合用于论文写作、图表版式定稿和结果分析框架搭建。若后续替换为真实数据，只需更新 `data/` 中对应 CSV，再重新运行脚本即可。
