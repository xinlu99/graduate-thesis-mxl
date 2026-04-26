第四章 4.5 节实验材料包

包含内容：
1. chapter4_section_4_5_experiments.tex
   - 完整 4.5 节 LaTeX 文本
   - 包含与图 4-9 数据一致的消融实验表
   - 图 4-9 使用 subfigure 方式组合 4 张 Python 绘图

2. figures/
   - fig4_9_a_avg_delay_cn.png
   - fig4_9_b_sla_cn.png
   - fig4_9_c_overhead_cn.png
   - fig4_9_d_variance_cn.png

3. data/
   - fig4_9_avg_delay.csv
   - fig4_9_sla_completion.csv
   - fig4_9_online_overhead.csv
   - fig4_9_cross_run_variance.csv
   - fig4_9_ablation_summary.csv
   - figure4_9_raw_data.xlsx（如存在）

4. code/
   - draw_figure4_9_cn.py

说明：
- 图 4-9 的高并发点（平均并发任务数 = 16）与表中数值完全一致。
- 在线决策开销与跨轮次波动直接对应图 4-9(c)(d)。
- Python 绘图脚本会自动选择系统可用中文字体；若本机中文显示异常，请安装 Noto Sans CJK SC 或 SimHei。
