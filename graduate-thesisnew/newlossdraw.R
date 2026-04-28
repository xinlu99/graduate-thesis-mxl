# 载入必要的包
library(ggplot2)
library(tidyr)

# 从指定路径读取数据
data <- read.table("D:/FedRS/R/loss-picture2.txt", header = TRUE, sep = "\t", fill = TRUE)

# 手动设置列名，确保与文件中的列名和顺序一致
colnames(data) <- c("round", 
                    "2_current_tasks_cluster", 
                    "4_current_tasks_cluster", 
                    "6_current_tasks_cluster", 
                    "8_current_tasks_cluster", 
                    "10_current_tasks_cluster", 
                    "no_cluster")

# 将数据转换为长格式
data_long <- pivot_longer(data, cols = !round, names_to = "tasks", values_to = "loss")

# 将任务类型设置为因子，并设置顺序
data_long$tasks <- factor(data_long$tasks, levels = c("2_current_tasks_cluster", 
                                                      "4_current_tasks_cluster", 
                                                      "6_current_tasks_cluster", 
                                                      "8_current_tasks_cluster", 
                                                      "10_current_tasks_cluster", 
                                                      "no_cluster"))

# 绘图
p <- ggplot(data_long, aes(x = round, y = loss, color = tasks, linetype = tasks)) +
  geom_line(size = 1.5) +
  scale_color_manual(values = c("#DB0B49","#DCAD01","#BD70DB","#BCDB70", "#70CFDB", "black")) +
  scale_linetype_manual(values = c("solid", "solid", "solid", "solid", "solid", "solid")) +
  labs(x = "Round", y = "Loss", title = "Loss Across Rounds for Different Image Processing Task Configurations") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 34, face = "bold", hjust = 0.5),        # 标题字体大小
    axis.title.x = element_text(size = 30, face = "bold"),                   # X轴标签字体大小
    axis.title.y = element_text(size = 30, face = "bold"),                   # Y轴标签字体大小
    axis.text.x = element_text(size = 30),                                   # X轴刻度字体大小
    axis.text.y = element_text(size = 30),                                   # Y轴刻度字体大小
    legend.title = element_blank(),                                          # 移除图例标题
    legend.text = element_text(size = 27, face = "bold"),                    # 图例内容字体大小
    legend.position = c(0.8, 0.7)                                            # 将图例放在右侧
  ) +
  guides(color = guide_legend(override.aes = list(size = 2, linetype = "solid"), 
                              keywidth = 4, keyheight = 2),
         linetype = guide_legend(override.aes = list(size = 2, linetype = "solid"), 
                                 keywidth = 4, keyheight = 2))

# 显示图形
print(p)

# 保存图形到指定路径
ggsave("D:/FedRS/R/loss-picture2.pdf", plot = p, width = 18, height = 10)





# 从指定路径读取数据
data <- read.table("D:/FedRS/R/fibon-loss.txt", header = TRUE, sep = "\t", fill = TRUE)

# 手动设置列名，确保与文件中的列名和顺序一致
colnames(data) <- c("round", 
                    "2_current_tasks_cluster", 
                    "4_current_tasks_cluster", 
                    "6_current_tasks_cluster", 
                    "8_current_tasks_cluster", 
                    "10_current_tasks_cluster", 
                    "no_cluster")

# 将数据转换为长格式
data_long <- pivot_longer(data, cols = !round, names_to = "tasks", values_to = "loss")

# 将任务类型设置为因子，并设置顺序
data_long$tasks <- factor(data_long$tasks, levels = c("2_current_tasks_cluster", 
                                                      "4_current_tasks_cluster", 
                                                      "6_current_tasks_cluster", 
                                                      "8_current_tasks_cluster", 
                                                      "10_current_tasks_cluster", 
                                                      "no_cluster"))

# 绘图
p <- ggplot(data_long, aes(x = round, y = loss, color = tasks, linetype = tasks)) +
  geom_line(size = 1.5) +
  scale_color_manual(values = c("#DB0B49","#DCAD01","#BD70DB","#BCDB70", "#70CFDB", "black")) +
  scale_linetype_manual(values = c("solid", "solid", "solid", "solid", "solid", "solid")) +
  labs(x = "Round", y = "Loss", title = "Loss Across Rounds for Different Computing Task Configurations") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 37, face = "bold", hjust = 0.5),        # 标题字体大小
    axis.title.x = element_text(size = 30, face = "bold"),                   # X轴标签字体大小
    axis.title.y = element_text(size = 30, face = "bold"),                   # Y轴标签字体大小
    axis.text.x = element_text(size = 30),                                   # X轴刻度字体大小
    axis.text.y = element_text(size = 30),                                   # Y轴刻度字体大小
    legend.title = element_blank(),                                          # 移除图例标题
    legend.text = element_text(size = 27, face = "bold"),                    # 图例内容字体大小
    legend.position = c(0.8, 0.7)                                            # 将图例放在右侧
  ) +
  guides(color = guide_legend(override.aes = list(size = 2, linetype = "solid"), 
                              keywidth = 4, keyheight = 2),
         linetype = guide_legend(override.aes = list(size = 2, linetype = "solid"), 
                                 keywidth = 4, keyheight = 2))

# 显示图形
print(p)

# 保存图形到指定路径
ggsave("D:/FedRS/R/loss-compute.pdf", plot = p, width = 18, height = 10)
