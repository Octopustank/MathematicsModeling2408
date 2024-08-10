import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import dataOutput

plt.rcParams['font.sans-serif'] = ['SimHei'] # 使用黑体显示中文
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False # 正确显示负号

# 读取数据
data = {}
for i in range(1, 15):
    df_one = pd.read_excel(f"./doc/各路段预测24h/路段{i}.xls")
    data[f"路段{i}"] = list(df_one["预测_车流量_模型_1"][-24:])

# 转换数据格式
hours = list(range(24))
df = pd.DataFrame(data, index=hours)
df.index.name = '小时数'


dataOutput.dump(df, 'traffic_data.csv')

# # 准备数据
# sections = list(range(1, 15))
# hours = list(range(24))
# X, Y = np.meshgrid(sections, hours)
# Z = np.array([data[f"路段{section}"] for section in sections]).T

# # 绘制三维图表
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap='viridis')

# # 设置标签
# ax.set_xlabel('路段')
# ax.set_ylabel('小时数')
# ax.set_zlabel('车流量')

# plt.show()