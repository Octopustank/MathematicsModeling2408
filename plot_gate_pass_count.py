"""
绘制车辆通过门架的时间、位置、数量信息的三维分布图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import dataLoader

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['axes.grid'] = False
plt.rcParams['font.sans-serif'] = ['SimHei'] # 使用黑体显示中文
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False # 正确显示负号

# 读取数据
dfs = []
for i in range(1, 31):
    dfs.append(dataLoader.read_raw_data(f"轨迹表{i}.xlsx"))
df = pd.concat(dfs, ignore_index=True)
location = dataLoader.read_location_data()
gates = location["gate"].keys()

# 统计记录车辆通过门架的时间戳
print("statisticting", end='...')
pass_gate_count = {}
for i in range(len(df)):
    row = df.iloc[i]
    if row["事件"] == "门架":
        gate = row["发生地点"]
        if gate in gates:
            if gate not in pass_gate_count:
                pass_gate_count[gate] = []
            pass_gate_count[gate].append(row['时间'])
print("Done.")

# 获取所有时间戳的最大值和最小值
all_times = [time for times in pass_gate_count.values() for time in times]
min_time = min(all_times)
max_time = max(all_times)
all_times = None

# 将时间戳均分区间
bins = np.linspace(min_time, max_time, 61)

# 创建三维图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 初始化 x, y, z 数据列表
x_data = np.arange(1, len(gates) + 1)
y_data = (bins[:-1] + bins[1:]) / 2
z_data = np.zeros((len(x_data), len(y_data)))

# 遍历 gates 和对应的时间戳列表，统计每个区间内的时间戳数量
print("counting", end='...')
for i, gate in enumerate(gates):
    times = pass_gate_count[gate]
    counts, _ = np.histogram(times, bins)
    z_data[i, :] = counts
print("Done.")

# 将时间戳转换为可读的时间格式
y_data_readable = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in y_data]

# 创建网格
X, Y = np.meshgrid(x_data, y_data)

# 绘制三维曲面图
ax.plot_surface(X, Y, z_data.T, cmap='viridis')

# 设置图形标签和标题
ax.set_xlabel('位置索引')
ax.set_ylabel('时间区间中点')
ax.set_zlabel('数量')
ax.set_title('时间、位置、数量信息的三维分布')

# 设置 y 轴刻度标签为可读的时间信息
ax.set_yticks(y_data)
ax.set_yticklabels(y_data_readable, rotation=45, ha='right')

# 显示图形
plt.show()