"""
收费站与区间车流量的统计分析
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

df = pd.read_csv("./doc/表1表2-出入站补全清洗数据.csv")
location = dataLoader.read_location_data()
gates = list(location["gate"].keys())
stations = [one for one in location["sequence"] if one not in gates]


traffic_stats = {station: {"入": 0, "出": 0} for station in stations}

segments = {f"路段{i}": {"入": 0, "出": 0} for i in range(1, len(gates))}

# 遍历数据
for index, row in df.iterrows():
    event = row["事件"]
    location = row["发生地点"]
    
    if event in ["补录入站", "MTC入站", "ETC入站"]:
        if location in traffic_stats:
            traffic_stats[location]["入"] += 1
    elif event in ["补录出站", "MTC出站"]:
        if location in traffic_stats:
            traffic_stats[location]["出"] += 1
    elif event == "门架":
        # 找到当前门架的索引
        current_gate_index = gates.index(location)
        # 更新路段统计
        if current_gate_index < len(gates) - 1:
            segment_name = f"路段{current_gate_index + 1}"
            segments[segment_name]["入"] += 1
        if current_gate_index > 0:
            segment_name = f"路段{current_gate_index}"
            segments[segment_name]["出"] += 1

# 输出结果
print("收费站车流量统计:")
for location, stats in traffic_stats.items():
    print(f"{location}: 入车流量={stats['入']}, 出车流量={stats['出']}")

print("\n路段车流量统计:")
for segment, stats in segments.items():
    print(f"{segment}: 入车流量={stats['入']}, 出车流量={stats['出']}")
