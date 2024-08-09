import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import dataLoader
import dataOutput

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['axes.grid'] = False
plt.rcParams['font.sans-serif'] = ['SimHei'] # 使用黑体显示中文
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False # 正确显示负号

df = pd.read_csv("./doc/表1表2_出入站补全清洗数据.csv")
location = dataLoader.read_location_data()
gates = list(location["gate"].keys())
stations = [one for one in location["sequence"] if one not in gates]

# 生成时间区间序列
start_time = datetime(2022, 2, 22)
end_time = datetime(2022, 2, 28)
time_intervals = np.linspace(start_time.timestamp(), end_time.timestamp(), num=24 * 6 + 1)  # 每小时一个区间


# 初始化结果列表
results = []

# 按照收费站筛选数据
for station in stations:
    station_data = df[df['发生地点'] == station]
    
    # 筛选入站和出站事件
    entry_data = station_data[station_data['事件'].str.contains('入站')]
    exit_data = station_data[station_data['事件'].str.contains('出站')]

    entry_hist, exit_hist = [], []
    if len(entry_data['时间']) > 0:
        # 统计入站车流
        entry_hist, _ = np.histogram(entry_data['时间'].astype(np.int64), bins=time_intervals)
    if len(exit_data['时间']) > 0:
        # 统计出站车流
        exit_hist, _ = np.histogram(exit_data['时间'].astype(np.int64), bins=time_intervals)
    
    # 保存结果
    for i in range(len(time_intervals) - 1):
        results.append([station, datetime.fromtimestamp(time_intervals[i]), entry_hist[i], exit_hist[i]])

# 转换为DataFrame并保存为CSV
result_df = pd.DataFrame(results, columns=['收费站', '时间区间起点', '入站车流', '出站车流'])
dataOutput.dump(result_df, 'station_traffic_hourly.csv')