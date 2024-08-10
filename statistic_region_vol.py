"""
统计逐小时门架/路段车流
"""
import pandas as pd
import numpy as np
from datetime import datetime

import dataLoader
import dataOutput


df = pd.read_csv("./doc/表1表2_出入站补全清洗数据.csv")
location = dataLoader.read_location_data()
gates = list(location["gate"].keys())

# 生成时间区间序列
start_time = datetime(2022, 2, 22)
end_time = datetime(2022, 2, 28)
time_intervals = np.linspace(start_time.timestamp(), end_time.timestamp(), num=24 * 6 + 1)  # 每小时一个区间


# 初始化结果列表
results = []

# 按照收费站筛选数据
for gate in gates:
    gate_data = df[df['发生地点'] == gate]

    pass_hist, _ = np.histogram(gate_data['时间'].astype(np.int64), bins=time_intervals)
    
    # 保存结果
    for i in range(len(time_intervals) - 1):
        results.append([gate, datetime.fromtimestamp(time_intervals[i]), pass_hist[i]])

# 转换为DataFrame并保存为CSV
result_df = pd.DataFrame(results, columns=['门架', '时间区间起点', '车流'])
dataOutput.dump(result_df, 'gate_traffic_hourly.csv')