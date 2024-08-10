import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus

import dataLoader
import dataOutput

df = pd.read_csv("./doc/预测的各收费站24小时车流.csv")
distance_gate = dataLoader.read_location_data()["gate"]
distance = {}
for i in range(1,15):
    distance[f"路段{i}"] = distance_gate[f"门架{i + 1}"] - distance_gate[f"门架{i}"]

print(distance)
print(df.head())

# 参数定义
N = 3  # 车道数
M = 140  # 最大限速
R = 150  # 车行间距
E = 10  # 限速差值

A = 1
B = 0.1 * A

# 计算车道车流
density = df.iloc[:, 1:].div([distance[f"路段{i}"] for i in range(1, 15)], axis=1) / N

# 定义问题
prob = LpProblem("SpeedOptimization", LpMinimize)


# 定义变量
num_hours = 24
num_segments = 14
speeds = LpVariable.dicts("Speed", (range(num_hours), range(num_segments)), lowBound=80, upBound=M)
ranges = LpVariable.dicts("Range", (range(num_hours), range(num_segments)), lowBound=R)

# 目标函数：

prob += lpSum(A * (speeds[hour][segment])+\
              B * ranges[hour][segment] / R
               for hour in range(num_hours) for segment in range(num_segments))

# 约束条件
# 相邻路段限速差值约束
for hour in range(num_hours):
    for segment in range(num_segments - 1):
        prob += speeds[hour][segment] - speeds[hour][segment + 1] <= E
        prob += speeds[hour][segment + 1] - speeds[hour][segment] <= E

 # 同一路段相邻时间限速差值约束
for hour in range(num_hours - 1):
    for segment in range(num_segments):
        prob += speeds[hour][segment] - speeds[hour + 1][segment] <= E
        prob += speeds[hour + 1][segment] - speeds[hour][segment] <= E

# 车行间距约束
for hour in range(num_hours):
    for segment in range(num_segments):
        flow = df.iloc[hour, segment + 1]  # 获取车流量
        prob += speeds[hour][segment] >= (flow / N * ranges[hour][segment] + distance[f"路段{segment + 1}"]) / 1000

# 求解问题
prob.solve()

# 结果解析
if LpStatus[prob.status] == 'Optimal':
    speed_results = np.zeros((num_hours, num_segments))
    for hour in range(num_hours):
        for segment in range(num_segments):
            speed_results[hour, segment] = speeds[hour][segment].varValue
    
    # 将所有大于120的值改为120
    speed_results[speed_results > 120] = 120
    
    # 将结果向上取整到最接近的5的倍数
    speed_results = np.ceil(speed_results / 5) * 5
    
    # 将结果转换为 DataFrame
    speed_df = pd.DataFrame(speed_results, columns=[f"路段{i+1}" for i in range(num_segments)])
    speed_df.index.name = '小时'
    
    # 保存到 CSV 文件
    dataOutput.dump(speed_df, "optimized_speeds.csv")
else:
    print("优化问题无解")

    