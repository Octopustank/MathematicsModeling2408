### 数据预读取
```python
"""
数据加载模块
"""

import pandas as pd
from datetime import datetime
import json
import os
import sys

PATH = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(PATH, 'cache')
DATA = os.path.join(PATH, 'data')
TRACE_TABLE = os.path.join(DATA, 'track_table')

def read_raw_data(file_name: str, chache=True) -> pd.DataFrame:
    """
    读取原始数据文件，返回 DataFrame。如果已经存在缓存文件，则直接读取缓存文件。

    param
    - file_name: xlsx文件名
    - chache: 是否使用缓存，默认为True

    return
    - DataFrame
    """
    print(f"Loading data from {file_name}", end='...')

    file_path = os.path.join(TRACE_TABLE, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    if not file_path.endswith('.xlsx'):
        raise ValueError(f"File {file_path} is not a xlsx file")

    # 判断缓存文件
    cache_file_name = ".".join([os.path.splitext(file_name)[0], 'csv'])
    cache_file = os.path.join(CACHE, cache_file_name)
    if not os.path.exists(CACHE):
        os.makedirs(CACHE)
    if os.path.exists(cache_file) and chache: # 有缓存文件，直接读取
        print("Cache found, using cache.")
        return pd.read_csv(cache_file)

    print("Reading data", end='...')
    df = pd.read_excel(file_path)

    # 初始化结果列表
    results = []
    track_temp = []

    # 临时变量存储当前车辆信息，用于解析数据块
    current_vehicle = None
    current_model = None
    current_track = None

    cnt = 0

    # 遍历每一行
    for index, row in df.iterrows():
        if pd.notna(row['车辆编号']): # 数据块首行，获取车辆信息
            # 处理数据块的第一行
            current_vehicle = row['车辆编号']
            current_model = row['收费站/门架编号'].split('：')[1]  # 提取车型
            current_track = None
            continue  # 跳过数据块的第一行
        if pd.notna(row['轨迹编号']): # 刷新轨迹编号
            current_track = row['轨迹编号']

            time_items = [item[4] for item in track_temp if item[4] is not None]
            time_max = max(time_items) if len(time_items) > 0 else 0
            time_min = min(time_items) if len(time_items) > 0 else 0
            if len(track_temp) > 0 and\
                time_max <= datetime.strptime("2022-02-27 23:59:59", "%Y-%m-%d %H:%M:%S").timestamp() and\
                time_min >= datetime.strptime("2022-02-22 0:0:0", "%Y-%m-%d %H:%M:%S").timestamp(): # 有效数据
                results.extend(track_temp)
            else: cnt += 1
            track_temp = []
            
        if (current_track is not None): # 只有读取到了轨迹信息才会进行数据处理，规避了轨迹编号缺失的情况（否则会受先前的影响）
            # 读取数据行信息
            event = row['信息类型'].replace('门架信息', '门架')
            location = row['收费站/门架编号'] if row['收费站/门架编号'] != '其他' else None
            time = row['记录时间'] if row['记录时间'] != '——' else None
            if time: # 转换时间格式为时间戳
                time = datetime.strptime(time, '%d/%m/%Y %H:%M:%S').timestamp()
            track_temp.append([current_vehicle, current_model, current_track, event, time, location])

    time_items = [item[4] for item in track_temp if item[4] is not None]
    time_max = max(time_items) if len(time_items) > 0 else 0
    if len(track_temp) > 0 and\
        time_max <= datetime.strptime("2022-02-27 23:59:59", "%Y-%m-%d %H:%M:%S").timestamp() and\
        time_min >= datetime.strptime("2022-02-22 0:0:0", "%Y-%m-%d %H:%M:%S").timestamp(): # 有效数据
        results.extend(track_temp)
    else: cnt += 1

    track_temp = []

    # 创建 DataFrame
    result_df = pd.DataFrame(results, columns=['车辆编号', '车型', '轨迹编号', '事件', '时间', '发生地点'])
    result_df.to_csv(cache_file, index=False)
    print("Done.")
    return result_df

def load_cache() -> None:
    for i in range(1, 31):
        read_raw_data(f"轨迹表{i}.xlsx", chache=False)

def read_location_data() -> dict:
    """
    读取位置数据文件，返回字典
    
    return
    - dict
        - gate: 门架编号和位置的字典
        - sequence: 门架和收费站的顺序列表
    """
    print("Loading location data", end='...')
    location_data_path = os.path.join(DATA, "location.json")
    if not os.path.exists(location_data_path):
        raise FileNotFoundError(f"File {location_data_path} not found")
    with open(location_data_path, 'r', encoding="utf-8") as f:
        location_data = json.load(f)
    print("Done.")

    return location_data
```
### 数据输出
```python
"""
输出结果文件
"""
import pandas as pd
import os

PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(PATH, 'output')

def dump(df: pd.DataFrame, file_name: str) -> None:
    """
    将 DataFrame 输出到文件

    param
    - df: DataFrame
    - file_name: 输出文件名
    """
    output_path = os.path.join(OUTPUT, file_name)
    df.to_csv(output_path, index=False)
    print(f"Data dumped to {output_path}")
```

### 统计门架漏失/正常统计结果的各项数据

```python
"""
统计门架漏失/正常统计结果的各项数据
"""
import pandas as pd
import dataLoader

dfs = []
for i in range(1, 31):
    dfs.append(dataLoader.read_raw_data(f"轨迹表{i}.xlsx"))
df = pd.concat(dfs, ignore_index=True)

# 记录所有门架事件的列表
events = []

# 按照车辆编号和轨迹编号分组
grouped = df.groupby(['车辆编号', '轨迹编号'])

for (vehicle_id, track_id), group in grouped:
    # 过滤出门架事件
    gantry_events = group[group['事件'] == '门架'].sort_values(by='时间')
    
    # 获取所有的门架名称
    gantry_names = gantry_events['发生地点'].tolist()
    
    # 检查门架是否连续
    for i in range(len(gantry_names) - 1):
        current_gantry = gantry_names[i]
        next_gantry = gantry_names[i + 1]
        
        # 提取门架编号
        current_gantry_num = int(current_gantry[2:])
        next_gantry_num = int(next_gantry[2:])
        
        # 如果门架编号不连续，记录缺失的门架
        if next_gantry_num != current_gantry_num + 1:
            for missing_gantry_num in range(current_gantry_num + 1, next_gantry_num):
                missing_gantry_name = f"门架{missing_gantry_num}"
                
                # 推断缺失门架的时间
                missing_time = (gantry_events.iloc[i]['时间'] + gantry_events.iloc[i + 1]['时间']) / 2
                
                events.append({
                    '门架名称': missing_gantry_name,
                    '状态': '漏过',
                    '车辆编号': vehicle_id,
                    '车型': group.iloc[0]['车型'],
                    '轨迹编号': track_id,
                    '时间戳': missing_time
                })
    
    # 记录正常的门架事件
    for _, row in gantry_events.iterrows():
        events.append({
            '门架名称': row['发生地点'],
            '状态': '正常',
            '车辆编号': vehicle_id,
            '车型': row['车型'],
            '轨迹编号': track_id,
            '时间戳': row['时间']
        })

events_df = pd.DataFrame(events)

print(events_df)
events_df.to_csv('gate_condition.csv', index=False)
```
### 对每个收费站的常规车道数量和应急系统数量进行关于总成本的优化

```python
"""
对每个收费站的常规车道数量和应急系统数量进行关于总成本的优化
> 数据输出中，应急系统关闭时间 指的是应急系统开启状态的最后一个小时。例如开启关闭分别为8和9，则实际开启了2小时。
"""

import pandas as pd
import pulp

import dataOutput

# 定义常量
C_x = 24 * 6 * 25.8  # 每条常规车道的成本
C_y = 414.8   # 每套应急收费系统每小时的成本
P = 3600 / (0.25 * 3.5 + 0.75 * 19)  # 每条车道每小时能处理的车流量
P_e = P  # 每套应急系统每小时能处理的车流量

def optimize_cost(D, L, R):
    # 创建问题实例
    problem = pulp.LpProblem("Minimize_Cost", pulp.LpMinimize)

    # 定义变量
    x = pulp.LpVariable("x", lowBound=1, cat='Integer')  # 常规车道数量
    y = pulp.LpVariable("y", lowBound=0, cat='Integer')  # 应急系统数量

    # 目标函数（总费用）
    problem += C_x * x + C_y * y * (R - L)

    # 约束条件
    for t in range(24 * 6):
        hour_of_day = t % 24
        if L <= hour_of_day <= R:  # 开启应急系统，应急系统和常规车道同时处理车流
            problem += (x * P + y * P_e >= D[t])
        else:
            problem += (x * P >= D[t])

    # 求解问题
    problem.solve()

    # 返回结果
    return pulp.LpStatus[problem.status], x.varValue, y.varValue, pulp.value(problem.objective)

if __name__ == '__main__':
    df = pd.read_csv("./doc/表1表2_各收费站逐小时进出车流.csv")
    stations = df["收费站"].unique()
    results = []

    for station in stations:
        D = list(map(lambda x: x * 281689 / 17542, (df[df["收费站"] == station]["入站车流"] + df[df["收费站"] == station]["出站车流"]).values))
        min_cost = 999999
        rec = []

        for L in range(24):  # 应急系统开启
            for R in range(L, 24):  # 应急系统关闭
                status, x, y, cost = optimize_cost(D, L, R)  # 求解最优

                if status == 'Optimal' and cost < min_cost:  # 如果找到更优解
                    min_cost = cost
                    rec = [x, y, L, R]

        results.append([station, rec[2], rec[3], rec[0], rec[1], min_cost])

    # 保存结果到CSV文件
    results_df = pd.DataFrame(results, columns=["收费站", "应急系统开启时间", "应急系统关闭时间", "常规车道数量", "应急系统数量", "总费用"])
    dataOutput.dump(results_df, "优化结果.csv")

```
### 对各路段逐小时的限速优化求解

```python
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
```
### 绘制车辆通过门架的时间、位置、数量信息的三维分布图

```python
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
# 生成时间区间序列
start_time = datetime(2022, 2, 22)
end_time = datetime(2022, 2, 28)

# 将时间戳均分区间
bins = np.linspace(start_time.timestamp(), end_time.timestamp(), num=24 * 6 + 1)

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
```
### 从`SPASS`输出文件读取数据

```python
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
```
### 搜索确认进出收费站，并修补或删除行程的“补录入站”和“补录出站”的“发生地点”

```python
"""
搜索确认进出收费站，并修补或删除行程的“补录入站”和“补录出站”的“发生地点”
"""

import pandas as pd
import numpy as np

import dataLoader
import dataOutput

dfs = []
for i in range(1, 3):
    dfs.append(dataLoader.read_raw_data(f"轨迹表{i}.xlsx"))
df = pd.concat(dfs, ignore_index=True)

location = dataLoader.read_location_data()

# 提取所有“补录入站”的记录
bu_lu_ru_zhan_records = df[df['事件'] == '补录入站']

# 速度（米/秒）
speed = 120 / 3.6

entry_results = []

print("begin to search missing enrty station...")
for index, record in bu_lu_ru_zhan_records.iterrows():
    vehicle_id = record['车辆编号']
    track_id = record['轨迹编号']

    # 找到相同车辆编号和轨迹编号的第一个门架事件
    gate_events = df[(df['车辆编号'] == vehicle_id) & (df['轨迹编号'] == track_id) & (df['事件'] == '门架')]
    
    # 如果没有门架事件，跳过该记录
    if gate_events.empty:
        continue
    
    first_gate_event = gate_events.iloc[0]

    # 计算时间差（秒）
    time_diff = first_gate_event['时间'] - record['时间']
    
    # 计算最大行进距离（米）
    max_distance = time_diff * speed
    
    # 从第一个门架往前遍历，找到可能的收费站
    first_gate_location = first_gate_event['发生地点']
    first_gate_index = location['sequence'].index(first_gate_location)
    
    distance_covered = 0
    possible_stations = []
    
    for i in range(first_gate_index, -1, -1):
        loc = location['sequence'][i]
        if '收费站' in loc:
            possible_stations.append(loc)
        if '门架' in loc:
            distance_covered = location["gate"][first_gate_location] - location["gate"][loc]
        if distance_covered > max_distance:
            break

    if len(possible_stations) == 0:
        continue
    else:
        possible_station = possible_stations[-1]
    
    entry_results.append({
        '车辆编号': vehicle_id,
        '轨迹编号': track_id,
        '可能的收费站': possible_station
    })
print("search missing enrty station done.")
df_gatesearch = pd.DataFrame(entry_results)
dataOutput.dump(df_gatesearch, 'enrty_search.csv')

# 确定“补录出站”的收费站
exit_results = []

print("begin to search missing exit station...")
for vehicle_id, track_id in df[['车辆编号', '轨迹编号']].drop_duplicates().values:
    # 获取该行程的所有记录
    trip_records = df[(df['车辆编号'] == vehicle_id) & (df['轨迹编号'] == track_id)]
    
    # 找到最后一个门架
    gate_events = trip_records[trip_records['事件'] == '门架']['发生地点']

    # 如果没有门架事件，跳过该记录
    if gate_events.empty:
        continue
    
    last_gate = gate_events.iloc[-1]
    
    # 定位最后一个门架的位置
    sequence = location['sequence']
    last_gate_index = sequence.index(last_gate)
    
    # 找到下一个收费站
    next_station = None
    for i in range(last_gate_index + 1, len(sequence)):
        if sequence[i].startswith('收费站'):
            next_station = sequence[i]
            break
    exit_results.append({
        '车辆编号': vehicle_id,
        '轨迹编号': track_id,
        '可能的收费站': next_station,
        '填补的时间': df[(df['车辆编号'] == vehicle_id) & (df['轨迹编号'] == track_id) & (df['事件'] == '门架')].iloc[-1]['时间']
    })
print("search missing exit station done.")

# 将 exit_results 列表保存到 exit_search.csv 文件中
df_exitsearch = pd.DataFrame(exit_results)
dataOutput.dump(df_exitsearch, 'exit_search.csv')

# 将 entry_results 和 exit_results 转换为字典，方便查找
entry_dict = {(result['车辆编号'], result['轨迹编号']): result['可能的收费站'] for result in entry_results}
exit_dict = {(result['车辆编号'], result['轨迹编号']): (result['可能的收费站'], result['填补的时间']) for result in exit_results}

# 记录删除的行程数
deleted_trips_count = 0

# 遍历每一个行程，修补“发生地点”
print("begin to fill missing station...")
for vehicle_id, track_id in df[['车辆编号', '轨迹编号']].drop_duplicates().values:
    trip_key = (vehicle_id, track_id)
    
    # 检查是否需要修补“补录入站”和“补录出站”
    needs_entry_fix = df[(df['车辆编号'] == vehicle_id) & (df['轨迹编号'] == track_id) & (df['事件'] == '补录入站')]['发生地点'].isnull().any()
    needs_exit_fix = df[(df['车辆编号'] == vehicle_id) & (df['轨迹编号'] == track_id) & (df['事件'] == '补录出站')]['发生地点'].isnull().any()
    
    # 如果需要修补，且对应修补方案中不存在，则删除该行程
    if (needs_entry_fix and trip_key not in entry_dict) or (needs_exit_fix and trip_key not in exit_dict):
        df = df[~((df['车辆编号'] == vehicle_id) & (df['轨迹编号'] == track_id))]
        deleted_trips_count += 1
    else:
        # 修补“补录入站”的“发生地点”
        if needs_entry_fix and trip_key in entry_dict:
            df.loc[(df['车辆编号'] == vehicle_id) & (df['轨迹编号'] == track_id) & (df['事件'] == '补录入站'), '发生地点'] = entry_dict[trip_key]
        
        # 修补“补录出站”的“发生地点”
        if needs_exit_fix and trip_key in exit_dict:
            df.loc[(df['车辆编号'] == vehicle_id) & (df['轨迹编号'] == track_id) & (df['事件'] == '补录出站'), '发生地点'] = exit_dict[trip_key][0]
            df.loc[(df['车辆编号'] == vehicle_id) & (df['轨迹编号'] == track_id) & (df['事件'] == '补录出站'), '时间'] = exit_dict[trip_key][1]
print("fill missing station done.")

dataOutput.dump(df, 'filled_data.csv')
remained_cnt = len(df[['车辆编号', '轨迹编号']].drop_duplicates())
print(f"删除了 {deleted_trips_count} 条行程，剩余有效行程 {remained_cnt} 条，删除占比 {deleted_trips_count / (deleted_trips_count + remained_cnt) * 100:.2f}%")

```
### 统计门架忽略与正常计数的数据

```python
"""
统计门架忽略与正常计数的数据
"""
import pandas as pd

df = pd.read_csv('./doc/总_门架忽略与正常计数.csv')

# 计算“状态”为漏过的占比
total_count = len(df)
missed_count = len(df[df['状态'] == '漏过'])
missed_ratio = missed_count / total_count

# 计算“状态”为漏过时，“门架名称”中各项的占比
missed_df = df[df['状态'] == '漏过']
missed_by_gate = missed_df['门架名称'].value_counts(normalize=True)

# 计算“状态”为漏过时，“门架名称”为“门架6”且时间戳范围在1645599600 ~ 1645747200（维修期间）之间的占比
missed_gate6_df = missed_df[(missed_df['门架名称'] == '门架6') & 
                            (missed_df['时间戳'] >= 1645599600) & 
                            (missed_df['时间戳'] <= 1645747200)]
missed_gate6_ratio = len(missed_gate6_df) / missed_count

# 计算不同的“车型”，“状态”为“漏过”的占比
missed_by_vehicle_type = missed_df['车型'].value_counts(normalize=True)

# 计算各门架的漏过率，排除门架6在维修期间的漏过记录
filtered_missed_df = missed_df[~((missed_df['门架名称'] == '门架6') & 
                                 (missed_df['时间戳'] >= 1645599600) & 
                                 (missed_df['时间戳'] <= 1645747200))]
total_by_gate = df['门架名称'].value_counts()
missed_by_gate_count = filtered_missed_df['门架名称'].value_counts()
missed_rate_by_gate = (missed_by_gate_count / total_by_gate).fillna(0)


print(f"状态为漏过的占比: {missed_ratio:.2%}")
print("\n状态为漏过时，门架名称中各项的占比:")
print(missed_by_gate)
print(f"\n状态为漏过时，门架名称为门架6且时间戳范围在1645599600 ~ 1645747200之间的占比: {missed_gate6_ratio:.2%}")
print("\n不同的车型，状态为漏过的占比:")
print(missed_by_vehicle_type)
print("\n各门架的漏过率:")
print(missed_rate_by_gate)
print("\n各门架的漏过率（排除门架6在维修期间的漏过记录）:")
print(missed_rate_by_gate)
```
### 收费站与区间车流量的统计分析

```python
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

```
### 统计逐小时门架/路段车流

```python
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
```
### 统计逐小时收费站车流

```python
"""
统计逐小时收费站车流
"""
import pandas as pd
import numpy as np
from datetime import datetime

import dataLoader
import dataOutput


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
```
### 检查数据中，同一轨迹内，通过两门的时间间隔是否合理，以及通过的顺序是否合理

```python
"""
检查数据中，同一轨迹内，通过两门的时间间隔是否合理，以及通过的顺序是否合理
"""
import pandas as pd
import dataLoader
import dataOutput

source = "轨迹表5.xlsx"

df = dataLoader.read_raw_data(source)

# 筛选出包含“门架”的记录
df['门架序号'] = df['发生地点'].str.extract(r'门架(\d+)').astype(float)

# 按车辆编号和轨迹编号分组（即按照“行程”分组）
grouped = df.groupby(['车辆编号', '轨迹编号'])

# 存储异常行程
abnormal_trips = []

# 遍历每个分组，检查数据
for (vehicle_id, track_id), group in grouped:
    # 筛选出门架记录并按时间排序
    gantry_records = group.dropna(subset=['门架序号']).sort_values(by='时间')
    
    # 初始化异常类型列表
    anomalies = []
    
    # 检查门架序号是否按升序排列
    if not gantry_records['门架序号'].is_monotonic_increasing:
        anomalies.append('门架顺序异常')
    
    # 检查整个分组的时间是否按升序排列
    if not group['时间'].dropna().is_monotonic_increasing:
        anomalies.append('时间顺序异常')
    
    if anomalies:
        abnormal_trips.append((vehicle_id, track_id, ', '.join(anomalies)))

abnormal_df = pd.DataFrame(abnormal_trips, columns=['车辆编号', '轨迹编号', '异常类型'])

abnormal_counts = abnormal_df['异常类型'].value_counts()

print()
print(f"there's {len(abnormal_df)} abnormal trips in {source}, the rate is {len(abnormal_df) / len(grouped):.2%}(total {len(grouped)})")
print("abnormal type:")
print(abnormal_counts)
print()

dataOutput.dump(abnormal_df, '异常行程.csv')
```
### 用于检查车辆行程，探究数目不匹配的原因

```python
"""
用于检查车辆行程，探究数目不匹配的原因
结果：发现了存在轨迹编号缺失的情况，需要处理，进而修改了dataLoader.py中的read_raw_data函数
"""
import pandas as pd
import dataLoader
import dataOutput

df = dataLoader.read_raw_data("轨迹表1.xlsx")

# 填充时间列中的空数据
df['时间'] = df['时间'].fillna(0)

# 新建一列，将“车辆编号”和“轨迹编号”合并为一个字符串，作为识别符
df['组合编号'] = df['车辆编号'] + '_' + df['轨迹编号'].astype(str)

# 获取不同的组合编号列表
unique_combinations = df['组合编号'].unique()


unique_combinations_df = pd.DataFrame(unique_combinations, columns=['组合编号'])
dataOutput.dump(unique_combinations_df, 'unique_combinations.csv')

print("over.")
```

### Makefile用于清理目录

```makefile
PYTHON = python3.10
DATA_LOADER = ./dataLoader.py

# 默认目标
.PHONY: all
all: cache

# 清理缓存
.PHONY: clean-cache
clean-cache:
	-rm -rf ./cache/**

# 生成缓存
.PHONY: cache
cache:
	$(PYTHON) $(DATA_LOADER) cache

# 清理输出
.PHONY: clean-output
clean-output:
	-rm -rf ./output/**
```

