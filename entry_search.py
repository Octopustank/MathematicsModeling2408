import pandas as pd
import numpy as np
from datetime import datetime

import dataLoader
import dataOutput

# 读取数据
dfs = []
for i in range(1, 2):
    dfs.append(dataLoader.read_raw_data(f"轨迹表{i}.xlsx"))
df = pd.concat(dfs, ignore_index=True)
location = dataLoader.read_location_data()

# 提取所有“补录入站”的记录
bu_lu_ru_zhan_records = df[df['事件'] == '补录入站']

# 速度（米/秒）
speed = 120 / 3.6

# 结果存储
results = []

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
    
    results.append({
        '车辆编号': vehicle_id,
        '轨迹编号': track_id,
        '可能的收费站': possible_stations
    })


processed_results = []
for result in results:
    if result['可能的收费站']:
        last_station = result['可能的收费站'][-1] # 提取每个结果中可能的收费站的倒数第一项
        processed_results.append({
            '车辆编号': result['车辆编号'],
            '轨迹编号': result['轨迹编号'],
            '可能的收费站': last_station
        })

# 将处理后的结果写入 CSV 文件
df = pd.DataFrame(processed_results)
dataOutput.dump(df, 'processed_results.csv')
