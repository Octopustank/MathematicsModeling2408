"""
搜索确认进出收费站，并修补或删除行程的“补录入站”和“补录出站”的“发生地点”
"""

import pandas as pd
import numpy as np
from datetime import datetime

import dataLoader
import dataOutput

dfs = []
for i in range(1, 2):
    dfs.append(dataLoader.read_raw_data(f"轨迹表{i}.xlsx"))
df = pd.concat(dfs, ignore_index=True)

location = dataLoader.read_location_data()

# 提取所有“补录入站”的记录
bu_lu_ru_zhan_records = df[df['事件'] == '补录入站']

# 速度（米/秒）
speed = 120 / 3.6

entry_results = []

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

df_gatesearch = pd.DataFrame(entry_results)
dataOutput.dump(df_gatesearch, 'enrty_search.csv')

# 确定“补录出站”的收费站
exit_results = []
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
        '可能的收费站': next_station
    })

# 将 exit_results 列表保存到 exit_search.csv 文件中
df_exitsearch = pd.DataFrame(exit_results)
dataOutput.dump(df_exitsearch, 'exit_search.csv')

# 将 entry_results 和 exit_results 转换为字典，方便查找
entry_dict = {(result['车辆编号'], result['轨迹编号']): result['可能的收费站'] for result in entry_results}
exit_dict = {(result['车辆编号'], result['轨迹编号']): result['可能的收费站'] for result in exit_results}

# 记录删除的行程数
deleted_trips_count = 0

# 遍历每一个行程，修补“发生地点”
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
            df.loc[(df['车辆编号'] == vehicle_id) & (df['轨迹编号'] == track_id) & (df['事件'] == '补录出站'), '发生地点'] = exit_dict[trip_key]

dataOutput.dump(df, 'filled_data.csv')
remained_cnt = len(df[['车辆编号', '轨迹编号']].drop_duplicates())
print(f"删除了 {deleted_trips_count} 条行程，剩余有效行程 {remained_cnt} 条，删除占比 {deleted_trips_count / (deleted_trips_count + remained_cnt) * 100:.2f}%")
