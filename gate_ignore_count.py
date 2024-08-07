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