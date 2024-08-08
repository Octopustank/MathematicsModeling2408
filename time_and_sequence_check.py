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