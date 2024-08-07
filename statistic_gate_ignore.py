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