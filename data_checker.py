"""
用于检查车辆行程，探究数目不匹配的原因
结果：发现了存在轨迹编号缺失的情况，需要处理，进而修改了dataLoader.py中的read_raw_data函数
"""
import pandas as pd
import dataLoader

# 读取数据
df = dataLoader.read_raw_data("轨迹表1.xlsx")

# 填充时间列中的空数据，可以使用一个非常早的时间戳，例如 0
df['时间'] = df['时间'].fillna(0)

# 新建一列，将“车辆编号”和“轨迹编号”合并为一个字符串
df['组合编号'] = df['车辆编号'] + '_' + df['轨迹编号'].astype(str)

# 获取不同的组合编号列表
unique_combinations = df['组合编号'].unique()

# 将不同的组合编号输出到CSV文件
unique_combinations_df = pd.DataFrame(unique_combinations, columns=['组合编号'])
unique_combinations_df.to_csv('unique_combinations.csv', index=False)

print("over.")