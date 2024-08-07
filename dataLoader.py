import pandas as pd
from datetime import datetime

def read_raw_data(file_path: str) -> pd.DataFrame:
    """
    读取原始数据文件，返回 DataFrame
    param
    - file_path: 文件路径
    return
    - DataFrame
    """
    df = pd.read_excel(file_path)

    # 初始化结果列表
    results = []

    # 临时变量存储当前车辆信息，用于解析数据块
    current_vehicle = None
    current_model = None
    current_track = None

    # 遍历每一行
    for index, row in df.iterrows():
        if pd.notna(row['车辆编号']): # 数据块首行，获取车辆信息
            # 处理数据块的第一行
            current_vehicle = row['车辆编号']
            current_model = row['收费站/门架编号'].split('：')[1]  # 提取车型
            continue  # 跳过数据块的第一行
        if pd.notna(row['轨迹编号']): # 刷新轨迹编号
            current_track = row['轨迹编号']
        # 读取数据行信息
        event = row['信息类型'].replace('门架信息', '门架')
        location = row['收费站/门架编号'] if row['收费站/门架编号'] != '其他' else None
        time = row['记录时间'] if row['记录时间'] != '——' else None
        if time: # 转换时间格式为时间戳
            time = datetime.strptime(time, '%d/%m/%Y %H:%M:%S').timestamp()
        
        results.append([current_vehicle, current_model, current_track, event, time, location])

    # 创建 DataFrame
    result_df = pd.DataFrame(results, columns=['车辆编号', '车型', '轨迹编号', '事件', '时间', '发生地点'])
    return result_df

if __name__ == '__main__':
    result_df = read_raw_data('./data/轨迹表1.xlsx')
    print(result_df.head())
    result_df.to_csv('./result.csv', index=False)