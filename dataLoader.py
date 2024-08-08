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

if __name__ == '__main__':
    if sys.argv[1] == 'cache': # 命令行调用重新生成缓存
        load_cache()
        sys.exit(0)

    result_df = read_raw_data("轨迹表1.xlsx", chache=False)
    print(result_df.head())
    # location_data = read_location_data()
    # print(location_data)