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

