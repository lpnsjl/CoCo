import numpy as np
import pandas as pd


def load_data(filename):
    df = pd.read_table(filename, sep=',', header=None) # 从文件读取数据
    # print(df)
    x = np.array(df.iloc[:, :-1]) # 特征值
    y = np.array(df.iloc[:, -1]) # 目标向量
    return x, y


if __name__ == '__main__':
    x, y = load_data('/home/sjl/桌面/machine learning/mlclass-ex2-005/mlclass-ex2/ex2data1.txt')
    print("x: ", x)
    print("y: ", y)