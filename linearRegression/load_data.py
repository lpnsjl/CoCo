import numpy as np
import pandas as pd


def load_data(filename):
    df = pd.read_table(filename, sep=',', header=None) # header=None表示数据文件第一行不做列名
    x = df.iloc[:, :-1] # 特征值
    y = df.iloc[:, -1:] # 目标向量
    return np.array(x), np.array(y)

if __name__ == '__main__':
    x, y = load_data('/home/sjl/桌面/machine learning/mlclass-ex1-005/mlclass-ex1/ex1data1.txt')
    # print(x)