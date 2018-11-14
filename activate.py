# 激活函数
import numpy as np


class Sigmoid(object):
    """
    sigmoid 激活函数
    """
    def forward(self, z):
        """
        前向传播
        :return:
        """
        return 1/(1+np.exp(-z))

    def backward(self, z):
        """
        反向传播
        :param z:
        :return:
        """
        a = 1/(1+np.exp(-z))
        return a*(1-a)