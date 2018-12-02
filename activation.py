# 激活函数

import numpy as np


class Sigmoid(object):
    """
    sigmoid激活函数
    """
    def forward(self, z):
        """
        前向传播
        :param z:
        :return:
        """
        return 1/(1+np.exp(-z))

    def backward(self, input):
        """
        反向传播,
        :param input: 连接层的输入
        :return:
        """
        return input*(1-input)


class Relu(object):
    """
    Relu激活函数
    """
    def forward(self, z):
        """
        前向传播
        :param z:
        :return:
        """
        z[z <= 0] = 0
        return z

    def backward(self, output):
        """
        反向传播
        :param output:
        :return:
        """
        output[output == 0] = 0
        output[output > 0] = 1
        return output
