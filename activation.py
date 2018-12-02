# 激活函数

import numpy as np


class Sigmoid(object):
    """
    sigmoid激活函数
    """

    @staticmethod
    def forward(z):
        """
        前向传播
        :param z:
        :return:
        """
        return 1/(1+np.exp(-z))

    @staticmethod
    def backward(output):
        """
        反向传播
        :param output: 连接层的输入
        :return:
        """
<<<<<<< HEAD
        return input*(1-input)
=======
        return output*(1-output)
>>>>>>> e48c7a46d66fc0b3191e6a9fe7cd0d1cc6378952


class Relu(object):
    """
    Relu激活函数
    """
<<<<<<< HEAD
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
=======

    @staticmethod
    def forward(z):
        z[z <= 0] = 0
        return z

    @staticmethod
    def backward(output):
        output[output <= 0] = 0
>>>>>>> e48c7a46d66fc0b3191e6a9fe7cd0d1cc6378952
        output[output > 0] = 1
        return output
