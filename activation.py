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

        return output*(1-output)


class Relu(object):
    """
    Relu激活函数
    """
    @staticmethod
    def forward(z):
        z[z <= 0] = 0
        return z

    @staticmethod
    def backward(output):
        output[output <= 0] = 0
        output[output > 0] = 1
        return output


class TanhActivator(object):
    """
    tanh激活函数
    """
    @staticmethod
    def forward_propagation(z):
        return np.tanh(z)

    @staticmethod
    def backward_propagation(o):
        return 1 - o**2


class SoftmaxActivator(object):
    """
    softmax激活函数
    """
    @staticmethod
    def forward_propagation(z):
        return np.exp(z)/np.sum(np.exp(z))

    @staticmethod
    def backward_propagation(o):
        return
