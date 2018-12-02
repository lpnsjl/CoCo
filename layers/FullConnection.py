# -*- coding:utf-8 -*-
__author__ = "andrew"
import numpy as np


class FC(object):
    """
    全连接层
    """
    def __init__(self, input_size, output_size, activation):
        """
        初始化
        :param input_size: 输入大小
        :param output_size: 输出大小
        """
        # 随机初始化连接层权重
        self.w = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        self.b = np.zeros((output_size, 1))
        self.grad_w = np.zeros(self.w.shape)
        self.grad_b = np.zeros(self.b.shape)
        self.activation = activation

    def forward(self, input_array):
        """
        前向传播
        :param input_array: 连接层输入
        :return:
        """
        self.input = input_array
        self.output = self.activation.forward(self.w@input_array)
        return self.output

    def backward(self, delta_array, activation):
        """
        反向传播
        :param delta_array: 该连接层直接导致的误差
        :return:
        """
        self.grad_w = delta_array@self.input.T
        self.grad_b = delta_array
        if activation != None:
            self.delta = activation.backward(self.input)*(self.w.T@delta_array) # 上一层直接导致的误差

    def update(self, rate):
        """
        更新梯度
        :param rate: 学习率
        :return:
        """
        self.w -= rate*self.grad_w
        self.b -= rate*self.grad_b