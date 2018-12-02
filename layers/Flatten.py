# -*- coding:utf-8 -*-
__author__ = "andrew"
import numpy as np


class Flatten(object):
    """
    Flatten层, 不需要更新参数, 只需传递误差
    """
    def forward(self, input_array):
        """
        前向传播
        :param input_array:
        :return:
        """
        self.depth, self.height, self.width = input_array.shape
        self.input = input_array
        self.output = input_array.flatten()

    def backward(self, delta_array):
        """
        反向传播
        :param delta_array:
        :return:
        """
        self.delta_array = delta_array.reshape((self.depth, self.height, self.width))

    def update(self, rate):
        """
        更新权重
        :param rate:
        :return:
        """
        pass
