# -*- coding:utf-8 -*-
__author__ = "andrew"
# pool层
import numpy as np
from layers.Convolution import get_patch_array


class MaxPooling(object):
    """
    最大池化层
    """
    def __init__(self, input_height, input_width, channel_num, filter_height, filter_width, stride=1):
        self.input_height = input_height
        self.input_width = input_width
        self.channel_num = channel_num
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.stride = stride
        self.output_height = (self.input_height - filter_height)//stride + 1
        self.output_width = (self.input_width - filter_width)//stride + 1
        self.output = np.zeros((channel_num, self.output_height, self.output_width))

    def forward(self, input_array):
        """
        前向传播
        :param input_array:
        :return:
        """
        self.input = input_array
        for d in range(self.channel_num):
            for i in range(self.input_height):
                for j in range(self.input_width):
                    self.output[d, i, j] = get_patch_array(input_array[d], np.zeros((self.filter_height, self.filter_width)), i, j, self.stride).max()

    def backward(self, delta_array):
        """
        反向传播
        :param delta_array:
        :return:
        """
        self.delta_array = np.zeros((self.channel_num, self.input_height, self.input_width))
        for d in range(self.input_height):
            for i in range(self.input_height):
                for j in range(self.input_width):
                    patch_array = get_patch_array(self.input[d], np.zeros((self.filter_height, self.filter_width)), i, j, self.stride)
                    ks, ls = np.where(patch_array == patch_array.max())
                    k, l = ks[0], ls[0]
                    self.delta_array[d, i*self.stride + k, j*self.stride + l] = delta_array[d, i, j]

    def update(self, rate):
        """
        更新
        :param rate:
        :return:
        """
        pass
