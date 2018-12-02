# -*- coding:utf-8 -*-
__author__ = "andrew"

# 卷积层
import numpy as np
from activation import Relu


class Kernel(object):
    """
    卷积核
    """
    def __init__(self, depth, height, width):
        """
        初始化卷积核
        :param depth:
        :param height:
        :param width:
        """
        self.weight = np.random.uniform(-1e-4, 1e-4, (depth, height, width))
        self.bias = 0

        self.grad_w = np.zeros((depth, height, width))
        self.grad_b = 0

    def update(self, rate):
        """
        更新卷积核
        :param rate: 学习率
        :return:
        """
        self.weight -= rate*self.grad_w
        self.bias -= rate*self.grad_b


def get_patch_array(input_array, kernel_array, i, j, stride):
    """
    得到卷积部分, 自动区分2D与3D
    :return:
    """
    ndim = input_array.ndim
    height = kernel_array.shape[-2]
    width = kernel_array.shape[-1]
    if ndim == 2:
        patch_array = input_array[i*stride:i*stride+height, j*stride:j*stride+width]
        return patch_array
    if ndim == 3:
        patch_array = input_array[:, i * stride:i * stride + height, j * stride:j * stride + width]
        return patch_array


# 卷积运算
def conv(input_array, kernel_array, output_array, stride, kernel_bias):
    height = output_array.shape[-2]
    width = output_array.shape[-1]
    for i in range(height):
        for j in range(width):
            output_array[i, j] = (get_patch_array(input_array, kernel_array, i, j, stride)*kernel_array).sum() + kernel_bias


def pad(input_array, zp):
    ndim = input_array.ndim
    height = input_array.shape[-2]
    width = input_array.shape[-1]
    if ndim == 2:
        pad_array = np.zeros((height + 2*zp, width + 2*zp))
        pad_array[zp:zp+height, zp:zp+width] = input_array
        return pad_array
    if ndim == 3:
        depth = input_array.shape[0]
        pad_array = np.zeros((depth, height + 2*zp, width + 2*zp))
        pad_array[:, zp:zp+height, zp:zp+width] = input_array
        return pad_array


class Convolution(object):
    """
    卷积层
    """
    def __init__(self, input_height, input_width, channel_num, kernel_height, kernel_width, kernel_num,
                 activator, stride=1, zp=0):
        self.input_height = input_height
        self.input_width = input_width
        self.channel_num = channel_num
        self.activator = activator
        self.stride = stride
        self.zp = zp
        self.kernel_num = kernel_num
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.kernels = [Kernel(channel_num, kernel_height, kernel_width) for i in range(kernel_num)]
        self.output_height = (input_height + 2*zp - kernel_height)//stride + 1
        self.output_width = (input_width + 2*zp - kernel_width)//stride + 1
        self.output = np.zeros((kernel_num, self.output_height, self.output_width))

    def forward(self, input_array):
        """
        前向传播
        :param input_array:
        :return:
        """
        self.input = input_array
        self.pad_input = pad(input_array, self.zp)
        for k in range(self.kernel_num):
            kernel = self.kernels[k]
            conv(self.pad_input, kernel.weight, self.output[k], self.stride, kernel.bias)
        self.output = self.activator.forward(self.output)
        return self.output

    def expand_sensitivity_map(self, sensitivity_map):
        """
        扩展sensitivity_map
        :param sensitivity_map:
        :return:
        """
        depth = sensitivity_map.shape[0]
        height = (self.input_height + 2*self.zp - self.kernel_height) + 1
        width = (self.input_width + 2*self.zp - self.kernel_width) + 1
        expanded_array = np.zeros((depth, height, width))
        for i in range(self.output_height):
            for j in range(self.output_width):
                expanded_array[:, i*self.stride, j*self.stride] = sensitivity_map[:, i, j]
        return expanded_array

    def bp_sensitivity_map(self, sensitivity_map):
        """
        计算上一层的sensitivity_map
        :param sensitivity_map: 本层的sensitivity_map
        :return:
        """
        self.pad_input_height = self.pad_input.shape[-2]
        self.pad_input_width = self.pad_input.shape[-1]
        expand_array = self.expand_sensitivity_map(sensitivity_map)
        expand_height = expand_array.shape[-2]
        expand_width = expand_array.shape[-1]
        zp = (self.pad_input_height + self.kernel_height - 1 - expand_height)//2
        sum_delta_array = np.zeros((self.channel_num, self.pad_input_height, self.pad_input_width))
        padded_array = pad(expand_array, zp)
        for k in range(self.kernel_num):
            kernel = self.kernels[k]
            flipped_weight = np.map(lambda i: np.rot90(i, 2), kernel.weight)  # 旋转180
            delta_array = np.zeros((self.channel_num, self.pad_input_height, self.pad_input_width))
            for d in range(delta_array.shape[0]):
                conv(padded_array[k], flipped_weight[d], delta_array[d], 1, 0)
            sum_delta_array += delta_array
        sum_delta_array *= self.activator.backward(self.pad_input)
        self.delta_array = sum_delta_array[:, self.zp: self.zp+self.input_height, self.zp: self.zp+self.input_width]
        return self.delta_array

    def backward(self, sensitivity_map):
        """
        反向传播
        :param sensitivity_map:
        :return:
        """
        self.bp_sensitivity_map(sensitivity_map)  # 计算上一层的sensitivity_map
        expand_array = self.expand_sensitivity_map(sensitivity_map)
        for k in range(self.kernel_num):
            kernel = self.kernels[k]
            for d in range(kernel.weight.shape[0]):
                if self.pad_input.ndim == 2:
                    conv(self.pad_input, expand_array[k], kernel.grad_w[d], 1, 0)
                if self.pad_input.ndim == 3:
                    conv(self.pad_input[d], expand_array[k], kernel.grad_w[d], 1, 0)
            kernel.grad_b = expand_array[k].sum()

    def update(self, rate):
        """
        更新卷积层参数
        :param rate:
        :return:
        """
        for kernel in self.kernels:
            kernel.update(rate)
