"""
卷积神经网络的实现
"""
import numpy as np


class ConvLayer(object):
    """
    卷积层
    """
    def __init__(self, in_channel, out_channel, kernel_size, pad=1, stride=1, alpha=0.01, reg=0.75):
        self.w = np.random.randn(out_channel, in_channel, kernel_size, kernel_size)
        self.b = np.random.randn(out_channel)
        self.pad = pad
        self.alpha = alpha
        self.reg = reg
        self.stride = stride


    def forward(self, in_data):
        N, F, H, W = in_data.shape
        O, _, HH, WW = self.w.shape

        in_data_padding = np.pad(in_data, ((0,), (0,), (self.pad,), (self.pad)), 'constant')
        H_OUT = 1 + (H - HH+self.pad) / self.stride
        W_OUT = 1 + (W - WW + self.pad) / self.stride

