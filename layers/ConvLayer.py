"""
卷积层
"""
import numpy as np


class Kernel(object):
    """
    卷积核类, 存储权重与偏差
    """
    def __init__(self, depth, height, width):
        self.weight = np.random.uniform(-1e-4, 1e-4, (depth, height, width))
        self.bias = 0
        self.grad_w = np.zeros((depth, height, width))
        self.grad_b = 0

    def update(self, learn_rate):
        """
        更新卷积核参数
        :param learn_rate:
        :return:
        """
        self.weight -= learn_rate*self.grad_w
        self.bias -= learn_rate*self.grad_b


def calculte_output_size(input_size, kernel_size, zp, stride):
    return (input_size + 2*zp - kernel_size)/stride


def pad_array(input_array, zp):
    """
    零填充数组, 自适应2D与3D
    :param input_array:
    :param zp:
    :return:
    """
    ndim = input_array.ndim
    height = input_array.shape[-2]
    width = input_array.shape[-1]
    if ndim == 3:
        depth = input_array.shape[0]
        output_array = np.zeros((depth, height+2*zp, width+2*zp))
        output_array[:, zp:zp+height, zp:zp+width] = input_array
        return output_array
    if ndim == 2:
        output_array = np.zeros(height+2*zp, width+2*zp)
        output_array[zp:zp+height, zp:zp+width] = input_array
        return output_array


def get_patch(input_array, i, j, kernel_height, kernel_width, stride):
    """
    得到输入数组的卷积区域
    :param input_array:
    :param i:
    :param j:
    :param kernel_height:
    :param kernel_width:
    :param stride:
    :return:
    """
    ndim = input_array.ndim
    if ndim == 3:
        output_array = input_array[:, i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width]
        return output_array
    if ndim == 2:
        output_array = input_array[i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width]
        return output_array


def conv(input_array, kernel_array, output_array, stride, bias):
    """
    卷积运算, 自动区分2D与3D
    :param input_array:
    :param kernel_array:
    :param output_array:
    :param stride:
    :param bias
    :return:
    """
    output_height = output_array.shape[-2]
    output_width = output_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    kernel_width = kernel_array.shape[-1]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i, j] = (get_patch(input_array, i, j, kernel_height,
                                            kernel_width, stride)*kernel_array).sum() + bias


class Convlution(object):
    """
    卷积类, 负责卷积层的运算
    """
    def __init__(self, input_height, input_width, channel_num, kernel_height,
                 kernel_width, kernel_num, activator, zp=0, stride=1):
        """
        卷基层初始化
        :param input_height: 输入高度
        :param input_width: 输入宽度
        :param channel_num: 频道数, 即输入第三维大小
        :param kernel_height: 卷积核高度
        :param kernel_width: 卷积核宽度
        :param kernel_num: 卷积核数目
        :param activator: 激活函数
        :param zp: 零填充数量
        :param stride: 步幅
        """
        self.input_height = input_height
        self.input_width = input_width
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.kernels = [Kernel(channel_num, kernel_height, kernel_width) for i in range(kernel_num)]
        self.activator = activator
        self.zp = zp
        self.stride = stride
        output_height = calculte_output_size(input_height, kernel_height, zp, stride)
        output_width = calculte_output_size(input_width, kernel_width, zp, stride)
        self.output = np.zeros((kernel_num, output_height, output_width))

    def forward(self, input_array):
        """
        前向传播
        :param input_array:
        :return:
        """
        self.input = input_array
        self.pad_input = pad_array(input_array, self.zp)
        for f in range(self.kernel_num):
            kernel = self.kernels[f]
            conv(input_array, kernel.weight, self.output[f], self.stride, kernel.bias)
        self.output = self.activator.forward(self.output)

    def expand_sensitivity_map(self, sensetivity_map):
        pass

    def backward(self, delta_array):
        """
        反向传播
        :param delta_array:
        :return:
        """
        pass










