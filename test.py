# 神经网络模块化
import numpy as np
from activate import Sigmoid


# 全连接层
class FullConnectLayer(object):
    """
    全连接层
    """
    def __init__(self, input_size, output_size, activation):
        self.w = np.random.randn(output_size, input_size)
        self.b = np.random.randn(output_size, 1)
        self.activation = activation()
        self.output = np.zeros((output_size, 1))
        self.input = np.zeros((input_size, 1))
        self.delta = np.zeros((input_size, 1))

    def forward(self, input_array):
        """
        前向传播
        :param input_array:
        :return:
        """
        self.input = input_array
        self.z = self.w@input_array
        self.output = self.activation.forward(z=self.z)

    def backward(self, delta_array):
        """
        反向传播
        :param delta_array: 上一层的误差
        :return:
        """
        self.delta = self.activation.backward(self.z)*(self.w.T@delta_array) # 通过上一层误差计算本层误差
        # 用上一层的误差计算本层的梯度
        self.grad_w = delta_array@self.input.T
        self.grad_b = delta_array

    def update(self, rate):
        self.w = self.w - rate*self.grad_w
        self.b = self.b - rate*self.grad_b


class Net(object):
    """
    神经网络
    """
    def __init__(self, layers):
        """
        初始化函数, 定义网络结构
        :param layers:
        """
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectLayer(layers[i], layers[i+1], Sigmoid)
            )

    def forward(self, input):
        """
        前向传播
        :param input_array:
        :return:
        """
        output = input
        for layer in self.layers:
            output = layer.forward(output)
            self.output = output
        return output


    def backward(self, label):
        """
        反向传播
        :return:
        """
        delta = (self.layers[-1].output - label)*self.layers[-1].activation.backward(self.layers[-1].z)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
    def update(self, rate):
        """
        更新整个网络的权重
        :param rate:
        :return:
        """
        for layer in self.layers:
            layer.update(rate)

    def train_one_sample(self, sample, label, rate):
        """
        训练一个样本
        :param sample:
        :param label:
        :param rate:
        :return:
        """
        self.forward(sample)
        self.backward(label)
        self.update(rate)

    def train(self, labels, data_set, epoch, rate):
        """
        训练函数
        :param labels:
        :param data_set:
        :param epoch:
        :param rate:
        :return:
        """
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(data_set[d], labels[d], rate)
            # 每迭代5次打印一次精度
            if i%5 == 0:
                self.evaluate(labels, data_set)

    def evaluate(self, labels, data_set):
        """
        神经网络评估函数
        :param labels:
        :param data_set:
        :return:
        """
        size = len(data_set)
        correct = 0
        for i in range(size):
            output = self.forward(data_set[i])
            predict = np.argmax(output)
            if predict == labels[i]:
                correct += 1
        accuracy = correct/size
        print("accuracy: {:.2%}".format(accuracy))

import tensorflow as tf

def convert(data_set):
    """
    转换数据集
    :param data_set:
    :return:
    """
    new = []
    for data in data_set:
        new.append(data.flatten().reshape(-1, 1))
    return new

def load_mnist():
    """
    加载mnist数据集, 用于训练与验证
    :return:
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = convert(x_train)
    x_test = convert(x_test)
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist()
    net = Net([784, 3, 10]) # 初始化一个神经网络
    net.train(y_train, x_train, 100, 0.01) # 训练


