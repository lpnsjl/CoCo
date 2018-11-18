# 深度神经网络DNN
import numpy as np
from activation import Sigmoid

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
        self.w = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        self.b = np.zeros((output_size, 1))
        self.activation = activation

    def forward(self, input_array):
        """
        前向传播
        :param input_array: 连接层输入
        :return:
        """
        self.input = input_array
        self.output = self.activation.forward(input_array)


    def backward(self, delta_array):
        """
        反向传播
        :param delta_array: 该连接层直接导致的误差
        :return:
        """
        self.grad_w = delta_array@self.input.T
        self.grad_b = delta_array
        self.delta = self.activation.backward(self.input)*(self.w.T@delta_array) # 上一层直接导致的误差

    def update(self, rate):
        """
        更新梯度
        :param rate: 学习率
        :return:
        """
        self.w -= rate*self.grad_w
        self.b -= rate*self.grad_b


class NetWork(object):
    """
    神经网络类
    """
    def __init__(self, layers):
        """
        神经网络类
        :param layers:
        """
        # 构造神经网络层次结构
        self.layers = [FC(ip, op, Sigmoid()) for ip, op in zip(layers[1:], layers[:-1])]

    def forward(self, input_array):
        """
        前向传播
        :param input_array: 输入数组
        :return:
        """
        output = input_array
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, label):
        """
        反向传播
        :param label: 真实标签
        :return:
        """
        delta = (self.layers[-1].output - label)*self.layers[-1].backward(self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta

    def update(self, rate):
        """
        梯度更新
        :param rate: 学习率
        :return:
        """
        for layer in self.layers:
            layer.update(rate)

    def train_one_sample(self, sample, label, rate):
        """
        训练一个样本
        :param sample: 样本值
        :param label: 样本标签
        :param rate: 学习率
        :return:
        """
        self.forward(sample)
        self.backward(label)
        self.update(rate)

    def train(self, data_set, labels, rate, epoch):
        """
        神经网络训练
        :param samples: 训练数据集
        :param labels: 训练样本标签
        :param rate: 学习率
        :param epoch: 迭代次数
        :return:
        """
        size = len(data_set) # 样本个数
        for i in range(epoch):
            for d in range(size):
                self.train_one_sample(data_set[d], labels[d], rate)
            # 每五次epoch打印一次精度
            if epoch%5 == 0:
                self.evaluate(data_set, labels)

    def predict(self, pre_data_set):
        """
        神经网络预测
        :param pre_data_set: 预测数据集
        :return: 预测标签
        """
        pre_labels = []
        size = len(pre_data_set) # 预测样本个数
        for i in range(size):
            output = self.forward(pre_data_set[i])
            pre_label = np.argmax(output)
            pre_labels.append(pre_label)
        return np.array(pre_labels)

    def evaluate(self, data_set, labels):
        """
        精度评估函数
        :param data_set: 测试数据集
        :param labels: 真实样本标签
        :return:
        """
        pre_labels = self.predict(data_set)
        size = len(pre_labels)
        correct_num = 0
        for i in range(size):
            if pre_labels[i] == labels[i]:
                correct_num += 1

        accuracy = correct_num/size
        print("accuracy: {:.2%}".format(accuracy))

# 加载mnist数据集
import tensorflow as tf
def convert(data):
    """
    转换数据集
    :param data:
    :return:
    """
    new = []
    for row in data:
        new.append(row.reshape(-1, 1))
    return new

def load_data():
    mnist = tf.keras.datasets.mnist
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_data, test_data = convert(train_data), convert(test_data)
    return (train_data, train_label), (test_data, test_label)

if __name__ == "__main__":
    (train_data, train_label), (test_data, test_label) = load_data()
    net = NetWork([784, 3, 10])
    net.train(train_data, train_label, 0.1, 100)









