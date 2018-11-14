# 梯度检查
import numpy as np


def gradient_check(net, sample, labels):
    """
    全连接层的梯度检查
    :param weight: 全连接层计算的梯度
    :param net: 已经训练好的一个神经网络(用sample与labels训练好的神经网络)
    :return:
    """
    # 实际可以只检查最后一层梯度, 基本就能确定正确性
    epsilon = 0.0001
    m, n = net.layers[-1].weight.shape
    for i, j in zip(range(m), range(n)):
        net.layers[-1].weight -= epsilon
        predict1 = net.predict(sample)
        e1 = 0.5*(predict1 - labels).T@(predict1 - labels)
        net.layers[-1].weight += 2*epsilon
        predict2 = net.predict(sample)
        e2 = 0.5*(predict2 - labels).T@(predict2 - labels)
        expected_grad = (e2 - e1)/2*epsilon
        actual_grad = net.layers[-1].grad_w
        net.layers[-1].weight -= epsilon
        print("actual: {}, expected: {}".format(expected_grad, actual_grad))