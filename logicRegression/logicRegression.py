import numpy as np
from scipy.optimize import *


def sigmoid(z):
    """
    逻辑函数
    :param z:
    :return:
    """
    return 1/(1 + np.exp(-z))


class logic(object):
    def __init__(self, x, y, initial_theta, lamda=0):
        """
        初始化输入值
        :param x: 特征值
        :param y: 目标向量
        :param initial_theta: 初始化模型参数
        :param lamda: 正则化项
        """
        self.m, self.n = x.shape
        x0 = np.ones(self.m)
        self.x = np.insert(x, 0, values=x0, axis=1)
        self.y = y
        self.initial_theta = initial_theta
        self.lamda = lamda

    def cost(self, theta):
        """
        逻辑回归的代价函数
        :param theta: 模型参数, 一定要输入一维的theta,这是优化算法所决定的
        :return: 代价函数的值
        """
        h = sigmoid(np.dot(self.x, theta)) # 假设函数
        cost = self.y*np.log(h) + (1 - self.y)*np.log(1-h)
        cost = -cost.sum()/self.m + np.sum(theta[1:]**2)*self.lamda/(2*self.m)
        return cost


    def gradient(self, theta):
        """
        逻辑回归的梯度函数
        :param theta: 模型参数, 一定要输入一维的theta,这是优化算法所决定的
        :return: 逻辑回归的梯度值
        """
        h = sigmoid(np.dot(self.x, theta))
        e = h - self.y
        grad = np.ones(self.n+1)
        grad[0] = np.dot(self.x[:, 0].T, e)
        grad[1:] = np.dot(self.x[:, 1:].T, e) + self.lamda*theta[1:]/self.m
        return grad

    def fit_cg(self):
        """
        共轭梯度优化算法(适用于线性)
        :return: 模型参数theta
        """
        theta = fmin_cg(self.cost, self.initial_theta, fprime=self.gradient)
        return theta


    def fit_ncg(self):
        """
        拟牛顿共轭梯度算法(适用于非线性)
        :return: 模型参数theta
        """
        theta = fmin_ncg(self.cost, self.initial_theta, fprime=self.gradient)
        return theta

    def fit_bfgs(self):
        """
        bfgs算法
        :return: 模型参数theta
        """
        theta = fmin_bfgs(self.cost, self.initial_theta, fprime=self.gradient)
        return theta

    def predict(self, predict_x, predict_y, theta):
        """
        逻辑回归的预测函数
        :param predict_x: 用于预测的特征值
        :param predict_y: 真实的目标向量
        :param theta: 逻辑回归模型参数
        :return: 预测的目标向量,同时打印出预测率
        """
        m, n = predict_x.shape
        x0 = np.ones(m)
        x = np.insert(predict_x, 0, values=x0, axis=1)
        h = sigmoid(np.dot(x, theta))
        # 大于等于0.5的为正类,否则为负类
        h[h>=0.5] = 1
        h[h<0.5] = 0
        print(h)
        sum = len(predict_y)
        correct = len(predict_y[h == predict_y])
        print("correct rate: %.2f%%"%(correct/sum*100))
        return h


from load_data import load_data
if __name__ == '__main__':
    x, y = load_data('/home/sjl/桌面/machine learning/mlclass-ex2-005/mlclass-ex2/ex2data1.txt')
    m, n = x.shape
    initial_theta = np.zeros(n+1)
    l = logic(x, y, initial_theta)
    theta = l.fit_cg()
    print(theta)
    l.predict(x, y, theta)













