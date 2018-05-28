import numpy as np
from scipy.optimize import *


class linear(object):
    """
    线性回归的类实现
    """
    def __init__(self, x, y, initial_theta, lamda=0):
        """
        初始化输入值
        :param x: 特征值
        :param y: 目标向量
        :param initial_theta: 初始化模型参数
        :param lamda: 正则项
        """
        self.m, self.n = x.shape
        x0 = np.ones(self.m)
        self.x = np.insert(x, 0, values=x0, axis=1) # 加入x0
        self.y = y
        self.lamda = lamda
        self.initial_theta = initial_theta


    def cost(self, theta):
        """
        代价函数
        :param theta: 模型参数,必须以一维矩阵的形式传入,否则优化算法会报错
        :return: 代价函数的值cost
        """
        theta = theta.reshape(-1, 1) # 将theta转变为向量
        h = np.dot(self.x, theta) # 假设函数
        e = h - self.y
        cost = np.dot(e.T, e)/(2*self.m) # 是一个一行一列的矩阵,但是没必要取cost[0][0]
        cost = cost[0][0] + np.sum(theta)*self.lamda/(2*self.m)
        return cost


    def gradient(self, theta):
        """
        梯度函数
        :param theta: 模型参数
        :return: 梯度值
        """
        theta = theta.reshape(-1, 1) # 将theta转变为向量
        h = np.dot(self.x, theta) # 假设函数
        e = h - self.y
        grad = np.ones((self.n+1)) # 初始化梯度值
        # 加入正则化项,对x0不做正则化处理
        grad[:1] = np.dot(self.x[:, :1].T, e)/self.m
        grad[1:] = np.dot(self.x[:, 1:].T, e)/self.m + theta[1:]*self.lamda/self.m
        return grad.flatten() # 必须将梯度转换成一维矩阵,否则优化算法会报错

    def fit_cg(self):
        """
        共轭梯度算法计算模型参数theta
        :return: 模型参数theta
        """
        theta = fmin_cg(self.cost, self.initial_theta, fprime=self.gradient)
        return theta.reshape(-1, 1)

    def fit_ncg(self):
        """
        拟牛顿共轭梯度算法计算模型参数theta
        :return: 模型参数theta
        """
        theta = fmin_ncg(self.cost, self.initial_theta, fprime=self.gradient)
        return theta.reshape(-1, 1)

    def fit_bfgs(self):
        """
        BFGS算法计算模型参数theta
        :return: 模型参数theta
        """
        theta = fmin_bfgs(self.cost, self.initial_theta, fprime=self.gradient)
        return theta.reshape(-1, 1)

    def predict(self, x, theta):
        """
        预测函数
        :param x: 特征值
        :return: 预测的目标向量
        """
        m, n = x.shape
        x0 = np.ones(m)
        x = np.insert(x, 0, values=x0, axis=1)
        predict = np.dot(x, theta)
        return predict


#################################################################################################
## 测试线性回归算法#################################################################################
from load_data import load_data
if __name__ == '__main__':
    x, y = load_data('/home/sjl/桌面/machine learning/mlclass-ex1-005/mlclass-ex1/ex1data1.txt')
    m, n = x.shape
    initial_theta = np.zeros(n+1)
    l = linear(x, y, initial_theta)
    theta = l.fit_bfgs()
    cost = l.cost(theta)
    print("theta: ", theta)
    print("cost: ", cost)
##################################################################################################