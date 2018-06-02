import numpy as np
from scipy import optimize, io
import time


def sigmoid(z):
    return 1/(1+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def reduction_weight(weight, layer_size):
    new_weight = []
    z = 0
    for x, y in zip(layer_size[1:], layer_size[:-1]):
        new_weight.append(weight[z:][:x*(y+1)].reshape(x, y+1))
        z += x*(y+1)
    return new_weight


class network(object):
    def __init__(self, layer_size, x, y, lamda=0):
        m, n = x.shape
        x0 = np.ones(m)
        self.m = m
        self.n = n
        # self.x = np.insert(x, 0, values=x0, axis=1)
        self.x = x
        self.y = y
        self.lamda = lamda
        self.layer_size = layer_size
        self.layer_num = len(layer_size)
        weight = [np.random.randn(x, y+1).flatten() for x, y in zip(layer_size[1:], layer_size[:-1])]
        self.initial_weight = np.concatenate([w for w in weight])

    def backprop(self, x, y, weight):
        # weight = reduction_weight(weight, self.layer_size)
        nbala_w = [np.zeros(w.shape) for w in weight]
        active = x
        actives = [x]
        zs = []
        for w in weight:
            m = 1
            active0 = np.ones(m)
            active = np.insert(active, 0, values=active0, axis=0)
            z = np.dot(w, active)
            zs.append(z)
            active = sigmoid_prime(z)
            actives.append(active)

        delta = (actives[-1] - y)*sigmoid_prime(zs[-1])
        # nbala_w[-1] = np.dot(delta.reshape(-1, 1), actives[-2].reshape(-1, 1).T)
        nbala_w[-1][:, 0] = delta
        nbala_w[-1][:, 1:] = np.dot(delta.reshape(-1, 1), actives[-2].reshape(1, -1))
        for l in range(2, self.layer_num):
            # print(weight[-l+1][:, 1:].T.shape)
            # print(sigmoid_prime(zs[-l]).shape)
            # print(delta.shape)
            delta = np.dot(weight[-l+1][:, 1:].T, delta)*sigmoid_prime(zs[-l])
            nbala_w[-l][:, 0] = delta
            nbala_w[-l][:, 1:] = np.dot(delta.reshape(-1, 1), actives[-l-1].reshape(1, -1))
        return nbala_w

    def gradient(self, weight):
        weight = reduction_weight(weight, self.layer_size)
        nbala_w = [np.zeros(w.shape) for w in weight]
        for x, y in zip(self.x, self.y):
            delta_nbala_w = self.backprop(x, y, weight)
            nbala_w = [(nw+dnw) for nw, dnw in zip(nbala_w, delta_nbala_w)]
        nbala_w = np.concatenate([nw.flatten()/self.m for nw in nbala_w])
        return nbala_w


    def feedforward(self, weight):
        active = self.x
        for w in weight:
            m = active.shape[0]
            active0 = np.ones(m)
            active = np.insert(active, 0, values=active0, axis=1)
            z = np.dot(active, w.T)
            active = sigmoid(z)

        return active

    def cost(self, weight):
        weight = reduction_weight(weight, self.layer_size)
        active = self.feedforward(weight)
        es = active - self.y
        cost = 0
        for e in es:
            cost += np.dot(e, e)
        return cost/(2*self.m)

        # cost = 0
        # for x, y in zip(self.x, self.y):
        #     active = self.feedforward(weight)
        #     e = active - y
        #     cost += np.dot(e, e)
        # return cost/(2*self.m)

    def fit_cg(self):
        weight = optimize.fmin_cg(self.cost, self.initial_weight, fprime=self.gradient)
        weight = reduction_weight(weight, self.layer_size)
        return weight



if __name__ == "__main__":
    start = time.time()
    data = io.loadmat('/home/sjl/桌面/CoCo/data/ex3data1.mat')
    x = data['X']  # 特征值
    y = data['y']  # 目标向量
    # 对y做预处理
    m, n = x.shape
    y[y == 10] = 0
    y1 = np.zeros((m, 10))
    for i in range(10):
        y1[:, i] = np.int32(y == i).reshape(1, -1)
    y = y1
    print(y)
        
    net = network([400, 25, 10], x, y)
    net.fit_cg()
    end = time.time()
    time = start - end
    print(time)