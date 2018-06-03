import numpy as np
from scipy import optimize, io
import time

# sigmoid函数
def sigmoid(z):
    return 1/(1+np.exp(-z))

# sigmoid函数的导数
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

# 将一维weight还原
def reduction_weight(weight, layer_size):
    # 列表，放置所有层的weight
    new_weight = []
    z = 0
    for x, y in zip(layer_size[1:], layer_size[:-1]):
        new_weight.append(weight[z:][:x*(y+1)].reshape(x, y+1))
        z += x*(y+1)
    return new_weight

# 神经网络类
class network(object):
    def __init__(self, layer_size, x, y, lamda=0):
        """
        初始化神经网络
        :param layer_size: 神经网络各层单元数的列表,例如：[401, 25, 10]代表输入层有401个单元,隐藏层
        有一层,有25个单元,输出层有10个单元.注意：输入层与输出层与你的特征值和目标向量有关,这个例子中表示：
        输入特征维度为400（因为有偏值层1，所以为401）,目标向量维度为10（代表分类数目）
        :param x: 特征值
        :param y: 目标向量,注意在我的类中,y不是一维向量,而是（shape(m, n)）,m是样本数,n是分类数目,手写数字识别
        例子中则是shape(5000, 10)
        :param lamda: 正则化参数
        """
        m, n = x.shape
        # x0 = np.ones(m)
        self.m = m
        self.n = n
        # self.x = np.insert(x, 0, values=x0, axis=1)
        self.x = x
        self.y = y
        self.lamda = lamda
        self.layer_size = layer_size
        self.layer_num = len(layer_size) # 获取神经网络的层数

        # 初始化权重,因为优化算法需要接受一维的权重,所以我将其处理为一维,后面可以通过reduction_weight函数对其还原
        weight = [np.random.randn(x, y+1).flatten() for x, y in zip(layer_size[1:], layer_size[:-1])]
        self.initial_weight = np.concatenate([w for w in weight])

    def backprop(self, x, y, weight):
        """
        反向传播算法（针对一个样本）
        :param x: 特征值（一个样本）
        :param y: 目标向量（一个样本）
        :param weight: 模型权重（[w1, w2, .....]）
        :return: 一个样本计算出来的模型权重(一个样本对weight的偏导数),最终的权重需要所有的样本进行决定
        """
        nbala_w = [np.zeros(w.shape) for w in weight] # 初始化偏导数
        active = x
        actives = [x]
        zs = []
        # 前向传播计算所有的z与a
        for w in weight:
            m = 1
            active0 = np.ones(m)
            active = np.insert(active, 0, values=active0, axis=0) # 加入偏值项１
            z = np.dot(w, active) # 计算ｚ
            zs.append(z)
            active = sigmoid(z) # 计算a
            actives.append(active)

        delta = (actives[-1] - y)*sigmoid_prime(zs[-1]) # 误差项（最后一层）
        # nbala_w[-1] = np.dot(delta.reshape(-1, 1), actives[-2].reshape(-1, 1).T)
        # 更新模型权重,偏值项不对前面的层产生影响
        nbala_w[-1][:, 0] = delta
        nbala_w[-1][:, 1:] = np.dot(delta.reshape(-1, 1), actives[-2].reshape(1, -1))

        # 依次更新所有的模型权重
        for l in range(2, self.layer_num):
            delta = np.dot(weight[-l+1][:, 1:].T, delta)*sigmoid_prime(zs[-l]) # 对应层的误差
            # 更新模型权重,偏值项不对前面的层产生影响
            nbala_w[-l][:, 0] = delta
            nbala_w[-l][:, 1:] = np.dot(delta.reshape(-1, 1), actives[-l-1].reshape(1, -1))
        return nbala_w

    def gradient(self, weight):
        """
        梯度更新（权重更新）
        :param weight: 模型权重
        :return:　更新后的梯度
        """
        # 还原weight
        weight = reduction_weight(weight, self.layer_size)
        nbala_w = [np.zeros(w.shape) for w in weight]
        #　所有样本参与对权重更新之中
        for x, y in zip(self.x, self.y):
            delta_nbala_w = self.backprop(x, y, weight)
            nbala_w = [(nw+dnw) for nw, dnw in zip(nbala_w, delta_nbala_w)]
        # 将梯度转为一维,优化算法所决定的
        nbala_w = np.concatenate([nw.flatten()/self.m for nw in nbala_w])
        return nbala_w


    def feedforward(self, weight):
        """
        前向传播
        :param weight: 模型权重
        :return: 最后一层的激活值ａ
        """
        active = self.x
        weight = reduction_weight(weight, self.layer_size)
        for w in weight:
            m = active.shape[0]
            active0 = np.ones(m)
            active = np.insert(active, 0, values=active0, axis=1)
            z = np.dot(active, w.T)
            active = sigmoid(z)
        predict_a = active.argmax(axis=1)
        y = self.y.argmax(axis=1)
        print("current rate: ", y[y==predict_a].size/y.size) # 打印预测率
        return active

    def cost(self, weight):
        """
        计算代价函数
        :param weight: 模型权重
        :return: 代价函数的值
        """
        # weight = reduction_weight(weight, self.layer_size)
        active = self.feedforward(weight) # 得到最后一层的激活值ａ
        cost = 0
        for a, y in zip(active, self.y):
            cost += np.dot(y, np.log(a)) + np.dot((1-y), np.log(1-a))
        cost = -cost/self.m
        return cost

    def fit_cg(self):
        """
        更新得到最终模型权重
        :return: 最终的模型权重（返回的是一维的权重，也可以将权重还原，最好不要这样做,因为前面几乎所有的函数都
        使用的是一维的权重,如果你想得到真正的权重，则可以在此函数外对其还原）
        """
        # 共轭梯度算法,也可以采用其他优化算法
        weight = optimize.fmin_cg(self.cost, self.initial_weight, fprime=self.gradient)
        # weight = reduction_weight(np.array(weight), self.layer_size)
        return weight





if __name__ == "__main__":
    start = time.time()
    # 加载数据
    data = io.loadmat('/home/sjl/桌面/CoCo/data/ex3data1.mat')
    x = data['X']  # 特征值
    y = data['y']  # 目标向量
    # 对y做预处理
    m, n = x.shape
    y[y == 10] = 0
    y1 = np.zeros((m, 10))
    # 对y进行预处理,得到符合network类的y
    for i in range(10):
        y1[:, i] = np.int32(y == i).reshape(1, -1)
    y = y1
    # print(y)
        
    net = network([400, 25, 10], x, y)
    weight = net.fit_cg() # 计算权重

    # 输出一部分的结果,审查算法的效果
    a = net.feedforward(weight)
    print(a.argmax(axis=1)[:100])
    print(a.argmax(axis=1)[500:600])
    print(a.argmax(axis=1)[1000:1100])

    end = time.time()
    time = end - start # 计算算法运行的时间
    print(time)
    # 直接运行该程序就可得到结果,可以观察预测率的变化