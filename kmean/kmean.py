import numpy as np
from math import *
import matplotlib.pyplot as plt


class kmean(object):
    def __init__(self, K, X, maxIter=10):
        """
        初始化kmean类
        :param K: 聚类数
        :param X: 特征值
        """
        self.K = K
        self.X = X
        self.maxIter = maxIter
        self.m, self.n = X.shape


    def calDistance(self, x, u):
        """
        计算两个样本之间的距离
        :param x:
        :param u:
        :return:
        """
        return sqrt(np.sum((x - u)**2))


    def randCentroid(self):
        """
        随机初始化聚类中心
        :return:
        """
        centroid = np.zeros((self.K, self.n))
        for j in range(self.n):
            minJ = self.X[:, j].min()
            maxJ = self.X[:, j].max()
            rangeJ = maxJ - minJ
            centroid[:, j] = minJ + rangeJ*np.random.rand(self.K)
        return centroid


    def fit(self):
        """
        kmean聚类
        :return:
        """
        minCost = inf
        minCenteroid = self.randCentroid()
        minCluster = np.zeros((self.m, 2))
        for iter in range(self.maxIter):
            centroid = self.randCentroid()
            cluster = np.zeros((self.m, 2))
            centroidChange = True
            while centroidChange:
                centroidChange = False
                for i in range(self.m):
                    minDistance = inf
                    minIndex = -1
                    for k in range(self.K):
                        distance = self.calDistance(self.X[i], centroid[k])
                        if distance < minDistance:
                            minDistance = distance
                            minIndex = k
                    if cluster[i, 0] != minIndex:
                        centroidChange = True
                    cluster[i, :] = minIndex, minDistance**2
                for cent in range(self.K):
                    newCentorid = self.X[np.nonzero(cent==cluster[:, 0])[0]]
                    centroid[cent] = np.mean(newCentorid, axis=0)
            cost = np.sum(cluster[:, 1])/(2*self.m)
            if cost < minCost:
                minCenteroid = centroid
                minCluster = cluster
                minCost = cost
        return minCenteroid, minCluster





