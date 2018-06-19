import numpy as np
from scipy import io

# pca算法
class PCA(object):
    def __init__(self, x, K):
        """
        初始化pca算法参数
        :param x: 特征值
        :param K: 保留主成分数目
        """
        self.x = x
        self.K = K
        self.m, self.n = x.shape

    def featureNormalize(self):
        """
        特征值归一化
        :return: 归一化后的特征值
        """
        normalizeX = self.x - np.mean(self.x, axis=0)
        std = np.std(self.x, axis=0)
        normalizeX = normalizeX/std
        return normalizeX

    def pca(self):
        normalizeX = self.featureNormalize()
        cov = np.cov(normalizeX.T)
        w, v = np.linalg.eig(cov)
        Z = np.dot(normalizeX, v[:, 0:self.K])
        return w, v, Z

    def recoverData(self):
        """
         还原数据
        :return:
        """
        w, v, Z = self.pca()
        recoverX = np.dot(Z, v[:, 0:self.K].T)
        return recoverX



if __name__ == "__main__":
    data = io.loadmat("/home/sjl/桌面/machine learning/mlclass-ex7-005/mlclass-ex7/ex7data1.mat")
    X = data['X']
    p = PCA(X, 1)
    p.recoverData()