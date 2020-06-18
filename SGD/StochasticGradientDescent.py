import numpy as np

class StochasticGradientDescent:
    def __init__(self, learning_rate=0.01, n_iter=10000,
                 tolerance=0.001):
        self.learning_rate = learning_rate  # 学习率
        self.n_iter = n_iter  # 最大迭代次数
        self.tolerance = tolerance  # 停止迭代误差阈值

    def fit(self, X, y):
        X = np.c_[np.ones((len(X))), X]  # 添加一列1，用于线性回归中的常量
        n_samples, n_features = X.shape  # 样本量，特征量
        self.theta = np.random.randn(n_features, 1)  # 初始迭代值
        self.errors = []
        self.loss = [0]
        for self.i in range(self.n_iter):
            # 随机抽取单个样本
            index = np.random.randint(n_samples)
            xi = X[index: index+1]
            yi = y[index: index+1]
            # 单个样本的梯度
            gradient_i = xi.T.dot(xi.dot(self.theta) - yi)
            # theta-update
            self.theta -= self.learning_rate * gradient_i
            # 停机检测
            y_pre = np.dot(X, self.theta)  # 当前迭代预测值
            error = 1 / (2*n_samples) * (np.linalg.norm(y_pre-y)**2)  # 计算均方误差
            self.errors.append(error)
            if self.errors[-1] < self.tolerance:
                print("迭代次数", self.i)
                break

        # 输出学习结果
        print("均方误差", self.errors[-1])
        print("学习结果：", self.theta.flatten(), '\n')
        return self
