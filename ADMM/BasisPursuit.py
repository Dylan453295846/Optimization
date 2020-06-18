import numpy as np

class BasisPursuit:
    def __init__(self, n_iter=1000, abstol=1e-4, reltol=1e-2):
        self.n_iter = n_iter  # 最大迭代次数
        self.abstol = abstol  # 绝对误差
        self.reltol = reltol  # 相对误差

    def admm_solver(self, A, b, rho=1.0, alpha=1.0):
        m, n = A.shape
        self.x = np.mat(np.zeros((n, 1)))
        z = np.mat(np.zeros((n, 1)))
        u = np.mat(np.zeros((n, 1)))
        self.objval, self.s_norm, self.r_norm = [], [], []
        self.eps_pri, self.eps_dual = [], []

        P = np.mat(np.eye(n)) - A.T * (np.linalg.inv(A * A.T) * A)
        q = A.T * (np.linalg.inv(A * A.T) * b)

        for i in range(self.n_iter):
            # x-update
            self.x = P * (z - u) + q

            # z-update
            z_old = z
            x_hat = alpha * self.x + (1 - alpha) * z_old  # 松弛alpha in [1.0,1.8]
            z = self.Shrinkage(x_hat + u, 1 / rho)

            # 等价于Lagrange乘子lambda-update
            u += x_hat - z

            # termination checks
            self.objval.append(self.Objective(self.x))  # 目标函数值
            self.r_norm.append(np.linalg.norm(self.x - z))  # 原始问题残差
            self.s_norm.append(np.linalg.norm(-rho * (z - z_old)))  # 对偶问题残差
            self.eps_pri.append(np.sqrt(n) * self.abstol
                                + self.reltol * max(np.linalg.norm(self.x), np.linalg.norm(-z)))  # 原始问题容许误差
            self.eps_dual.append(np.sqrt(n) * self.abstol + self.reltol * np.linalg.norm(rho * u))  # 对偶问题容许误差
            if self.r_norm[-1] < self.eps_pri[-1] and self.s_norm[-1] < self.eps_dual[-1]:
                break

    @staticmethod
    def Shrinkage(z, tau):
        for i in range(z.shape[0]):
            z[i][0] = max(0, np.abs(z[i][0]) - tau) * np.sign(z[i][0])
        return z

    @staticmethod
    def Objective(x):
        return np.linalg.norm(x, 1)
