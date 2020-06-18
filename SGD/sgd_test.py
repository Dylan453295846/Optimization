import matplotlib.pyplot as plt
import numpy as np
from StochasticGradientDescent import StochasticGradientDescent

np.random.seed(3)
# 批量生成数据
# y = w0*x0 + w1*x1 + w2*x2 + ...
def GenerateData(data_num, w_num):
    x = 2*np. random.rand(data_num, w_num)
    w = 5*np.random.rand(w_num, 1)
    # 添加高斯噪声
    mu, sigma = 0, 0.1  # 均值和标准差
    noise = np.random.normal(mu, sigma, (data_num, 1))
    y = x.dot(w) + noise
    print("系数", w.flatten())
    return x, y

sgd = StochasticGradientDescent(learning_rate=0.01, n_iter=10000, tolerance=0.001)
# 一元线性回归可视化
X, y = GenerateData(100, 1)
s = sgd.fit(X, y)
x_new = np.array([[0], [2]])
x_new_b = np.c_[(np.ones((2, 1))), x_new]
y_predict = x_new_b.dot(s.theta)

fig1 = plt.figure(1)
plt.subplot(211)
plt.plot(X, y, 'y.')
plt.plot(x_new, y_predict, 'r-')
plt.xlabel("x")
plt.ylabel("y")
plt.title("simple linear regression")
plt.subplot(212)
error_X = range(s.i+1)
plt.plot(error_X, s.errors, 'b-')
plt.axis([0, 2000, 0, 0.25])
plt.xlabel("iterations")
plt.ylabel("mean square error")
plt.show()

# 二元线性回归可视化
X, y = GenerateData(100, 2)
s = sgd.fit(X, y)

x1 = [X[i][0] for i in range(len(X))]
x2 = [X[i][1] for i in range(len(X))]

ax1 = plt.axes(projection='3d')
ax1.scatter3D(x1, x2, y, cmap='Blues')
xx, yy = np.meshgrid(x1, x2)
y_predict = s.theta[0]+s.theta[1]*xx+s.theta[2]*yy
ax1.plot_surface(xx, yy, y_predict, cmap='Oranges')
plt.title("binary linear regression")
plt.show()

error_X = range(s.i+1)
plt.plot(error_X, s.errors, 'b-')
plt.axis([0, 2000, 0, 0.25])
plt.xlabel("iterations")
plt.ylabel("mean square error")
plt.show()

# 多元线性回归
X, y = GenerateData(100, 5)
s = sgd.fit(X, y)
