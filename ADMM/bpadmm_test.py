import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import sparse
from numpy import *
import time
from BasisPursuit import BasisPursuit

m = 20
n = 40  # n>m
np.random.seed(1)
A = np.mat(np.random.randn(m, n))  # 生成随机矩阵
# 稀疏向量生成约束Ax=b右端项
num_ele = n//2  # 稀疏度
x = np.mat(np.zeros((n, 1)))
for i in range(num_ele):
    pos = np.random.randint(0, n)
    x[pos][0] = np.random.randn()
# x =sparse.rand(n, 1, density=0.5, format='coo', dtype='float') ##无负值
b = A * x
xtrue = x

tic = time.time()
bp = BasisPursuit()
bp.admm_solver(A, b)
toc = time.time()
print('耗时', toc-tic)

K = len(bp.objval)
arr_it = np.array([k for k in range(1, K+1, 1)])

if np.linalg.norm(A * bp.x - b) < 1e-8:
    print('容许误差范围内的迭代次数为', K)
    print('满足约束条件的优化目标函数值为', bp.objval[K-1])


fig1 = plt.figure(1)
x_true = np.linalg.norm(xtrue, 1)
plt.plot([0], x_true, 'r.')
plt.plot(arr_it, bp.objval, 'b')
plt.xlabel("iterations")
plt.ylabel("objective value")
plt.show()

fig2 = plt.figure(2)
plt.subplot(211)
plt.plot(arr_it, bp.r_norm, 'b', label='residual')
plt.plot(arr_it, bp.eps_pri, 'b--', label='tolerance')
plt.ylabel("pri-problem residual")
plt.legend()

plt.subplot(212)
plt.plot(arr_it, bp.s_norm, 'r', label='residual')
plt.plot(arr_it, bp.eps_dual, 'r--', label='tolerance')
plt.xlabel("iterations")
plt.ylabel("dual-problem residual")
plt.legend()
plt.show()
