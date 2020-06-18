# ADMM算法求解Basis pursuit 问题

SA19001053 武亚琼

## ADMM算法原理

问题形式：
$$
\min_{x,z} f(x)+g(z)\\
s.t. Ax+Bz=c
$$
其增广Lagrange函数为：
$$
L_\rho(x,z,\lambda)=f(x)+g(z)+\lambda^T(Ax+Bz-c)+\frac{\rho}{2}||Ax+Bz-c||_2^2
$$
更新：
$$
x^{k+1}:=\arg \min_x L_\rho(x,z^k,\lambda^k)\\
z^{k+1}:=\arg \min_z L_\rho(x^{k+1},z,\lambda^k)\\
\lambda^{k+1}:= \lambda_k+\rho(Ax^{k+1}+Bz^{k+1}-c)
$$

## BP问题描述

优化目标：
$$
\min_x ||x||_1\\
s.t. Ax=b
$$
写成ADMM形式：
$$
\min_{x,z} f(x)+||z||_1\\
s.t. x=z\\
Ax=b
$$
更新 x：
$$
\begin{align*}
x^{k+1}&=\arg \min_{Ax=b}\{ \frac{\rho}{2}||x-(z^k-u^k)||_2^2\},\; u^k=\frac{\lambda^k}{\rho}\\
&=P(x)=P(z^k-u^k)\\
&= x-A^T(AA^T)^{-1}(Ax-b)\\
&=(I-A^T(AA^T)^{-1}A)x+A^T(AA^T)^{-1}b\\
&=(I-A^T(AA^T)^{-1}A)(z^k-u^k)+A^T(AA^T)^{-1}b\\
 
\end{align*}
$$
更新 z (可用Soft-thresholding shrinkage方法):
$$
\begin{align*}
z^{k+1}&=\arg \min_{z\in \mathbb{R}^n} ||z||_1+(\lambda^k)^T(x^{k+1}-z)+ \frac{\rho}{2}||x^{k+1}-z||_2^2\\
&=\arg \min_{z\in \mathbb{R}^n} ||z||_1+\frac{\rho}{2}||z-(x^{k+1}+u^k)||_2^2\\
&=Shrinkage(z^k,\frac{1}{\rho})=Shrinkage(x^{k+1}+u^k,\frac{1}{\rho})
\end{align*}
$$
其中：
$$
Shrinkage(z,\tau)=\left \{
\begin{align*}
& z-\tau, &z>\tau\\
& 0,&|z|<\tau\\
&z+\tau, &z<-\tau
\end{align*}
\right.
$$


更新乘子 lambda：
$$
乘子更新\lambda:\lambda^{k+1}=\lambda^k+\rho(x^{k+1}-z^{k+1})\\
等价于更新 u: u^{k+1}=u^k+x^{k+1}-z^{k+1}
$$


## 程序使用指南

```python
## 带ADMM求解器的BasisPursuit类
class BasisPursuit:
    def __init__(self, n_iter=1000, abstol=1e-4, reltol = 1e-2):# 设置最大迭代次数，绝对误差，相对误差
        
     def admm_solver(self, A, b, rho=1.0, alpha=1.0): # 设置增广Lagrange乘子rho，松弛因子alpha
        if self.r_norm[-1] < self.eps_pri[-1] and self.s_norm[-1] < self.eps_dual[-1]: # 停机准则设为原始问题残差和对偶问题残差小于容许误差
            
## 运行bpadmm_test.py测试
```

## 程序测试

```python
# 设置测试参数
m = 20
n = 40  
np.random.seed(1)
A = np.mat(np.random.randn(m, n))
x =sparse.rand(n, 1, density=0.5, format='coo', dtype='float')
b = A * x

bp = BasisPursuit()
bp.admm_solver(A, b)
```

<img src="D:\Lesson\optimization_theory\project\ADMM\figure_3.png" style="zoom:80%;" />

<img src="D:\Lesson\optimization_theory\project\ADMM\Figure_1.png" alt="5" style="zoom:80%;" />

<img src="D:\Lesson\optimization_theory\project\ADMM\Figure_2.png" alt="Figure_2" style="zoom:80%;" />

## 结论 

基追踪法是信号处理的一种重要方法，目的是想找到一组稀疏解恢复信号，即找到线性系统的一个稀疏解。

对于凸函数的优化问题，对偶上升法核心思想是引入一个对偶变量，然后利用交替优化的思路，使得两者同时达到最优。增广Lagrange方法，放松了对于f(x)严格凸的假设和其他一些条件，同时还能使得算法更加稳健，但破坏了对偶上升法利用分解参数来并行的优势。ADMM算法把原始变量和目标函数拆分，将拆开的变量分别看做是不同的变量x和z，同时处理约束条件，从而后面不需要一起融合x和z，保证了前面优化过程的可分解性。

