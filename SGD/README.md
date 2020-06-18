# 随机梯度下降法求解多元线性回归问题

## 随机梯度下降法原理

问题描述：
$$
\min_wR_n(w)\\
R_n(w)=\frac{1}{n}\sum_{i=1}^{n}f(w,\xi_i),\;\{\xi_i\}_{i=1}^n
$$
SGD算法：
$$
\begin{align*}
& 初始化迭代 w\\
&for\; k=1,2,3,... do\\
&\qquad 选取随机变量\xi_k\\
&\qquad 计算迭代方向g(w_k,\xi_k)=\nabla f(w_k,\xi_k)\\
&\qquad 选取步长\alpha_k\\
&\qquad 更新迭代w_{k+1}:=w_k-\alpha_kg(w_k,\xi_k)\\
&end
\end{align*}
$$

## 多元线性回归问题描述

目标函数：
$$
h_w(x)=w_0x_0+w_1x_1+w_2x_2+\cdots+w_mx_m\\
input: \{(x_0^{(i)},x_1^{(i)},\cdots,x_m^{(i)},y^{(i)})\}_{i=1}^n;\;learn \;ouput: w_0,w_1,,\cdots,w_m
$$
写成SGD形式：
$$
均方误差：R_n(w)=\frac{1}{n}\sum_{i=1}^n(h_w(x^{(i)})-y^{(i)})^2\\
\min_wR_n(w);\;w=(w_0,w_1,,\cdots,w_m)\\
其中，h_w(x^{(i)})-y^{(i)}=\sum_{j=0}^m(w_jx_j^{(i)}-y_j^{(i)})
$$
梯度下降法迭代序列：
$$
\begin{align*}
w_j&:=w_j-\alpha \frac{\partial}{\partial w_j}R(w),\qquad j=0,\cdots,m\\
&=w_j-\alpha \frac{2}{n} \sum_{i=0}^n(h_w(x^{(i)})-y^{(i)})x_j^{(i)}\\
\end{align*}
$$
随机梯度下降法迭代序列：

​	注意到，在梯度下降法中，所有样本点均参与调整w，在样本很多的情况下，会影响收敛速度，随机梯度下降为每次更新迭代只用一个随机的样本点来调整w.
$$
w_j:=w_j-2\alpha(h_w(x^{(i)})-y^{(i)})x^{(i)}_j;\qquad j=0,\cdots,m;\quad i=randint()
$$


## 程序使用指南

```python
## Stochastic Gradient Descent类
class StochasticGradientDescent:
    def __init__(self, learning_rate=0.01, n_iter=10000,tolerance=0.001):# 设置学习率（步长），最大迭代次数，容许误差
        
     def fit(self, X, y): # 求解多元线性回归问题， X={{x_i^j}},i=1:n样本点；j=1:m特征
        X = np.c_[np.ones((len(X))), X]  # 添加一列1，用于线性回归中的常量
        n_samples, n_features = X.shape  # 样本数，特征
        self.theta = np.random.randn(n_features, 1)  # 初始迭代值
        
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
                break
              
        # 输出学习结果
        print("误差", self.errors[-1])
        print("学习结果：", self.theta.flatten(), '\n')
                
## 运行sgd_test.py测试
	# 批量生成数据，包含一元、二元线性回归可视化，和多元线性回归
```

## 程序测试

```python
## 批量创造数据
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

# 二元线性回归可视化
X, y = GenerateData(100, 2)
s = sgd.fit(X, y)

# 多元线性回归
X, y = GenerateData(100, 5)
s = sgd.fit(X, y)
```

<img src="D:\Lesson\optimization_theory\project\hw3\Figure_1.png" style="zoom:80%;" />

<center class="half">
<img src="D:\Lesson\optimization_theory\project\hw3\Figure_2.png" width = "50%" alt="***" align=left />
<img src="D:\Lesson\optimization_theory\project\hw3\Figure_3.png" width = "50%"  alt="***" align=right />
<center>





## 结论 

梯度下降法在更新每一个参数时，都需要所有的训练样本，所以训练过程会随着样本数量的加大而变得异常的缓慢，随机梯度下降法即用一个样本来近似所有的样本，来调整*w*，伴随的一个问题是噪音较多，使得SGD并不是每次迭代都向着整体最优化方向。对于最优化问题，凸问题，虽然不是每次迭代得到的损失函数都向着全局最优方向， 但是整体的方向是向全局最优解的，最终的结果往往是在全局最优解附近。