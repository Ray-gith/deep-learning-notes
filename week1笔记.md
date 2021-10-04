# Deep Learning Notes

## What i will learn in the course?
- Neural Network and Deep Learning; 
- Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization; 
- Structuring your Machine Learning Project; 
- Convolutional Neural Networks; 
- Natural Language Processing: Building sequence model.

## What is Neural Network
- ReLU rectified linear unit 修正线性单元
- 介绍了神经网络
  
## Supervised Learning
1.列举一些用途
| Input(x) | Output(y) | Application|Using|
|-------|--------|------------|---|
|Home features|Price|Real Estate|standard Netword|
|Ad, User Info|Click on ad?(0/1)|Online Advertising|standard|
|Image|Object(1, ......, 1000)|Photo tagging|CNN|
|Audio|Text transcript|Speech recognition|RNN|
|English|Chinese|Machine translation|RNN|
|Image, Radar info|Position of other cars|Autonomous driving|Hybird|

2.数据的区别
- Structured Data: such as a dataframe; database;
- Unstructured Data: Audio; Image; Text;
  
## 为什么神经网络现在才流行
- 以往的数据量较小，使用传统机器学习算法可以得到性能提升；
- 随着数据量增大，传统算法性能陷入瓶颈。
- 神经网络越大，性能越好，提升空间越大。
- scale driving:
  - Data
  - Computation
  - Algorithms

## Outline of the first course
- Week 1: Introduction
- Week 2: Basic of Neural Network programming
- Week 3: One hidden layer Neural Network
- Week 4: Deep Neural Network

## Binary Classification
### Notation of this course
- $(x, y)$ is a sample where $x \in \R^{n_x}$. That means x is a $n_x$ dimensional vector. And $y \in \lbrace {0, 1} \rbrace$, is a label.
- The training set has $m$ sample: 
- $$ Training Set=\lbrace (x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), ..., (x^{(m)}, y^{(m)})\rbrace $$
- $m_{train}$ and $m_{test}$ denote the number of training/testing samples respectively.
- We define a matrix to contain all $x$:
- $$X=[x^{(1)}, x^{(2)}, ..., x^{(m)}]$$
- Then, the matrix has $m$ columns and $n_x$ rows, $X \in \R^{n_x\times m}$. (stack all $x$ in columns)
```python
>> X.shape
>> (n_x, m)
```
- Likewise, all labels $y$ can be defined as follow:
- $$Y=[y^{(1)}, y^{(2)}, ..., y^{(m)}]$$
- where $Y \in \R^{1 \times m }$

### Logistic Regression
- Given $x$, and we want to know $y=P(y=1|x)$
- where $x \in \R^{n_x}$, and we have a regression which has parameters $w \in \R^{n_x}$ and $b \in \R$ ($w$ is a vector who has the same dimension, and $b$ just a real number)
- In this case, $y$ must between 0 and 1. So we have the regression:
- $$\hat y = \sigma(w^Tx+b)$$
- where $\sigma(z)=\frac{1}{1+e^{-z}}$, Sigmoid function.

![svg](source\sigmoid_funtion.svg)

### Logistic Regression Cost Function
- Given $\lbrace (x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), ..., (x^{(m)}, y^{(m)}) \rbrace$, and we want $\hat y^{(i)}\approx y^{(i)}$, we will use following loss function:
- $$\mathcal{L}(\hat y^{(i)}, y^{(i)}) = -(y\log(\hat y)+(1-y)\log(1-\hat y))$$
- If $y=1$: $\mathcal{L}(\hat y, y)=-\log(y)$, that means, we want $\log(\hat y)$ large $\rightarrow$ we want $\hat y$ large.
- vice versa.
- Then, we can define the Cost Function:
- $$\mathcal{J}(w, b)=\frac{1}{m}\sum_{i=1}^{m}\mathcal{L}(\hat y^{(i)}, y^{(i)})=-\frac{1}{m}\sum_{i=1}^{m}[y\log(\hat y)+(1-y)\log(1-\hat y)]$$

### Gradient Descent
- We want to find $w, b$ that minimize $\mathcal{J}(w, b)$
- It turns out that the function $\mathcal{J}$ is a convex function（凸函数）.
- In order to do that, we can use following method to renew our parameters:
- $$w:=w-\alpha \frac{\partial J(w)}{\partial w}$$
- where $\alpha$ is the learning rate.
- In logistic regression, we can use following fomula to renew out pamameters (note that $w$ is a vector rather than a scalar):
- $$w:=w-\alpha \frac{\partial J(w, b)}{\partial w}$$
- $$b:=b-\alpha \frac{\partial J(w, b)}{\partial b}$$

### Derivatives with a Computation Graph
- chain rule in Derivatives
- we only mind the final output varible!

### Logistic Regression Gradient descent
- 1 sample and m sample. overall gradient:
- $$\frac{\partial}{\partial w}\mathcal{J}(w, b)=\frac{1}{m}\sum_{i=1}^{m}\frac{\partial}{\partial w}\mathcal{L}(a^{(i)}, y^{(i)})$$
- Use a for loop to calculate overall gardient.
```python
import numpy as np

alpha = .0001
J, dw, db = 0, np.array([0, ..., 0]), 0
while True:
    for i in range(m):
        zi = w.dot(x) + b
        ai = 1 / (1 + np.exp(-zi))
        J -= (yi * np.log(ai) + (1 - yi) * np.log(1 - ai)) / m
        dzi = (ai - yi) / m
        dw += (xi * dzi) / m
        db += dzi / m
    w = w - alpha * dw
    b = b - alpha * db
    if np.sum([np.sum(np.square(dw)), db**2]) <= 1e-4:
        break

```
- Vectorization! Get rid of these explicit for loops!

### Vectorization
- both CPU and GPU have SIMD, which means single instruction multiple data.(单指令多数据流) 用Numpy可以直接并行运算。

### Numpy技巧
- 尽量不要用(5,)数据结构，用(5, 1) instead.
```python
try:
    a = np.random.randn(5, 1)
    assert a.shape == (5, 1)
except:
    a = a.reshape(5, 1)
```
### Logistic Regression Explaination
> If $y=1$:
> $$P(y|x)=\hat y$$
> If $y=0$:
> $$P(y|x)=1-\hat y$$
> So:
> $$P(y|x)=\hat y^y(1-\hat y)^{(1-y)}$$
> $$\mathcal{L}(\hat y, y)=\log(\hat y^y(1-\hat y)^{(1-y)})=y\log \hat y + (1-y)\log(1-\hat y)$$
> For all sample, Maximun likelyhood estimation:
> $$\mathcal{J}(w, b)=\frac{1}{m}\sum_{i=1}^{m}\mathcal{L}(\hat y^{(i)}, y^{(i)})=-\frac{1}{m}\sum_{i=1}^{m}[y\log(\hat y)+(1-y)\log(1-\hat y)]$$

