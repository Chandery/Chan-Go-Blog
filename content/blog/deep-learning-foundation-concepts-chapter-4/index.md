+++
# Content Identity
title = "Deep Learning: Foundation & Concepts - Chapter 4"
description = "单层网络：回归 (Regression) - 从线性回归框架讨论神经网络的基本思想，涵盖基函数、似然函数、极大似然、最小二乘法、决策理论和偏差-方差权衡等核心概念。"
summary = "本章使用线性回归框架讨论了神经网络背后的一些基本思想。线性回归模型对应于具有单层可学习参数的简单形式的神经网络。尽管单层网络的实际应用非常有限，但它们具有简单的分析性质，并为引入许多核心概念提供了一个很好的框架，这些概念将为深度神经网络奠定基础。"
# Authoring
author = "Chandery"
date = "2025-10-22T17:03:04+08:00"
lastmod = "2025-10-22T17:03:04+08:00"
license = "CC-BY"

# Organization
categories = ["Deep Learning"]
tags = ["技术相关", "深度学习", "机器学习", "线性回归", "神经网络"]
## Series
# series = "Deep Learning: Foundation & Concepts"
# parts = "Chapter 4"
# weight = 4

# Display
featured = true
recommended = true
thumbnail = "https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/image-20251022171907248.png"

# Publication Control
draft = false
layout = "page"

# Advanced SEO
seo_type = "BlogPosting"
seo_image = "https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/image-20251022171907248.png"
twitter_username = ""
+++

# 单层网络:回归 (Regression)

**Summary by Chandery** | Chapter 4 in Deep Learning: Foundation and Concepts | 2025年10月

---

在本章中，我们使用**线性回归**框架讨论了神经网络背后的一些基本思想，这能简要地帮助我们了解多项式曲线拟合。我们将看到，线性回归模型对应于具有单层可学习参数的简单形式的神经网络。尽管单层网络的实际应用非常有限，但它们具有简单的分析性质，并为引入许多核心概念提供了一个很好的框架，这些概念将为我们在后面的章节中讨论深度神经网络奠定基础。

## 线形回归

回归的目的是在给定$D$维向量$x$作为输入变量的条件下**预测**连续的目标向量$t$。一般来说我们有训练集包括$N$个观察值(\{$x_n$\},\{$t_n$\})，目标是对于一个新值$x$预测它的目标值$t$。我们建立一个函数$y(x,\omega)$来表示这种转换，其中$\omega$表示从训练集中训练而得的参数向量。

最简单的回归模型的形式被表示为对输入变量的线性组合:

$$y(\textbf{x},\textbf{w})=\omega_0+\omega_1x_1+...+\omega_Dx_D \quad (1)$$

其中$x=(x_1,x_2,...,x_D)^T$。

线性回归这一术语有时特指这种形式的模型。该模型的关键特性是它是参数$\omega_0$,...,$\omega_D$的线性函数。然而，它也是输入变量$x_i$的线性函数，这对模型造成了很大的限制。

> **注意**: 这里是因为由于对于输入变量是线性的，模型不管怎么叠加，都存在一种一层的线性变换与之等价。模型无法表达非线性的分布。

### 基函数

我们可以把先前简单的模型形式扩展为使用非线性函数对输入变量$x$进行修正的形式:

$$y(\textbf{x},\textbf{w})=\omega_0+\sum_{j=1}^{M-1}\omega_j\phi_j(\textbf{x}) \quad (2)$$

$\phi_j(\textbf{x})$被称为**基函数**。j的最大值时M-1，所以模型的最大值是M。

$\omega_0$为模型提供了偏移矫正，通常被称为偏移量(bias)。这里我们可以定义$\phi_0(\textbf{x})=1$使得形式可以化简为：

$$y(\textbf{x},\textbf{w})=\sum_{j=0}^{M-1}\omega_j\phi_j(\textbf{x})=\textbf{w}^T\boldsymbol{\phi}(\textbf{x}) \quad (3)$$

其中$\textbf{w}=(\omega_0,...,\omega_{M-1})^T$，$\boldsymbol{\phi}=(\phi_0,...,\phi_{M-1})^T$，图4.1展示了使用神经网络的形式表示这个等式：

![4.1](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/4.1.png)

*这里，每个基函数$\phi_j{\textbf{x}}$都由一个输入节点表示，实心节点表示"偏置"基函数$\phi_0$，函数$y(\textbf{x},\textbf{w})$由一个输出节点表示。每个参数$\omega_j$由连接相应基函数和输出的线表示。*

使用了基于非线性的函数后，$f(\textbf{x},\textbf{w})$对于输入向量$x$来说变成了一个非线性的函数。然而，当前的表达形式被称为**线性模型**因为它对于**w**是线性的。这种对于参数的线性性极大的简化了对于此类模型的分析。但是，它依然拥有一些显著的限制。

在深度学习出现之前，机器学习的时间中通常对输入变量**x**进行一些形式的预处理——通常被称为特征提取(feature extraction)——用一组基函数表示\{$\phi_j(\textbf{x})$\}。我们希望找到一个高效强力的基函数集使得训练任务可以用一个很简单的模型表示。不幸的是，出了最简单的应用外，我们很难构造一个适合所有应用的基函数。深度学习使用训练数据集本身的方式训练处所需的非线性变换来解决这个问题。

在第一章中简要提到的线性模型的形式为：

$$y(x,\textbf{w})=\omega_0+\omega_1x+\omega_2x^2+...+\omega_Mx^M=\sum_{j=0}^{M}\omega_jx^j \quad (4)$$

这里的$\phi_j(x)=x^j$。当然，基函数还有很多表达方式，例如

$$\phi_j(x)=e^{-\frac{(x-\mu_j)^2}{2s^2}} \quad (5)$$

这里的$\mu_j$控制了基函数在输入空间中的位置，参数$s$控制了空间范围。这些通常被称为"高斯"基函数，但应该注意的是，它们不需要有概率意义。特别是归一化系数不重要，因为这些基函数将乘以可学习参数$\omega_j$。

> **注意**: 归一化系数指的是高斯分布前面的$\frac{1}{\sqrt{2\pi}s}$

或者用sigmoid(> **注意**: S型的)基函数

$$\phi_j(\textbf{x})=\sigma(\frac{x-\mu_j}{s}) \quad (6)$$

$\sigma(a)$是逻辑sigmoid函数，其表达式为

$$\sigma(a)=\frac{1}{1+exp(-a)} \quad (7)$$

同样我们也可以用tanh函数因为它的表达式$\tanh(a)=2\sigma(2a)-1$与$\sigma$有关，因此，$\sigma$函数的一般线性组合等价于tanh函数的一般直线组合，因为它们可以表示同一类输入-输出函数。图4.2展示了不同选择基函数的图像。

![4_2](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/4_2.png)

*左边展示基于多项式的基函数；中间展示高斯基函数；右边展示Sigmoidal(S型)基函数*

### 似然函数

我们使用最小化平方损失函数的方法来你和多项式函数，在一种高斯噪声模型的假定下，这种误差函数可以作为最大似然解。

> **推导说明**:
>
> 这里对这句话进行一个推导：
>
> 我们考察一个假设：对于观测值 $y$，我们认为它是某个真实函数的输出加上高斯噪声。假设有一个模型预测值 $\hat{y} = f(x; \theta)$，其中 $f$ 是我们要拟合的函数，$\theta$ 是模型参数。
>
> **模型设定**：
> 我们假设观测值可以表示为：
> $$y = \hat{y} + \epsilon = f(x; \theta) + \epsilon,$$
> 其中 $\epsilon$ 是一个高斯噪声，满足 $\epsilon \sim \mathcal{N}(0, \sigma^2)$。
>
> **似然函数**：
> 因此，在给定参数 $\theta$ 的条件下，观测值 $y$ 的概率密度可以表达为：
> $$p(y | \hat{y}) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{(y - \hat{y})^2}{2\sigma^2}\right).$$
>
> **对数似然函数**：
> 对上述似然函数取对数，得到对数似然函数：
> $$\log p(y | \hat{y}) = -\frac{1}{2}\log(2\pi \sigma^2) - \frac{(y - \hat{y})^2}{2\sigma^2}.$$
>
> **最大化对数似然**：
> 为了执行最大似然估计，我们需要最大化对数似然函数。由于 $-\frac{1}{2}\log(2\pi \sigma^2)$ 是一个常数，不依赖于 $\hat{y}$，因此最大化对数似然函数相当于最大化项：
> $$-\frac{(y - \hat{y})^2}{2\sigma^2}.$$
>
> 最大化这个表达式意味着最小化平方项，即：
> $$\min_{\theta} (y - \hat{y})^2.$$
>
> 因此，从这里我们得出最小化平方损失函数与最大化似然函数是等价的。

同样的，我们假设目标变量$t$来自于确定的函数$y(\textbf{x},\textbf{w})$加上高斯噪声

$$t=y(\textbf{x},\textbf{w})+\epsilon \quad (8)$$

这里$\epsilon$是一个均值为0方差为$\sigma^2$的高斯噪声。因此我们可以写

$$p(t|\textbf{x},\textbf{w},\sigma^2)=\mathcal{N}(t|y(\textbf{x},\textbf{w}),\sigma^2) \quad (9)$$

然后我们考虑输入的集合$X=\{x_1,...,x_N\}$以及与之对应的目标值$t_1,...,t_N$，我们把这个集合写成$\textbf{t}$。假设这些数据从式(9)中独立抽取的，我们可以得到一个似然函数的表达（这里我们用到式（3））

$$p(\textbf{t}|X,\textbf{w},\sigma^2)=\prod_{n=1}^{N}\mathcal{N}(t_n|\textbf{w}^T\phi(\textbf{x}_n),\sigma^2) \quad (10)$$

对这个似然函数取对数

$$\ln p(\textbf{t}|X, \textbf{w},\sigma^2) = \sum_{n=1}^{N}\ln\mathcal{N}(t_n|\textbf{w}^T\phi(\textbf{x}_n),\sigma^2)=-\frac{N}{2}\ln\sigma^2-\frac{N}{2}\ln(2\pi)-\frac{1}{2\sigma^2}\sum_{n=1}^{N}\{t_n-\textbf{w}^T\boldsymbol{\phi}(\textbf{x}_n)\}^2 \quad (11)$$

### 极大似然

写完似然函数之后我们使用极大似然来得到**w**和$\sigma^2$。首先考虑**w**。式(11)对**w**求导得

$$\nabla_\textbf{w}\ln p(\textbf{t}|X,\textbf{w},\sigma^2)=\frac{1}{\sigma^2}\sum_{n=1}^{N}\{t_n-\textbf{w}^T\boldsymbol{\phi}(\textbf{x}_n)\}\boldsymbol{\phi}(\textbf{x}_n)^T \quad (12)$$

令该梯度为零得

$$0=\sum_{n=1}^{N}t_n\boldsymbol{\phi}(\textbf{x}_n)^T-\textbf{w}^T\left(\sum_{n=1}^{N}\boldsymbol{\phi}(\textbf{x}_n)\boldsymbol{\phi}(\textbf{x}_n)^T\right) \quad (13)$$

化简得到**w**的值为

$$\textbf{w}_{ML}=(\Phi^T\Phi)^{-1}\Phi^T\textbf{t} \quad (14)$$

这个被称为最小二乘法问题的**正规方程**。这里的$\Phi$是一个$N\times M$的矩阵，被称为设计矩阵，其中元素$\Phi_{nj}=\phi_j(\textbf{x}_n)$,因此

$$\boldsymbol{\Phi}=\left(\begin{array}{cccc}\phi_0(\mathbf{x}_1) & \phi_1(\mathbf{x}_1) & \cdots & \phi_{M-1}(\mathbf{x}_1) \\\phi_0(\mathbf{x}_2) & \phi_1(\mathbf{x}_2) & \cdots & \phi_{M-1}(\mathbf{x}_2) \\\vdots & \vdots & \ddots & \vdots \\\phi_0(\mathbf{x}_N) & \phi_1(\mathbf{x}_N) & \cdots & \phi_{M-1}(\mathbf{x}_N)\end{array}\right) \quad (15)$$

式子

$$\boldsymbol{\Phi}^{\dagger}\equiv\left(\boldsymbol{\Phi}^{\mathrm{T}}\boldsymbol{\Phi}\right)^{-1}\boldsymbol{\Phi}^{\mathrm{T}} \quad (16)$$

被称为矩阵$\Phi$的Moore-Penrose广义逆，它可以被看成是逆矩阵在非方阵的泛化。当然如果其是方阵，使用定理$(AB)^{-1}=B^{-1}A^{-1}$可以容易看出$\boldsymbol{\Phi}^{\dagger}\equiv\boldsymbol{\Phi}^{\mathrm{T}}$

此时我们可以对偏置参数$\omega_0$的作用有一些了解。如果我们显式地将偏置项写出来，式(11)中的误差函数就变成

$$\frac{1}{2}\sum_{n=1}^N\{t_n-w_0-\sum_{j=1}^{M-1}w_j\phi_j(\mathbf{x}_n)\}^2 \quad (17)$$

令该式对$\omega_0$的导数为0并化简，我们得到

$$w_0=\bar{t}-\sum_{j=1}^{M-1}w_j\overline{\phi_j} \quad (18)$$

其中我们定义

$$\bar{t}=\frac{1}{N}\sum_{n=1}^Nt_n,\quad\overline{\phi_j}=\frac{1}{N}\sum_{n=1}^N\phi_j(\mathbf{x}_n) \quad (19)$$

可以看到偏差$\omega_0$补偿了目标值的平均值（在训练集上）与基函数值的平均值的加权和之间的差异。

同样的，我们可以对取对数后的似然函数对方差$\sigma^2$求导，得到

$$\sigma_{\mathrm{ML}}^2=\frac{1}{N}\sum_{n=1}^N\{t_n-\mathbf{w}_{\mathrm{ML}}^\mathrm{T}\boldsymbol{\phi}(\mathbf{x}_n)\}^2 \quad (20)$$

因此，我们看到方差参数的最大似然值由回归函数周围目标值的残差方差给出。

### 最小二乘法的几何意义

我们考虑一个N维空间的轴由$t_n$来定，因此**t**=$(t_1,...,t_n)^T$是空间中的一个向量。每个基函数$\phi_j(\textbf{x}_n)$可以表示为N个点，也可以看成是空间中的一个向量$\varphi_j$，如图4.3。可以注意到$\varphi_j$表示矩阵$\Phi$的第j列，而$\phi(\textbf{x}_n)$表示矩阵$\Phi$第n行的转置。如果M小于N，M个向量$\phi_j(\textbf{x}_n)$会从N维空间中分割出一个维度为M的子空间。我们定义向量**y**表示第n个分量由$y(\textbf{x}_n,\textbf{w})$给出的N维向量。因为**y**是向量组$\varphi_j$的任意组合，因此**y**也处于M维子空间中。这时候均方误差就表示**y**和**t**的平方欧几里得距离。因此，最小二乘法算出来的**w**表示在子空间S中选择出离**t**最近的**y**的参数。在图4.3中直观的来说，我们希望**y**是**t**在子空间S上的垂直投影。

在实际应用中，如果$\Phi^T\Phi$接近奇异值的时候，直接求解方程可能会导致数值困难。特别是，当两个活多个基向量$\varphi_j$共线或几乎共线时，得到的参数值可能具有较大的范围。在处理真是数据集的时候，这种近乎退化的情况并不罕见。由此产生的数值困难可以使用奇异值分解的技术来解决。注意，添加正则化项可以确保矩阵是非奇异的，即使在存在退化的情况下也是如此。

> **注意**: 这点在后面的 1.6正则化最小二乘会详细阐述

![4_3](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/4_3.png)

*最小二乘法的几何解释*

### 顺序学习

极大似然估计的方法设计一次性处理整个训练集，被称为批处理方法。这种方法对于发数据集来说计算成本变得很高。如果使用顺序算法（也被称为在线算法）可能是更好的。在这种算法中，依次考虑一个数据点，并在每次实施后更新模型参数。

我们可以通过一个叫随机梯度下降也叫顺序梯度下降的技巧来实现顺序学习算法。具体的，如果误差函数被表示为和的形式如$E=\sum_nE_n$，在数据点n给出后，随机梯度下降算法使用以下式子来更新参数向量**w**

$$\mathbf{w}^{(\tau+1)}=\mathbf{w}^{(\tau)}-\eta\nabla E_n \quad (21)$$

其中$\tau$表示轮数，$\eta$表示学习率。参数**w**的值被初始化为$\textbf{w}_0$。把式(11)代入上式可得

$$\mathbf{w}^{(\tau+1)}=\mathbf{w}^{(\tau)}+\eta(t_n-\mathbf{w}^{(\tau)\mathrm{T}}\boldsymbol{\phi}_n)\boldsymbol{\phi}_n \quad (22)$$

其中$\phi_n=\phi(\textbf{x}_n)$。这个被称为最小均方或LMS算法。

### 正则化最小二乘

我们先前介绍了在误差函数中天机啊正则项来控制过拟合的想法，因此总的误差函数的形式为

$$E_D(\mathbf{w})+\lambda E_W(\mathbf{w}) \quad (23)$$

其中$\lambda$是正则项的系数，用于控制依赖数据的误差$E_D(\textbf{w})$和正则项$E_W(\textbf{w})$的相对重要性。最简单的正则项形式是使用权重向量的平方和

$$E_W(\mathbf{w})=\frac{1}{2}\sum_jw_j^2=\frac{1}{2}\mathbf{w}^\mathrm{T}\mathbf{w}. \quad (24)$$

如果我们考虑均方误差函数

$$E_D(\mathbf{w})=\frac{1}{2}\sum_{n=1}^N\{t_n-\mathbf{w}^\mathrm{T}\boldsymbol{\phi}(\mathbf{x}_n)\}^2, \quad (25)$$

那么总的误差函数就变成

$$\frac{1}{2}\sum_{n=1}^N\{t_n-\mathbf{w}^\mathrm{T}\boldsymbol{\phi}(\mathbf{x}_n)\}^2+\frac{\lambda}{2}\mathbf{w}^\mathrm{T}\mathbf{w}. \quad (26)$$

在统计学中，这个正则化提供了参数收缩方法的一种示例，因为它将参数值收缩到零。他的优点是误差函数仍是**w**的二次函数，因此它的精确最小值可以以闭合形式找到。具体来说，将式(26)对**w**的梯度设为零，并且求解**w**，得到

$$\mathbf{w}=\left(\lambda\mathbf{I}+\boldsymbol{\Phi}^{\mathrm{T}}\boldsymbol{\Phi}\right)^{-1}\boldsymbol{\Phi}^{\mathrm{T}}\boldsymbol{t}. \quad (27)$$

这提供了一个最小二乘法(式14)的简单的扩展。

### 多输出

目前，我们已经考虑了只有一个目标变量$t$的情况。在一些应用中，我们希望预测多个目标变量。我们可以把它们集中表示为向量$\textbf{t}=(t_1,...,t_K)^T$。通过对$t$的每一个分量都引入不同的基函数集可以实现多重、独立的回归算法。然而，更多的方法使用同一个基函数集来对所有目标向量的分量进行建模，因此表示为

$$\mathbf{y}(\mathbf{x},\mathbf{w})=\mathbf{W}^{\mathrm{T}}\mathbf{\phi}(\mathbf{x}) \quad (28)$$

其中$\mathbf{y}$是一个K 维的列向量，$\mathbf{W}$是一个$M\times K$的矩阵。$\phi(\textbf{x})$是一个 K 维列向量，其中$\phi_0(\textbf{x})=1$。同样的，这个式子也可以表示为一个神经网络表示，如图4.4

![4_4](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/4_4.png)

*类似图4.1,输出端换成了多输出*

考虑使目标向量的条件分布写成一个高斯各向同性的形式(> **注意**: 每个方向的协方差都相等，没有偏好)

$$p(\mathbf{t}|\mathbf{x},\mathbf{W},\sigma^2)=\mathcal{N}(\mathbf{t}|\mathbf{W}^\mathrm{T}\boldsymbol{\phi}(\mathbf{x}),\sigma^2\mathbf{I}). \quad (29)$$

如果我们有一组观测值$t_1,...t_N$，我们可以把它们合并称为一个$N\times K$的矩阵$\textbf{T}$，第 n 行为$\textbf{t}_n^T$。同样的，我们把输入向量$\textbf{x}_1...\textbf{x}_N$合并为$\textbf{X}$。对数似然函数就变为

$$\begin{aligned}\ln p(\mathbf{T}|\mathbf{X},\mathbf{W},\sigma^2) & =\sum_{n=1}^N\ln\mathcal{N}(\mathbf{t}_n|\mathbf{W}^\mathrm{T}\boldsymbol{\phi}(\mathbf{x}_n),\sigma^2\mathbf{I}) \\& =-\frac{NK}{2}\ln\left(2\pi\sigma^2\right)-\frac{1}{2\sigma^2}\sum_{n=1}^N\left\|\mathbf{t}_n-\mathbf{W}^\mathrm{T}\boldsymbol{\phi}(\mathbf{x}_n)\right\|^2.\end{aligned} \quad (30)$$

和之前一样，我们对**W**求导，令它为零，得到

$$\mathbf{W}_{\mathrm{ML}}=\left(\boldsymbol{\Phi}^{\mathrm{T}}\boldsymbol{\Phi}\right)^{-1}\boldsymbol{\Phi}^{\mathrm{T}}\mathbf{T} \quad (31)$$

其中我们把输入特征向量 $\phi(\mathbf{x}_1)$,...,$\phi(\mathbf{x}_N)$ 合并为矩阵$\Phi$。对于每个目标变量$t_k$，我们有

$$\mathbf{w}_k=\left(\boldsymbol{\Phi}^\mathrm{T}\boldsymbol{\Phi}\right)^{-1}\boldsymbol{\Phi}^\mathrm{T}\mathbf{t}_k=\boldsymbol{\Phi}^\dagger\mathbf{t}_k \quad (32)$$

其中 $\mathbf{t}_k$ 是一个分量 $t_{nk}$ 组成的 N 维列向量。因此，回归问题的解在不同的目标变量之间解耦，我们只需要计算一个广义逆矩阵$\Phi \dagger$，这个矩阵由所有向量 $\mathbf{w}_k$ 共享。

对具有任意协方差矩阵的一般高斯噪声分布的扩展是直接的。同样，这导致了与K无关的回归问题的解耦。这一结果并不令人惊讶，因为参数**W**仅定义了高斯噪声分布的均值，我们知道多元高斯均值的最大似然解与其协方差无关。因此，从现在开始，为了简单起见，我们将考虑单个目标变量t。

## 决策理论

先前，我们把回归任务转化为了建立一个条件概率分布$p(t|\textbf{x})$，然后我们选择了高斯噪声模型对真实分布进行建模，得到了**x**依赖的均值$y(\textbf{x},\textbf{w})$。这个均值由参数**w**和方差$\sigma^2$控制。参数**w**和方差$\sigma^2$都可以使用极大似然在数据中学到，其预测分布的结果为

$$p(t|\textbf{x},\textbf{w}_{ML}，\sigma^2_{ML})=\mathcal{N}(t|y(\textbf{x},\textbf{w}_{ML}),\sigma^2_{ML}) \quad (33)$$

这个预测分布表达了我们在重新输入一个新的**x**之后结果的**t**的不确定性。然而，在很多实际的应用中我们需要具体的值$t$而不是一整个分布，特别是当我们需要做一个具体的操作的时候。例如，如果我们的目标是确定用于治疗肿瘤的最佳辐射水平，并且我们的模型预测了辐射剂量的概率分布，那么我们必须使用该分布来决定要施用的特定剂量。因此，我们的任务分为两个阶段。

1. **推理阶段**：我们经过推理得到预测的分布$p(t|\textbf{x})$。
2. **决策阶段**：我们使用得到的分布确定一个具体的值$f(\textbf{x})$，这个值依赖于**x**，并且遵循一系列的最优判别标准。

我们可以使用最小化同时依赖于预测分布$p(t|\textbf{x})$和$f$的**损失函数**。

直觉来说，我们会选择条件分布的均值，因此我们令$f(\textbf{x})=y(\textbf{x},\textbf{w}_{ML})$。在一些例子中这个直觉是成立的，但是在一些情况下结果却很糟糕。因此很有必要构建一种能够让我们理解应该在什么时候施行，应该基于什么前提施行的框架。这种框架被称为**决定理论**(decision theory)。

考虑我们在预测的时候选择一个值$f(\textbf{x})$，并假设此时的真实值是$t$。这么做了之后我们就可以引入某种形式的惩罚或者花费——损失，我们将其表示为$L(t,f(\textbf{x}))$。当然我们不知道真实值$t$，因此我们并不是直接最小化$L$本身，而是最小化$L$的期望，表示为

$$\mathbb{E}[L]=\int\int L(t,f(\mathbf{x}))p(\mathbf{x},t)\operatorname{d}\mathbf{x}\operatorname{d}t \quad (34)$$

其中，我们对输入变量和目标变量的分布进行平均，由它们的联合分布$p(\textbf{x},t)$加权。在回归问题中一个常见的选择是使用均方损失，表示为$L(t,f(\textbf{x}))=\{f(\textbf{x})-t\}^2$。因此，期望损失被写为

$$\mathbb{E}[L]=\int\int\{f(\mathbf{x})-t\}^2p(\mathbf{x},t)\operatorname{d}\mathbf{x}\operatorname{d}t. \quad (35)$$

> **注意**: 值得注意的是，不要弄混**均方损失函数**和前面介绍的**平方和误差函数**。误差函数用来在训练中设置参数，从而确定条件概率分布$p(t|\textbf{x})$，而损失函数控制着如何使用条件分布来对于每一个**x**值确定具体的预测函数$f(\textbf{x})$。

我们的目标是选择一个$f(\textbf{x})$来最小化$\mathbb{E}[L]$。如果我们假设一个完全灵活的函数$f(\textbf{x})$——> **注意**: 这里的灵活我认为是能够适应目标函数，满足符合下述变分式的条件——我们可以使用变分法得到

$$\frac{\delta\mathbb{E}[L]}{\delta f(\mathbf{x})}=2\int\{f(\mathbf{x})-t\}p(\mathbf{x},t)\mathrm{~d}t=0. \quad (36)$$

使用概率的和与积的法则，我们可以推到

$$\begin{aligned}\int f(\textbf{x})p(\textbf{x},t)dt&-\int tp(\textbf{x},t)dt=0\\f(\textbf{x})\int p(\textbf{x},t)dt&=\int tp(\textbf{x},t)dt\\f(\textbf{x})p(\textbf{x})&=\int tp(\textbf{x},t)dt\\f^\star(\mathbf{x})=\frac{1}{p(\mathbf{x})}\int tp(\mathbf{x},t)\mathrm{d}t&=\int tp(t|\mathbf{x})\mathrm{d}t=\mathbb{E}_t[t|\mathbf{x}]\end{aligned} \quad (37)$$

它是以**x**为条件的$t$的条件平均值，被称为**回归函数**。

![4_5](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/4_5.png)

*回归函数$f^*(x)$通过条件分布$​p(t|\textbf{x})$的平均值给出最小化预期平方损失的平均值。*

结果如图4.5所示，并且很容易扩展到多输出的t,此时最优的条件均值解为$f^*(\textbf{x})=\mathbb{E}_t[\textbf{t}|\textbf{x}]$。对于一个高斯条件分布而言，条件均值可以被简化为

$$\mathbb{E}[t|\mathbf{x}]=\int tp(t|\mathbf{x})\mathrm{~d}t=y(\mathbf{x},\mathbf{w}). \quad (38)$$

在式(37)中使用变分法进行推导意味着我们正在优化所有可能的函数$f(\textbf{x})$。
我们可以从一种不同的方式来推导这个问题，同样可以阐明回归问题的本质。首先明确优化问题本质上是一种条件期望，我们可以把平方项扩展为

$$\begin{aligned}& \{f(\mathbf{x})-t\}^2=\{f(\mathbf{x})-\mathbb{E}[t|\mathbf{x}]+\mathbb{E}[t|\mathbf{x}]-t\}^2 \\& =\{f(\mathbf{x})-\mathbb{E}[t|\mathbf{x}]\}^2+2\{f(\mathbf{x})-\mathbb{E}[t|\mathbf{x}]\}\{\mathbb{E}[t|\mathbf{x}]-t\}+\{\mathbb{E}[t|\mathbf{x}]-t\}^2\end{aligned} \quad (39)$$

其中为了符号简洁，我们使用$\mathbb{E}[t|\mathbf{x}]$来表示$\mathbb{E}_t[t|\mathbf{x}]$。代入损失函数式（35）并在t上进行积分，我们看到交叉项消失，我们得到损失函数的表达式

$$\mathbb{E}[L]=\int\left\{f(\mathbf{x})-\mathbb{E}[t|\mathbf{x}]\right\}^2p(\mathbf{x})\operatorname{d}\mathbf{x}+\int\operatorname{var}\left[t|\mathbf{x}\right]p(\mathbf{x})\operatorname{d}\mathbf{x}. \quad (40)$$

> **推导说明**:
>
> 这里进行推导
>
> 1. 交叉项$\int 2\{f(\mathbf{x})-\mathbb{E}[t|\mathbf{x}]\}\{\mathbb{E}[t|\mathbf{x}]-t\} p(\mathbf{x},t) dt=0$是因为，首先$f(\mathbf{x})-\mathbb{E}[t|\mathbf{x}]$不依赖$t$,$\int \{\mathbb{E}[t|\mathbf{x}]-t\}p(t|\mathbf{x})p(x)dt=\mathbb{E}[t|\mathbf{x}]-\mathbb{E}[t|\mathbf{x}]=0$
>
> 2. 第三项$\int\{\mathbb{E}[t|\mathbf{x}]-t\}^2p(\mathbf{x},t)dxdt$,$\{\mathbb{E}[t|\mathbf{x}]-t\}^2=\operatorname{var}[t|\mathbf{x}]$

$f(\textbf{x})$只在第一项中，当$f(\textbf{x})=\mathbb{E}[t|\textbf{x}]$的时候，该项最小，而这种情况下该项就会消失。这是我们之前就得到的结果，表明最优最小二乘预测器由条件均值给出，第二项是 t 分布的方差，在 x 上平均，表示目标数据的内在变异性，可以视为噪声。因为它与$f(\textbf{x})$无关，所以它是损失函数的不可约最小值。

均方损失并不是这里的唯一选择，我们可以简单地考虑一种经过简单泛化的函数，被称为**闵可夫斯基损失**，其期望值由下式给出

$$\mathbb{E}[L_q]=\int\int|f(\mathbf{x})-t|^qp(\mathbf{x},t)\operatorname{d}\mathbf{x}\operatorname{d}t, \quad (41)$$

图4.6展示了不同 q 的取值下$|f-t|^q$和$f-t$的关系

![4_6](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/4_6.png)

*不同 q 的损失图像*

## 偏差和方差的权衡

在线性模型中目前有几个问题：

- 太多的基函数会导致过拟合
- 如果限制基函数的数量又会导致欠拟合
- 虽然正则化可以一定程度控制过拟合，但是引出了如何选择系数$\lambda$的问题
- 如果同时考虑权重向量**w**和正则化系数$\lambda$来最小化正则损失显然不行，因为这会使$\lambda=0$

这时候考虑模型复杂性问题中的频率论观点，即**偏差-方差权衡**。

在决策理论中我们使用了很多不同类型的损失函数，当我们引入条件分布$p(t|\textbf{x})$后，他们无一例外的和最优预测有关。我们定义

$$\mathbf{h}(\mathbf{x})=\mathbb{E}[t|\mathbf{x}]=\int tp(t|\mathbf{x})\mathrm{~d}t. \quad (42)$$

使用我们常用的平方损失得

$$\mathbb{E}[L]=\int\left\{f(\mathbf{x})-h(\mathbf{x})\right\}^2p(\mathbf{x})\operatorname{d\mathbf{x}}+\int\int\{h(\mathbf{x})-t\}^2p(\mathbf{x},t)\operatorname{d\mathbf{x}}\operatorname{d}t. \quad (43)$$

其中第二项和$f(\textbf{x})$无关，它表示来自数据上固有噪声时损失最小的可实现值。第一项依赖于函数$f(\textbf{x})$的选择，我们需要找到一个$f(\textbf{x})$来使得这一项最小。上面讲过这里应该取零。

如果我们有无限的数据，无限的计算资源，理论上我们可以在任意的精度下找到回归函数$h(\textbf{x})$，然后这就是$f(\textbf{x})$。但是实际上我们的数据集 $D$只有有限个数$N$个数据点，因此我们无法准确得到$h(\textbf{x})$。

如果我们使用由参数向量**w**控制的函数对$h(\textbf{x})$进行建模，那么从贝叶斯的角度来看，我们模型中的不确定性将通过**w**上的后验分布来表示。然而，频率主义的处理会根据数据集D对**w**进行点估计，并试图通过以下思维实验来解释这一估计的不确定性。假设我们有大量数据集，每个数据集的大小为$N$，并且每个数据集都独立地从分布$p(t,\textbf{x})$获取。对于任何给定的数据集$D$，我们可以运行我们的学习算法并获得预测函数$f(\textbf{x};D)$。来自集合的不同数据集将给出不同的函数，从而产生不同的平方损失值。然后通过取该数据集集合的平均值来评估特定学习算法的性能。

对于式(43)中第一项的被积函数，我们加入特定的数据集$\{f(\textbf{x};D)-h(\textbf{x})\}^2$
同样的我们对其进行类似式(39)的变形

$$\begin{gathered}\{f(\mathbf{x};\mathcal{D})-\mathbb{E}_\mathcal{D}[f(\mathbf{x};\mathcal{D})]+\mathbb{E}_\mathcal{D}[f(\mathbf{x};\mathcal{D})]-h(\mathbf{x})\}^2 \\=\{f(\mathbf{x};\mathcal{D})-\mathbb{E}_\mathcal{D}[f(\mathbf{x};\mathcal{D})]\}^2+\{\mathbb{E}_\mathcal{D}[f(\mathbf{x};\mathcal{D})]-h(\mathbf{x} \\+2\{f(\mathbf{x};\mathcal{D})-\mathbb{E}_{\mathcal{D}}[f(\mathbf{x};\mathcal{D})]\}\{\mathbb{E}_{\mathcal{D}}[f(\mathbf{x};\mathcal{D})]-h(\mathbf{x})\}.\end{gathered} \quad (4)$$

然后我们对它在给定数据集$D$算期望，> **注意**: 注意到交叉项中的$\{\mathbb{E}_{\mathcal{D}}[f(\mathbf{x};\mathcal{D})]-h(\mathbf{x})\}$都是常数和$D$无关，$\{f(\mathbf{x};\mathcal{D})-\mathbb{E}_{\mathcal{D}}[f(\mathbf{x};\mathcal{D})]\}$期望后显然为零，所以交叉项又没了，剩下

$$\begin{aligned}& \mathbb{E}_{\mathcal{D}}\left[\{f(\mathbf{x};\mathcal{D})-h(\mathbf{x})\}^2\right] \\& =\{\mathbb{E}_{\mathcal{D}}[f(\mathbf{x};\mathcal{D})]-h(\mathbf{x})\}^2+\mathbb{E}_{\mathcal{D}}\left[\{f(\mathbf{x};\mathcal{D})-\mathbb{E}_{\mathcal{D}}[f(\mathbf{x};\mathcal{D})]\}^2\right].\end{aligned} \quad (45)$$

该式分为两项

1. 第一项$\{\mathbb{E}_{\mathcal{D}}[f(\mathbf{x};\mathcal{D})]-h(\mathbf{x})\}^2$称为偏差平方($bias^2$)，表示所有数据集的平均预测和期望回归函数的不同程度
2. 第二项$\mathbb{E}_{\mathcal{D}}\left[\{f(\mathbf{x};\mathcal{D})-\mathbb{E}_{\mathcal{D}}[f(\mathbf{x};\mathcal{D})]\}^2\right]$被称为方差，用来衡量每个数据集的结果和所有数据集平均预测之间的不同程度。

现在如果我们把上述过程代入到式(43)中可得期望平方损失为

$$\text{expected loss}=(\mathrm{bias})^2+\mathrm{variance}+\mathrm{noise} \quad (46)$$

其中

$$(\mathrm{bias})^2=\int\{\mathbb{E}_{\mathcal{D}}[f(\mathbf{x};\mathcal{D})]-h(\mathbf{x})\}^2p(\mathbf{x})\operatorname{d}\mathbf{x} \quad (47)$$

$$\mathrm{variance}=\int\mathbb{E}_{\mathcal{D}}\left[\{f(\mathbf{x};\mathcal{D})-\mathbb{E}_{\mathcal{D}}[f(\mathbf{x};\mathcal{D})]\}^2\right]p(\mathbf{x})\operatorname{d}\mathbf{x} \quad (48)$$

$$\mathrm{noise}=\int\int\{h(\mathbf{x})-t\}^2p(\mathbf{x},t)\operatorname{d\mathbf{x}}\operatorname{d}t \quad (49)$$

> **注意**: 这里相当于把损失项的第一项拆开，对于有限的数据集进行考虑，得到偏差和方差；而 noise 就是我们刚提到的因为高斯噪声模型而得到的期望损失最小可实现值。

因此这个损失函数的优化可以被看作是**偏差和方差的权衡**。对于约束少的模型来说可以做到偏差很小，但是方差较大；对于约束较多的模型来说可以做到方差很小，但是偏差较大。

作者在这里做了个实验，对函数$h(x)=sin(2\pi x)$进行采样，拟合，使用式(26)中的平方正则项形式进行拟合，结果如图4.7。

![4_7](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/4_7.png)

*使用三种不同大小的$\lambda$​值得到的结果对比。左边可以看出方差，右边可以看出偏差*

定量地，我们可以计算平均预测

$$\overline{f}(x)=\frac{1}{L}\sum_{l=1}^Lf^{(l)}(x), \quad (50)$$

偏差方和方差的积分式可以用离散均值的方式给出

$$(\mathrm{bias})^2=\frac{1}{N}\sum_{n=1}^N\left\{\overline{f}(x_n)-h(x_n)\right\}^2 \quad (51)$$

$$\mathrm{variance}=\frac{1}{N}\sum_{n=1}^N\frac{1}{L}\sum_{l=1}^L\left\{f^{(l)}(x_n)-\overline{f}(x_n)\right\}^2 \quad (52)$$

图4.8展示了定量的结果对比。

![4_8](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/4_8.png)*可以看到偏差方和方差的曲线趋势是相反的，这进一步印证了权衡。其中偏差方+方差的曲线的最小值和测试误差的最小值出现在相同点$ln\lambda=0.43$*

偏差-方差分解的实用价值有限，因为它基于数据集集合的平均值，而在实践中，我们只有一个观察到的数据集。如果我们有大量给定大小的独立训练集，我们最好将它们组合成一个更大的训练集，这当然会降低对给定模型复杂性的过度拟合程度。然而，偏差-方差分解经常为模型复杂性问题提供有用的见解，尽管我们在本章中从回归问题的角度介绍了它，但潜在的直觉具有广泛的适用性。

