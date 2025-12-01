---
title: "数学推导"
author: "MCW"
date: "2025-11-27"
output:
  pdf_document:
    pdf_engine: xelatex
header-includes:
  - |
    \usepackage{amsmath}

mainfont: "Noto Serif CJK SC"
CJKmainfont: "Noto Sans CJK SC"
monofont: "Noto Sans Mono CJK SC"
geometry: margin=2cm
fontsize: 12pt
---

## 训练

传统模型的前向过程，在第 $t$ 步 $$z_t\sim \mathcal{N}(\alpha_t x_0,\sigma_t^2I) \tag{1} $$
分数坐标下的前向过程 $$z_t\sim \mathcal{N}_{wrap}(\alpha_t x_0,\sigma_t^2I) \tag{2} $$

其中  $$\mathcal{N}_{wrap}(\alpha_t x_0,\sigma_t^2I) := \sum_{k\in\mathbb{Z}^3}\mathcal{N}(\alpha_t x_0 -k,\sigma_t^2I) \tag{3} $$

称为 *wrapped normal distribution*（微软 MatterGen 附录A.6）

由于在此分布下难以定义“噪声”的概念，因此改为 score-matching 架构，不在进行“噪声匹配”，而是进行“得分函数匹配”（即score function matching）。具体而言，模型预测目标（即得分函数）为：$$\nabla_{z_t}\log p(z_t|x_0) \tag{4} $$
将上述$(2)(3)$式代入，计算得：
$$ \nabla_{z_t}\log p(z_t|x_0) = \sum_{k\in\mathbb{Z}^3} w_k(\frac{\alpha_t x_0 - k - z_t}{\sigma_t^2}) \tag{5}$$
其中$$ w_k = \frac{1}{Z}\exp(- \frac{|| z_t - \alpha_t x_0 + k||_2^2}{2\sigma_t^2}) \tag{6} $$
$$ Z = \sum_{k'\in\mathbb{Z}^3} \exp(- \frac{|| z_t - \alpha_t x_0 + k'||_2^2}{2\sigma_t^2}) \tag{7}$$
从而训练 loss 为 $$\mathcal{L}_{pos} = ||s_{\phi}(z_t, t, L)-\nabla_{z_t}\log p(z_t|x_0)||_2^2 \tag{8}$$
其中 $L$ 为已经先一步采出的 Lattice，而 $s_\phi(.,.,.)$ 是预测得分的神经网络，具体在本工作中采用 Equiformer 架构。

## 采样
原始模型是基于“去噪”视角实现的VDM，但采用得分匹配之后，采样算法需要采用基于SDE的迭代算法（见Song Yang 于 2021 年发的文章 *Score-Based Generative Modeling through Stochastic Differential Equations* ），即通过得分匹配函数来进行逆向生成，因此需要推导从VDM过渡到SDE的迭代系数，事实上，由于SDE的逆向迭代各系数由前向系数、训练得到的得分预测网络唯一确定，因此只需要定出前向迭代各部分系数。

基于噪声的前向过程为
$$z_t = \alpha_t x_0 + \sigma_t \epsilon, \epsilon \sim \mathcal{N}(0,I)$$
且在DDPM、VDM中一般有 $$\alpha_t^2+\sigma_t^2=1 \tag{9}$$
基于SDE的前向、后向过程为
$$dz = f(z,t)dt + g(t)dW \tag{10}$$
$$dz = [f(z,t)-g^2(t)\nabla_{z}\log p(z|x_0)]dt + g(t)dW \tag{11}$$
经过较繁琐的推导可定出
$$f(z,t) = -\frac{1}{2}\beta(t)z \tag{12}$$
$$\beta(t) = -2 \frac{d}{dt}(\log \alpha_t) \tag{13}$$
$$ g(t) = \sqrt{\beta(t)} \tag{14}$$

标准SDE的离散化方法
$$dz = f(z,t)dt + g(t)dW \\ \to z_{i+1} - z_i = f(z_i, i) + g(i)\epsilon, \epsilon \sim \mathcal{N}(0,I)$$
