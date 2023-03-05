#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   Lecture_03_BGD.py    
@Contact :   2718629413@qq.com
@Software:   PyCharm

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2023/3/4 12:15      zyz        1.0          None
"""
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
loss_list = []
w = 1.0
lr = 0.01


def forward(x):
    return x * w


def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2

    return cost / len(xs)


def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)

    return grad / len(xs)


print('Predict before training', 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= lr * grad_val
    loss_list.append(cost_val)
    print('Epoch:', epoch, '\tw=:', w, '\tloss=:', cost_val)
print('Predict after training', 4, forward(4))

epoch = [i for i in range(100)]
plt.plot(epoch, loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
