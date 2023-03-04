#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   Lecture_03_SGD.py    
@Contact :   2718629413@qq.com
@Software:   PyCharm

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2023/3/4 13:40      zyz        1.0          None
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

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

def gradient(x, y):
    return 2 * x * (w * x - y)

print('Predict before training', 4, forward(4))
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w -= lr * grad
        print("\tgrad: ", x, y, grad)
        l = loss(x, y)
    loss_list.append(l)
    print("progress:", epoch, "\tw=", w, "\tloss=", l)
print('Predict after training', 4, forward(4))

epoch = [i for i in range(100)]
plt.plot(epoch, loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()