#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   Lecture_02_Linear_Model.py    
@Contact :   2718629413@qq.com
@Software:   PyCharm

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2023/3/2 16:12      zyz        1.0          None
"""

import numpy as np
import matplotlib.pyplot as plt


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w_list, mse_list = [], []

for w in np.arange(0.0, 4.1, 0.1):
    print("w=", w)
    loss_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        loss_sum += loss_val
        print("x_val:", x_val)
        print('y_val:', y_val)
        print('y_pred_val:', y_pred_val)
        print('loss:', loss_val, '\n')
    MSE = loss_sum / len(x_data)
    print('MSE=', MSE, '\n')
    w_list.append(w)
    mse_list.append(MSE)

plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()
