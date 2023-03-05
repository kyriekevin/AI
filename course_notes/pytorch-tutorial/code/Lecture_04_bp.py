#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   Lecture_04_bp.py    
@Contact :   2718629413@qq.com
@Software:   PyCharm

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2023/3/5 11:18      zyz        1.0          None
"""
import torch
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

lr = 0.01
w = torch.tensor([1.0])
w.requires_grad = True

loss_list = []


def forward(x):
    return w * x


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print("Predict before training:", 4, forward(4).item())
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print("\tgrad:", x, y, w.grad.item())
        w.data -= lr * w.grad.data
        w.grad.data.zero_()
    print("process:", epoch, l.item())
    loss_list.append(l.item())
print("Predict after training:", 4, forward(4).item())

epoch = [i for i in range(100)]
plt.plot(epoch, loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
