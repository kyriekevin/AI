#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   Lecture_05_Linear_Regression.py    
@Contact :   2718629413@qq.com
@Software:   PyCharm

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2023/3/11 14:41      zyz        1.0          None
"""
import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("w: ", model.linear.weight.item())
print("b: ", model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print("y_pred: ", y_test.data)