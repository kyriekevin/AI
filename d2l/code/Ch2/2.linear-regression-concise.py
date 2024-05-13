import torch
from torch.utils import data
from d2l import torch as d2l
from icecream import ic
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)

    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)
ic(next(iter(data_iter)))

net = nn.Sequential(nn.Linear(2, 1))

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        l.backward()
        trainer.step()
        trainer.zero_grad()
    l = loss(net(features), labels)
    ic(f"epoch {epoch + 1}, loss {l:f}")

w = net[0].weight.data
b = net[0].bias.data
ic(f"error in estimating w: {true_w - w.reshape(true_w.shape)}")
ic(f"error in estimating b: {true_b - b}")
