import torch
from icecream import ic

x = torch.arange(12)
ic(x)
ic(x.shape)
ic(x.numel())

X = x.reshape(3, 4)
ic(X)

ic(torch.zeros((2, 3, 4)))
ic(torch.ones((2, 3, 4)))
ic(torch.randn(3, 4))
ic(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
ic(x + y, x - y, x * y, x / y, x**y)

ic(torch.exp(x))

X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
ic(torch.cat((X, Y), dim=0))
ic(torch.cat((X, Y), dim=1))

ic(X == Y)

ic(X.sum())

a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)
ic(a, b)
ic(a + b)

ic(X[-1], X[1:3])

X[1, 2] = 9
ic(X)

X[:2, :] = 12
ic(X)

before = id(Y)
Y = Y + X
ic(id(Y) == before)

Z = torch.zeros_like(Y)
ic(id(Z))
Z = X + Y
ic(id(Z))
Z += X
ic(id(Z))

before = id(X)
X += Y
ic(id(X) == before)

A = X.numpy()
B = torch.tensor(A)
ic(type(A), type(B))

a = torch.tensor([3.5])
ic(a, a.item(), float(a), int(a))
