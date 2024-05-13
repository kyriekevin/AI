import torch
from icecream import ic

x = torch.tensor([3.0])
y = torch.tensor([2.0])
ic(x + y, x * y, x / y, x**y)

x = torch.arange(4)
ic(x)
ic(x[3])

ic(len(x))
ic(x.shape)

A = torch.arange(20).reshape(5, 4)
ic(A)
ic(A.T)

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
ic(B)
ic(B == B.T)

X = torch.arange(24).reshape(2, 3, 4)
ic(X)

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
ic(A)
ic(A + B)
ic(A * B)

a = 2
X = torch.arange(24).reshape(2, 3, 4)
ic(X)
ic(a + X)
ic((a * X).shape)

x = torch.arange(4, dtype=torch.float32)
ic(x)
ic(x.sum())

ic(A)
ic(A.shape)
ic(A.sum())

A_sum_axis0 = A.sum(axis=0)
ic(A)
ic(A_sum_axis0)
ic(A_sum_axis0.shape)

A_sum_axis1 = A.sum(axis=1)
ic(A)
ic(A_sum_axis1)
ic(A_sum_axis1.shape)

ic(A)
ic(A.sum(axis=[0, 1]))

ic(A.mean())
ic(A.sum() / A.numel())

ic(A.mean(axis=0))
ic(A.sum(axis=0) / A.shape[0])

sum_A = A.sum(axis=1, keepdims=True)
ic(sum_A)

ic(A)
ic(sum_A)
ic(A / sum_A)

ic(A.cumsum(axis=0))

y = torch.ones(4, dtype=torch.float32)
ic(x)
ic(y)
ic(torch.dot(x, y))
ic(torch.sum(x * y))

ic(A, x)
ic(A.shape, x.shape)
ic(torch.mv(A, x))

B = torch.ones(4, 3)
ic(torch.mm(A, B))

u = torch.tensor([3.0, -4.0])
ic(torch.norm(u))

ic(torch.abs(u).sum())

ic(torch.norm(torch.ones((4, 9))))

t = torch.ones(4, 9)
t *= -1
ic(t)
ic(torch.norm(t * t))
