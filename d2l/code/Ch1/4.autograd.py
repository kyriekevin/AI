import torch
from icecream import ic

x = torch.arange(4.0)
ic(x)

x.requires_grad_(True)
ic(x.grad)

y = 2 * torch.dot(x, x)
ic(y)

y.backward()
ic(x.grad)
ic(x.grad == 4 * x)

x.grad.zero_()
y = x.sum()
ic(y)
y.backward()
ic(x.grad)

x.grad.zero_()
y = x * x
ic(y)
ic(y.sum())
y.sum().backward()
ic(x.grad)
ic(x.grad == 2 * x)

x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
ic(x)
ic(y)
ic(z)

z.sum().backward()
ic(x.grad == u)

x.grad.zero_()
y.sum().backward()
ic(x.grad == 2 * x)


def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

ic(a.grad == d / a)
