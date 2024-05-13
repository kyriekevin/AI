# torch.is_nonzero

## Parameters
```
input (Tensor) – the input tensor.
```

Example:
```
>>> torch.is_nonzero(torch.tensor([0.]))
False
>>> torch.is_nonzero(torch.tensor([1.5]))
True
>>> torch.is_nonzero(torch.tensor([False]))
False
>>> torch.is_nonzero(torch.tensor([3]))
True
>>> torch.is_nonzero(torch.tensor([1, 3, 5]))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Boolean value of Tensor with more than one value is ambiguous
>>> torch.is_nonzero(torch.tensor([]))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Boolean value of Tensor with no values is ambiguous
```
