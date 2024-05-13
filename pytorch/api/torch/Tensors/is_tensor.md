# torch.is_tensor

```python
def is_tensor(obj):
  return isinstance(obj, torch.Tensor)
```

## Parameters
```
obj(Object) - Object to test.
```

Example:
```
>>> x = torch.tensor([1,2,3])
>>> torch.is_tensor(x)
True
```
