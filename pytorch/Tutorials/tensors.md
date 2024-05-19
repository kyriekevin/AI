# Tensors

```python
import torch
import numpy as np
```

## Initializing a Tensor

### Directly from data
```python
>>> data = [[1, 2], [3, 4]]
>>> x_data = torch.tensor(data)
>>> x_data
tensor([[1, 2],
        [3, 4]])
```

### From a NumPy array
```python
>>> np_array = np.array(data)
>>> x_np = torch.from_numpy(np_array)
>>> x_np
tensor([[1, 2],
        [3, 4]])
```

### From another tensor
```python
>>> x_ones = torch.ones_like(x_data)
>>> print(f"Ones Tensor: \n {x_ones} \n")
Ones Tensor:
 tensor([[1, 1],
        [1, 1]])
        
>>> x_rand = torch.rand_like(x_data, dtype=torch.float)
>>> print(f"Random Tensor: \n {x_rand} \n")
Random Tensor:
 tensor([[0.7763, 0.8921],
        [0.0248, 0.9663]])
```

### With random or constant values
```python
>>> shape = (2,3,)
>>> rand_tensor = torch.rand(shape)
>>> ones_tensor = torch.ones(shape)
>>> zeros_tensor = torch.zeros(shape)
>>>
>>> print(f"Random Tensor: \n {rand_tensor} \n")
Random Tensor:
 tensor([[0.3310, 0.6151, 0.0701],
        [0.8589, 0.8770, 0.8089]])
```

## Attributes of a Tensor
```python
In [2]: tensor = torch.rand(3, 4)
   ...: ic(tensor.shape)
   ...: ic(tensor.dtype)
   ...: ic(tensor.device)
   ...:
ic| tensor.shape: torch.Size([3, 4])
ic| tensor.dtype: torch.float32
ic| tensor.device: device(type='cpu')
Out[2]: device(type='cpu')
```

## Operations on Tensors
```python
# Linux, Windows
In [3]: if torch.cuda.is_available():
   ...:     tensor = tensor.to('cuda')
   ...: ic(tensor.device)
   ...:
ic| tensor.device: device(type='cpu')
Out[3]: device(type='cpu')

In [4]: # MacOS
   ...: if torch.backends.mps.is_available():
   ...:     tensor = tensor.to('mps')
   ...: ic(tensor.device)
   ...:
ic| tensor.device: device(type='mps', index=0)
Out[4]: device(type='mps', index=0)
```

### Standard numpy-like indexing and slicing
```python
In [5]: tensor = torch.ones(4, 4)
   ...: ic(tensor[0])
   ...: ic(tensor[:, 0])
   ...: ic(tensor[..., -1])
   ...: tensor[:, 1] = 0
   ...: ic(tensor)
   ...:
ic| tensor[0]: tensor([1., 1., 1., 1.])
ic| tensor[:, 0]: tensor([1., 1., 1., 1.])
ic| tensor[..., -1]: tensor([1., 1., 1., 1.])
ic| tensor: tensor([[1., 0., 1., 1.],
                    [1., 0., 1., 1.],
                    [1., 0., 1., 1.],
                    [1., 0., 1., 1.]])
Out[5]:
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

### Joining tensors
```python
In [6]: t1 = torch.cat([tensor, tensor, tensor], dim=1
   ...: )
   ...: ic(t1)
   ...:
ic| t1: tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0
., 1., 1.],
                [1., 0., 1., 1., 1., 0., 1., 1., 1., 0
., 1., 1.],
                [1., 0., 1., 1., 1., 0., 1., 1., 1., 0
., 1., 1.],
                [1., 0., 1., 1., 1., 0., 1., 1., 1., 0
., 1., 1.]])
Out[6]:
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1
.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1
.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1
.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1
.]])
```

### Arithmetic operations
```python
In [7]: y1 = tensor @ tensor.T
   ...: y2 = tensor.matmul(tensor.T)
   ...: y3 = torch.rand_like(y1)
   ...: torch.matmul(tensor, tensor.T, out=y3)
   ...: ic(y1)
   ...:
ic| y1: tensor([[3., 3., 3., 3.],
                [3., 3., 3., 3.],
                [3., 3., 3., 3.],
                [3., 3., 3., 3.]])
Out[7]:
tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])
```

```python
In [8]: z1 = tensor * tensor
   ...: z2 = tensor.mul(tensor)
   ...: z3 = torch.rand_like(tensor)
   ...: torch.mul(tensor, tensor, out=z3)
   ...: ic(z1)
   ...:
ic| z1: tensor([[1., 0., 1., 1.],
                [1., 0., 1., 1.],
                [1., 0., 1., 1.],
                [1., 0., 1., 1.]])
Out[8]:
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

### Single-element tensors
```python
In [9]: agg = tensor.sum()
   ...: agg_item = agg.item()
   ...: ic(agg_item)
   ...: ic(type(agg_item))
   ...:
ic| agg_item: 12.0
ic| type(agg_item): <class 'float'>
Out[9]: float
```

### In-place operations
```python
In [10]: ic(tensor)
    ...: tensor.add_(5)
    ...: ic(tensor)
    ...:
ic| tensor: tensor([[1., 0., 1., 1.],
                    [1., 0., 1., 1.],
                    [1., 0., 1., 1.],
                    [1., 0., 1., 1.]])
ic| tensor: tensor([[6., 5., 6., 6.],
                    [6., 5., 6., 6.],
                    [6., 5., 6., 6.],
                    [6., 5., 6., 6.]])
Out[10]:
tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])
```

## Bridge with NumPy

### Tensor to NumPy array
```python
In [11]: t = torch.ones(5)
    ...: ic(t)
    ...: n = t.numpy()
    ...: ic(n)
    ...:
ic| t: tensor([1., 1., 1., 1., 1.])
ic| n: array([1., 1., 1., 1., 1.], dtype=float32)
Out[11]: array([1., 1., 1., 1., 1.], dtype=float32)
```

```python
In [12]: t.add_(1)
    ...: ic(t)
    ...: ic(n)
    ...:
ic| t: tensor([2., 2., 2., 2., 2.])
ic| n: array([2., 2., 2., 2., 2.], dtype=float32)
Out[12]: array([2., 2., 2., 2., 2.], dtype=float32)
```

### NumPy array to Tensor
```python
In [13]: n = np.ones(5)
    ...: t = torch.from_numpy(n)
    ...: np.add(n, 1, out=n)
    ...: ic(t)
    ...: ic(n)
    ...:
ic| t: tensor([2., 2., 2., 2., 2.], dtype=torch.float6
4)
ic| n: array([2., 2., 2., 2., 2.])
Out[13]: array([2., 2., 2., 2., 2.])
```
