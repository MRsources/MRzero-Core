# Differentiable Rounding Helpers for Pulseq Zero Sequences

This module provides **differentiable stand-ins** for common rounding operations (`ceil`, `floor`, `round`) used in **Pulseq sequences**.
They are designed to keep the optimization pipeline differentiable by **passing gradients through unchanged**, effectively treating these operations as the identity during backpropagation.


---

## Design idea

* **Forward pass**: Behaves exactly like the corresponding PyTorch / NumPy rounding operation.
* **Backward pass**: Returns the incoming gradient unchanged.
* **Use case**: Allows differentiable optimization of otherwise discrete parameters.


---



## Functional Wrappers

To use the wrapper functions:

```python
from pulseqzero import round, ceil, floor
``` 


### `ceil(x)`

Differentiable version of `torch.ceil`.

```python
y = ceil(x)
```

### `floor(x)`

Differentiable version of `torch.floor`.

```python
y = floor(x)
```

### `round(x)`

Differentiable version of `torch.round`.

```python
y = round(x)
```

---


