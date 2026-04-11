import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.layers.linear import Linear


def test_linear_forward_shape():
    layer = Linear(3, 2)
    x = Tensor(np.random.randn(4, 3))
    y = layer(x)
    assert y.shape == (4, 2)


def test_linear_backward_has_grads():
    layer = Linear(2, 1)
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = layer(x).sum()
    y.backward()
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert x.grad is not None
