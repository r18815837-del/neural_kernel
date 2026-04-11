import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.layers.flatten import Flatten


def test_flatten_forward_shape():
    x = Tensor(np.random.randn(4, 3, 5, 5), requires_grad=True)
    layer = Flatten()

    y = layer(x)

    assert y.data.shape == (4, 75)


def test_flatten_backward_shape():
    x = Tensor(np.random.randn(4, 3, 5, 5), requires_grad=True)
    layer = Flatten()

    y = layer(x).sum()
    y.backward()

    assert x.grad is not None
    assert x.grad.shape == x.data.shape