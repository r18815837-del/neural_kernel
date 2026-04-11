import numpy as np

from kernel.core.tensor import Tensor


def test_tensor_creation():
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    assert x.shape == (2, 2)
    assert x.requires_grad is True
    assert x.grad is None
    assert x.dtype == np.float64


def test_zero_grad_sets_zeros():
    x = Tensor([1.0, 2.0], requires_grad=True)
    x.zero_grad()
    assert np.allclose(x.grad, np.array([0.0, 0.0]))
