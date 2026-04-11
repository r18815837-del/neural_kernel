import numpy as np

from kernel.core.tensor import Tensor


def test_add_backward():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    z = (x + y).sum()
    z.backward()
    assert np.allclose(x.grad, np.ones(3))
    assert np.allclose(y.grad, np.ones(3))


def test_mul_backward():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    z = (x * y).sum()
    z.backward()
    assert np.allclose(x.grad, y.data)
    assert np.allclose(y.grad, x.data)


def test_mean_backward():
    x = Tensor([2.0, 4.0], requires_grad=True)
    y = x.mean()
    y.backward()
    assert np.allclose(x.grad, np.array([0.5, 0.5]))


def test_broadcast_backward():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[1.0, 2.0]], requires_grad=True)
    y = (x + b).sum()
    y.backward()
    assert np.allclose(x.grad, np.ones((2, 2)))
    assert np.allclose(b.grad, np.array([[2.0, 2.0]]))
