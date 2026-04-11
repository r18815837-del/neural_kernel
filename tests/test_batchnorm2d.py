import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.normalization import BatchNorm2d


def test_batchnorm2d_forward_shape():
    x = Tensor(np.random.randn(4, 3, 8, 8), requires_grad=True)
    bn = BatchNorm2d(3)

    y = bn(x)

    assert y.data.shape == (4, 3, 8, 8)


def test_batchnorm2d_backward():
    np.random.seed(42)

    x = Tensor(np.random.randn(4, 3, 8, 8), requires_grad=True)
    bn = BatchNorm2d(3)

    y = bn(x).sum()
    y.backward()

    assert x.grad is not None
    assert bn.weight.grad is not None
    assert bn.bias.grad is not None

    assert x.grad.shape == x.data.shape
    assert bn.weight.grad.shape == bn.weight.data.shape
    assert bn.bias.grad.shape == bn.bias.data.shape


def test_batchnorm2d_updates_running_stats():
    np.random.seed(42)

    x = Tensor(np.random.randn(8, 3, 6, 6), requires_grad=True)
    bn = BatchNorm2d(3)

    mean_before = bn.running_mean.copy()
    var_before = bn.running_var.copy()

    _ = bn(x)

    assert not np.allclose(bn.running_mean, mean_before)
    assert not np.allclose(bn.running_var, var_before)


def test_batchnorm2d_eval_uses_running_stats():
    np.random.seed(42)

    bn = BatchNorm2d(3)

    x_train = Tensor(np.random.randn(8, 3, 6, 6), requires_grad=True)
    _ = bn(x_train)

    running_mean = bn.running_mean.copy()
    running_var = bn.running_var.copy()

    bn.eval()
    x_eval = Tensor(np.random.randn(8, 3, 6, 6), requires_grad=True)
    y = bn(x_eval)

    expected = (x_eval.data - running_mean) / np.sqrt(running_var + bn.eps)
    expected = expected * bn.weight.data + bn.bias.data

    assert np.allclose(y.data, expected)