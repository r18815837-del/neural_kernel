import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.normalization import BatchNorm2d


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "data"):
        data = x.data
        if isinstance(data, np.ndarray):
            return data
        try:
            return np.array(data)
        except Exception:
            pass
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.array(x)


def test_batchnorm2d_forward_shape():
    x = Tensor(np.random.randn(4, 3, 8, 8), requires_grad=True)
    bn = BatchNorm2d(3)

    y = bn(x)

    assert to_numpy(y).shape == (4, 3, 8, 8)


def test_batchnorm2d_backward():
    np.random.seed(42)

    x = Tensor(np.random.randn(4, 3, 8, 8), requires_grad=True)
    bn = BatchNorm2d(3)

    y = bn(x).sum()
    y.backward()

    assert x.grad is not None
    assert bn.weight.grad is not None
    assert bn.bias.grad is not None

    assert to_numpy(x.grad).shape == to_numpy(x).shape
    assert to_numpy(bn.weight.grad).shape == to_numpy(bn.weight).shape
    assert to_numpy(bn.bias.grad).shape == to_numpy(bn.bias).shape


def test_batchnorm2d_updates_running_stats():
    np.random.seed(42)

    x = Tensor(np.random.randn(8, 3, 6, 6), requires_grad=True)
    bn = BatchNorm2d(3)

    mean_before = np.array(to_numpy(bn.running_mean), copy=True)
    var_before = np.array(to_numpy(bn.running_var), copy=True)

    _ = bn(x)

    assert not np.allclose(to_numpy(bn.running_mean), mean_before)
    assert not np.allclose(to_numpy(bn.running_var), var_before)


def test_batchnorm2d_eval_uses_running_stats():
    np.random.seed(42)

    bn = BatchNorm2d(3)

    x_train = Tensor(np.random.randn(8, 3, 6, 6), requires_grad=True)
    _ = bn(x_train)

    running_mean = np.array(to_numpy(bn.running_mean), copy=True)
    running_var = np.array(to_numpy(bn.running_var), copy=True)

    bn.eval()
    x_eval = Tensor(np.random.randn(8, 3, 6, 6), requires_grad=True)
    y = bn(x_eval)

    # reshape running stats and affine params for NCHW broadcasting
    rm = running_mean.reshape(1, -1, 1, 1)
    rv = running_var.reshape(1, -1, 1, 1)
    w = to_numpy(bn.weight).reshape(1, -1, 1, 1)
    b = to_numpy(bn.bias).reshape(1, -1, 1, 1)

    expected = (to_numpy(x_eval) - rm) / np.sqrt(rv + bn.eps)
    expected = expected * w + b

    assert np.allclose(to_numpy(y), expected)