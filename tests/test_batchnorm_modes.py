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


def test_batchnorm2d_train_updates_running_stats():
    np.random.seed(42)

    bn = BatchNorm2d(3)
    bn.train()

    x = Tensor(np.random.randn(8, 3, 4, 4), requires_grad=False)

    old_mean = np.array(to_numpy(bn.running_mean), copy=True)
    old_var = np.array(to_numpy(bn.running_var), copy=True)

    _ = bn(x)

    assert not np.allclose(to_numpy(bn.running_mean), old_mean)
    assert not np.allclose(to_numpy(bn.running_var), old_var)


def test_batchnorm2d_eval_uses_running_stats():
    np.random.seed(42)

    bn = BatchNorm2d(3)
    bn.train()

    x_train = Tensor(np.random.randn(8, 3, 4, 4), requires_grad=False)
    _ = bn(x_train)

    running_mean = np.array(to_numpy(bn.running_mean), copy=True)
    running_var = np.array(to_numpy(bn.running_var), copy=True)

    bn.eval()

    x_test = Tensor(np.random.randn(8, 3, 4, 4), requires_grad=False)
    y = bn(x_test)

    assert np.allclose(to_numpy(bn.running_mean), running_mean)
    assert np.allclose(to_numpy(bn.running_var), running_var)
    assert to_numpy(y).shape == to_numpy(x_test).shape