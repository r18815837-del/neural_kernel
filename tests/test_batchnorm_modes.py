import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.normalization import BatchNorm2d


def test_batchnorm2d_train_updates_running_stats():
    np.random.seed(42)

    bn = BatchNorm2d(3)
    bn.train()

    x = Tensor(np.random.randn(8, 3, 4, 4), requires_grad=False)

    old_mean = bn.running_mean.copy()
    old_var = bn.running_var.copy()

    _ = bn(x)

    assert not np.allclose(bn.running_mean, old_mean)
    assert not np.allclose(bn.running_var, old_var)


def test_batchnorm2d_eval_uses_running_stats():
    np.random.seed(42)

    bn = BatchNorm2d(3)
    bn.train()

    x_train = Tensor(np.random.randn(8, 3, 4, 4), requires_grad=False)
    _ = bn(x_train)

    running_mean = bn.running_mean.copy()
    running_var = bn.running_var.copy()

    bn.eval()

    x_test = Tensor(np.random.randn(8, 3, 4, 4), requires_grad=False)
    y = bn(x_test)

    assert np.allclose(bn.running_mean, running_mean)
    assert np.allclose(bn.running_var, running_var)
    assert y.data.shape == x_test.data.shape