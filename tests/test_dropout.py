import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.dropout import Dropout


def test_dropout_eval_returns_same_values():
    np.random.seed(42)

    x = Tensor(np.ones((4, 4)), requires_grad=True)
    layer = Dropout(p=0.5)
    layer.eval()

    y = layer(x)

    assert np.allclose(y.data, x.data)


def test_dropout_train_changes_values():
    np.random.seed(42)

    x = Tensor(np.ones((100, 100)), requires_grad=True)
    layer = Dropout(p=0.5)
    layer.train()

    y = layer(x)

    assert y.data.shape == x.data.shape
    assert not np.allclose(y.data, x.data)


def test_dropout_train_keeps_expected_mean():
    np.random.seed(42)

    x = Tensor(np.ones((1000, 1000)), requires_grad=True)
    layer = Dropout(p=0.5)
    layer.train()

    y = layer(x)

    mean_value = y.data.mean()
    assert 0.95 <= mean_value <= 1.05


def test_dropout_backward_shape():
    np.random.seed(42)

    x = Tensor(np.ones((8, 8)), requires_grad=True)
    layer = Dropout(p=0.5)
    layer.train()

    y = layer(x).sum()
    y.backward()

    assert x.grad is not None
    assert x.grad.shape == x.data.shape