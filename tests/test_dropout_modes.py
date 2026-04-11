import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.dropout import Dropout


def test_dropout_train_changes_values():
    np.random.seed(42)

    x = Tensor(np.ones((100, 10), dtype=np.float64), requires_grad=False)
    layer = Dropout(p=0.5)
    layer.train()

    y = layer(x)

    assert y.data.shape == x.data.shape
    assert np.any(y.data == 0.0)


def test_dropout_eval_keeps_values():
    np.random.seed(42)

    x_data = np.ones((20, 5), dtype=np.float64)
    x = Tensor(x_data.copy(), requires_grad=False)

    layer = Dropout(p=0.5)
    layer.eval()

    y = layer(x)

    assert np.allclose(y.data, x_data)