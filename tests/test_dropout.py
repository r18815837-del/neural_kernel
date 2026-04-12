import numpy as np

from kernel import Dropout, Tensor


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


def test_dropout_preserves_shape_in_train_mode():
    layer = Dropout(p=0.5)
    layer.train()

    x = Tensor(np.ones((4, 5), dtype=np.float32))
    y = layer(x)
    arr = to_numpy(y)

    assert arr.shape == (4, 5)


def test_dropout_changes_values_in_train_mode():
    layer = Dropout(p=0.5)
    layer.train()

    x = Tensor(np.ones((100, 100), dtype=np.float32))
    y = layer(x)
    arr = to_numpy(y)

    assert arr.shape == (100, 100)
    assert np.any(arr == 0)


def test_dropout_eval_mode_is_identity():
    layer = Dropout(p=0.5)
    layer.eval()

    x = Tensor(np.ones((8, 8), dtype=np.float32))
    y = layer(x)

    x_arr = to_numpy(x)
    y_arr = to_numpy(y)

    assert y_arr.shape == x_arr.shape
    assert np.allclose(y_arr, x_arr)


def test_dropout_zero_probability_is_identity_in_train_mode():
    layer = Dropout(p=0.0)
    layer.train()

    x = Tensor(np.arange(12, dtype=np.float32).reshape(3, 4))
    y = layer(x)

    x_arr = to_numpy(x)
    y_arr = to_numpy(y)

    assert y_arr.shape == x_arr.shape
    assert np.allclose(y_arr, x_arr)


def test_dropout_outputs_finite_values():
    layer = Dropout(p=0.3)
    layer.train()

    x = Tensor(np.ones((16, 16), dtype=np.float32))
    y = layer(x)
    arr = to_numpy(y)

    assert np.isfinite(arr).all()