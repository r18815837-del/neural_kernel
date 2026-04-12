import numpy as np

from kernel import BatchNorm1d, BatchNorm2d, LayerNorm, Tensor


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


def get_grad(x):
    grad = getattr(x, "grad", None)
    if grad is None:
        return None
    return to_numpy(grad)


def test_batchnorm1d_preserves_shape():
    layer = BatchNorm1d(4)
    layer.train()

    x = Tensor(np.random.randn(8, 4).astype(np.float32), requires_grad=True)
    y = layer(x)
    arr = to_numpy(y)

    assert arr.shape == (8, 4)
    assert np.isfinite(arr).all()


def test_batchnorm1d_eval_preserves_shape():
    layer = BatchNorm1d(4)
    layer.eval()

    x = Tensor(np.random.randn(8, 4).astype(np.float32))
    y = layer(x)
    arr = to_numpy(y)

    assert arr.shape == (8, 4)
    assert np.isfinite(arr).all()


def test_batchnorm1d_backward_produces_input_grad():
    layer = BatchNorm1d(4)
    layer.train()

    x = Tensor(np.random.randn(8, 4).astype(np.float32), requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()

    grad = get_grad(x)

    assert grad is not None
    assert grad.shape == (8, 4)
    assert np.isfinite(grad).all()


def test_batchnorm2d_preserves_shape():
    layer = BatchNorm2d(3)
    layer.train()

    x = Tensor(np.random.randn(2, 3, 4, 4).astype(np.float32), requires_grad=True)
    y = layer(x)
    arr = to_numpy(y)

    assert arr.shape == (2, 3, 4, 4)
    assert np.isfinite(arr).all()


def test_batchnorm2d_eval_preserves_shape():
    layer = BatchNorm2d(3)
    layer.eval()

    x = Tensor(np.random.randn(2, 3, 4, 4).astype(np.float32))
    y = layer(x)
    arr = to_numpy(y)

    assert arr.shape == (2, 3, 4, 4)
    assert np.isfinite(arr).all()


def test_batchnorm2d_backward_produces_input_grad():
    layer = BatchNorm2d(3)
    layer.train()

    x = Tensor(np.random.randn(2, 3, 4, 4).astype(np.float32), requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()

    grad = get_grad(x)

    assert grad is not None
    assert grad.shape == (2, 3, 4, 4)
    assert np.isfinite(grad).all()


def test_layernorm_preserves_shape():
    layer = LayerNorm(4)

    x = Tensor(np.random.randn(6, 4).astype(np.float32), requires_grad=True)
    y = layer(x)
    arr = to_numpy(y)

    assert arr.shape == (6, 4)
    assert np.isfinite(arr).all()


def test_layernorm_backward_produces_input_grad():
    layer = LayerNorm(4)

    x = Tensor(np.random.randn(6, 4).astype(np.float32), requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()

    grad = get_grad(x)

    assert grad is not None
    assert grad.shape == (6, 4)
    assert np.isfinite(grad).all()


def test_layernorm_3d_input_preserves_shape():
    layer = LayerNorm(4)

    x = Tensor(np.random.randn(2, 3, 4).astype(np.float32))
    y = layer(x)
    arr = to_numpy(y)

    assert arr.shape == (2, 3, 4)
    assert np.isfinite(arr).all()