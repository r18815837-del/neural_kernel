import numpy as np

from kernel import Linear, Tensor


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


def test_linear_forward_shape_single_batch():
    layer = Linear(3, 2)
    x = Tensor([[1.0, 2.0, 3.0]])

    y = layer(x)
    arr = to_numpy(y)

    assert arr.shape == (1, 2)


def test_linear_forward_shape_multi_batch():
    layer = Linear(4, 3)
    x = Tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]])

    y = layer(x)
    arr = to_numpy(y)

    assert arr.shape == (2, 3)


def test_linear_has_weight_parameter():
    layer = Linear(3, 2)

    assert hasattr(layer, "weight")
    assert layer.weight is not None


def test_linear_weight_shape():
    layer = Linear(3, 2)
    w = to_numpy(layer.weight)

    assert w.ndim == 2
    assert sorted(w.shape) == [2, 3]


def test_linear_bias_exists_if_enabled():
    layer = Linear(3, 2)

    assert hasattr(layer, "bias")


def test_linear_backward_produces_input_grad():
    layer = Linear(3, 2)
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)

    y = layer(x)
    loss = y.sum()
    loss.backward()

    x_grad = get_grad(x)

    assert x_grad is not None
    assert x_grad.shape == (1, 3)


def test_linear_backward_produces_weight_grad():
    layer = Linear(3, 2)
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)

    y = layer(x)
    loss = y.sum()
    loss.backward()

    w_grad = get_grad(layer.weight)

    assert w_grad is not None
    assert w_grad.ndim == 2
    assert sorted(w_grad.shape) == [2, 3]


def test_linear_backward_produces_bias_grad_when_present():
    layer = Linear(3, 2)
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)

    y = layer(x)
    loss = y.sum()
    loss.backward()

    if getattr(layer, "bias", None) is None:
        return

    b_grad = get_grad(layer.bias)

    assert b_grad is not None
    assert b_grad.shape in [(2,), (1, 2)]


def test_linear_forward_outputs_finite_values():
    layer = Linear(3, 2)
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    y = layer(x)
    arr = to_numpy(y)

    assert np.isfinite(arr).all()