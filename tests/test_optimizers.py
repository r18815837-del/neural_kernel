import numpy as np

from kernel import Linear, Tensor
from kernel.optim import SGD, Adam, StepLR


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


def clone_array(x):
    return np.array(to_numpy(x), copy=True)


def get_grad(x):
    grad = getattr(x, "grad", None)
    if grad is None:
        return None
    return to_numpy(grad)


def has_nonzero_grad(x):
    grad = get_grad(x)
    return grad is not None and np.any(np.abs(grad) > 0)


def make_linear_problem():
    layer = Linear(3, 2)
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = layer(x)
    loss = y.sum()
    return layer, x, y, loss


def test_sgd_step_updates_weight():
    layer, _, _, loss = make_linear_problem()

    before = clone_array(layer.weight)

    optimizer = SGD(layer.parameters(), lr=0.01)
    loss.backward()
    optimizer.step()

    after = to_numpy(layer.weight)

    assert not np.allclose(before, after)


def test_adam_step_updates_weight():
    layer, _, _, loss = make_linear_problem()

    before = clone_array(layer.weight)

    optimizer = Adam(layer.parameters(), lr=0.01)
    loss.backward()
    optimizer.step()

    after = to_numpy(layer.weight)

    assert not np.allclose(before, after)


def test_sgd_zero_grad_clears_weight_grad():
    layer, _, _, loss = make_linear_problem()

    optimizer = SGD(layer.parameters(), lr=0.01)
    loss.backward()

    assert has_nonzero_grad(layer.weight)

    optimizer.zero_grad()

    grad = get_grad(layer.weight)

    if grad is None:
        return

    assert np.allclose(grad, np.zeros_like(grad))


def test_adam_zero_grad_clears_weight_grad():
    layer, _, _, loss = make_linear_problem()

    optimizer = Adam(layer.parameters(), lr=0.01)
    loss.backward()

    assert has_nonzero_grad(layer.weight)

    optimizer.zero_grad()

    grad = get_grad(layer.weight)

    if grad is None:
        return

    assert np.allclose(grad, np.zeros_like(grad))


def test_bias_updates_when_present():
    layer, _, _, loss = make_linear_problem()

    if getattr(layer, "bias", None) is None:
        return

    before = clone_array(layer.bias)

    optimizer = SGD(layer.parameters(), lr=0.01)
    loss.backward()
    optimizer.step()

    after = to_numpy(layer.bias)

    assert not np.allclose(before, after)


def test_step_lr_changes_learning_rate():
    layer, _, _, _ = make_linear_problem()

    optimizer = SGD(layer.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

    before_lr = getattr(optimizer, "lr", None)
    if before_lr is None and hasattr(optimizer, "param_groups"):
        before_lr = optimizer.param_groups[0]["lr"]

    scheduler.step()

    after_lr = getattr(optimizer, "lr", None)
    if after_lr is None and hasattr(optimizer, "param_groups"):
        after_lr = optimizer.param_groups[0]["lr"]

    assert before_lr is not None
    assert after_lr is not None
    assert after_lr < before_lr


def test_optimizer_step_keeps_weights_finite():
    layer, _, _, loss = make_linear_problem()

    optimizer = SGD(layer.parameters(), lr=0.01)
    loss.backward()
    optimizer.step()

    w = to_numpy(layer.weight)
    assert np.isfinite(w).all()


def test_adam_step_keeps_weights_finite():
    layer, _, _, loss = make_linear_problem()

    optimizer = Adam(layer.parameters(), lr=0.01)
    loss.backward()
    optimizer.step()

    w = to_numpy(layer.weight)
    assert np.isfinite(w).all()