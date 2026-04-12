import numpy as np

from kernel import CrossEntropyLoss, Tensor


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


def test_cross_entropy_loss_returns_scalar_like():
    criterion = CrossEntropyLoss()

    logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
    targets = Tensor([0])

    loss = criterion(logits, targets)
    arr = to_numpy(loss)

    assert np.isscalar(arr) or arr.shape == ()
    assert np.isfinite(arr).all()


def test_cross_entropy_loss_backward_produces_logits_grad():
    criterion = CrossEntropyLoss()

    logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
    targets = Tensor([0])

    loss = criterion(logits, targets)
    loss.backward()

    grad = get_grad(logits)

    assert grad is not None
    assert grad.shape == (1, 3)
    assert np.isfinite(grad).all()


def test_cross_entropy_loss_smaller_for_better_logit():
    criterion = CrossEntropyLoss()

    good_logits = Tensor([[5.0, 1.0, 0.1]], requires_grad=True)
    bad_logits = Tensor([[0.2, 1.0, 5.0]], requires_grad=True)
    targets = Tensor([0])

    good_loss = to_numpy(criterion(good_logits, targets))
    bad_loss = to_numpy(criterion(bad_logits, targets))

    assert good_loss < bad_loss


def test_cross_entropy_loss_batch_input():
    criterion = CrossEntropyLoss()

    logits = Tensor(
        [
            [3.0, 1.0, 0.1],
            [0.5, 2.5, 0.3],
        ],
        requires_grad=True,
    )
    targets = Tensor([0, 1])

    loss = criterion(logits, targets)
    arr = to_numpy(loss)

    assert np.isscalar(arr) or arr.shape == ()
    assert np.isfinite(arr).all()

    loss.backward()
    grad = get_grad(logits)

    assert grad is not None
    assert grad.shape == (2, 3)
    assert np.isfinite(grad).all()


def test_cross_entropy_loss_non_negative():
    criterion = CrossEntropyLoss()

    logits = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    targets = Tensor([2])

    loss = criterion(logits, targets)
    arr = to_numpy(loss)

    assert arr >= 0.0


def test_cross_entropy_loss_confident_correct_is_small():
    criterion = CrossEntropyLoss()

    logits = Tensor([[10.0, 0.0, 0.0]], requires_grad=True)
    targets = Tensor([0])

    loss = criterion(logits, targets)
    arr = to_numpy(loss)

    assert arr < 1.0