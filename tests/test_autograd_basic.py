import numpy as np

from kernel import Tensor


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


def test_backward_square_sum():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

    y = (x * x).sum()
    y.backward()

    grad = get_grad(x)
    expected = np.array([2.0, 4.0, 6.0], dtype=grad.dtype)

    assert grad is not None
    assert grad.shape == (3,)
    assert np.allclose(grad, expected, atol=1e-6)


def test_backward_mean_of_shifted_tensor():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

    y = (x + 2.0).mean()
    y.backward()

    grad = get_grad(x)
    expected = np.array([1 / 3, 1 / 3, 1 / 3], dtype=grad.dtype)

    assert grad is not None
    assert grad.shape == (3,)
    assert np.allclose(grad, expected, atol=1e-6)


def test_backward_scalar_multiplication():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

    y = (x * 5.0).sum()
    y.backward()

    grad = get_grad(x)
    expected = np.array([5.0, 5.0, 5.0], dtype=grad.dtype)

    assert grad is not None
    assert grad.shape == (3,)
    assert np.allclose(grad, expected, atol=1e-6)


def test_backward_on_matrix_sum():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    y = x.sum()
    y.backward()

    grad = get_grad(x)
    expected = np.ones((2, 2), dtype=grad.dtype)

    assert grad is not None
    assert grad.shape == (2, 2)
    assert np.allclose(grad, expected, atol=1e-6)


def test_backward_on_mean_matrix():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    y = x.mean()
    y.backward()

    grad = get_grad(x)
    expected = np.full((2, 2), 0.25, dtype=grad.dtype)

    assert grad is not None
    assert grad.shape == (2, 2)
    assert np.allclose(grad, expected, atol=1e-6)


def test_grad_exists_after_backward():
    x = Tensor([2.0, 4.0, 6.0], requires_grad=True)

    y = (x * x).sum()
    y.backward()

    assert getattr(x, "grad", None) is not None


def test_backward_output_is_scalar_like():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

    y = (x * x).sum()
    y_arr = to_numpy(y)

    assert np.allclose(y_arr, np.array(14.0))