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


def test_sum_all_elements():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])

    y = x.sum()
    arr = to_numpy(y)

    assert np.allclose(arr, np.array(10.0))


def test_sum_axis0():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])

    y = x.sum(axis=0)
    arr = to_numpy(y)

    expected = np.array([4.0, 6.0], dtype=arr.dtype)
    assert arr.shape == (2,)
    assert np.allclose(arr, expected)


def test_sum_axis1():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])

    y = x.sum(axis=1)
    arr = to_numpy(y)

    expected = np.array([3.0, 7.0], dtype=arr.dtype)
    assert arr.shape == (2,)
    assert np.allclose(arr, expected)


def test_sum_axis1_keepdims():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])

    y = x.sum(axis=1, keepdims=True)
    arr = to_numpy(y)

    expected = np.array([[3.0], [7.0]], dtype=arr.dtype)
    assert arr.shape == (2, 1)
    assert np.allclose(arr, expected)


def test_mean_all_elements():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])

    y = x.mean()
    arr = to_numpy(y)

    assert np.allclose(arr, np.array(2.5))


def test_mean_axis0():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])

    y = x.mean(axis=0)
    arr = to_numpy(y)

    expected = np.array([2.0, 3.0], dtype=arr.dtype)
    assert arr.shape == (2,)
    assert np.allclose(arr, expected)


def test_mean_axis1():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])

    y = x.mean(axis=1)
    arr = to_numpy(y)

    expected = np.array([1.5, 3.5], dtype=arr.dtype)
    assert arr.shape == (2,)
    assert np.allclose(arr, expected)


def test_mean_axis1_keepdims():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])

    y = x.mean(axis=1, keepdims=True)
    arr = to_numpy(y)

    expected = np.array([[1.5], [3.5]], dtype=arr.dtype)
    assert arr.shape == (2, 1)
    assert np.allclose(arr, expected)


def test_sum_matches_numpy():
    x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
    x = Tensor(x_np)

    y = x.sum(axis=0)
    arr = to_numpy(y)

    expected = x_np.sum(axis=0)
    assert np.allclose(arr, expected)


def test_mean_matches_numpy_keepdims():
    x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
    x = Tensor(x_np)

    y = x.mean(axis=1, keepdims=True)
    arr = to_numpy(y)

    expected = x_np.mean(axis=1, keepdims=True)
    assert arr.shape == expected.shape
    assert np.allclose(arr, expected)