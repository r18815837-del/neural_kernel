import numpy as np
import pytest

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


def test_tensor_reshape_changes_shape():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])

    y = x.reshape(4)
    arr = to_numpy(y)

    assert arr.shape == (4,)
    assert np.allclose(arr, np.array([1.0, 2.0, 3.0, 4.0], dtype=arr.dtype))


def test_tensor_reshape_preserves_values():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])

    y = x.reshape(1, 4)
    arr = to_numpy(y)

    assert arr.shape == (1, 4)
    assert np.allclose(arr, np.array([[1.0, 2.0, 3.0, 4.0]], dtype=arr.dtype))


def test_tensor_transpose_matrix():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])

    y = x.transpose(1, 0)
    arr = to_numpy(y)

    assert arr.shape == (2, 2)
    assert np.allclose(
        arr,
        np.array([[1.0, 3.0], [2.0, 4.0]], dtype=arr.dtype),
    )

def test_tensor_permute_3d():
    x = Tensor(np.arange(24).reshape(2, 3, 4))

    y = x.permute(1, 0, 2)
    arr = to_numpy(y)

    assert arr.shape == (3, 2, 4)
    expected = np.transpose(np.arange(24).reshape(2, 3, 4), (1, 0, 2))
    assert np.allclose(arr, expected)


def test_tensor_unsqueeze_adds_dim():
    x = Tensor([1.0, 2.0, 3.0])

    y = x.unsqueeze(0)
    arr = to_numpy(y)

    assert arr.shape == (1, 3)
    assert np.allclose(arr, np.array([[1.0, 2.0, 3.0]], dtype=arr.dtype))


def test_tensor_unsqueeze_adds_last_dim():
    x = Tensor([1.0, 2.0, 3.0])

    y = x.unsqueeze(1)
    arr = to_numpy(y)

    assert arr.shape == (3, 1)
    assert np.allclose(
        arr,
        np.array([[1.0], [2.0], [3.0]], dtype=arr.dtype),
    )


def test_tensor_squeeze_removes_singleton_dim():
    x = Tensor([[1.0, 2.0, 3.0]])

    y = x.squeeze(0)
    arr = to_numpy(y)

    assert arr.shape == (3,)
    assert np.allclose(arr, np.array([1.0, 2.0, 3.0], dtype=arr.dtype))


def test_tensor_squeeze_without_axis():
    x = Tensor(np.array([[[1.0], [2.0], [3.0]]]))

    y = x.squeeze()
    arr = to_numpy(y)

    assert arr.shape == (3,)
    assert np.allclose(arr, np.array([1.0, 2.0, 3.0], dtype=arr.dtype))


def test_tensor_permute_preserves_values():
    x_np = np.arange(6).reshape(1, 2, 3)
    x = Tensor(x_np)

    y = x.permute(2, 1, 0)
    arr = to_numpy(y)

    expected = np.transpose(x_np, (2, 1, 0))
    assert arr.shape == expected.shape
    assert np.allclose(arr, expected)