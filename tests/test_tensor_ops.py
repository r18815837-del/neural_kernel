import numpy as np
import pytest

from kernel import Tensor


def to_numpy(x):
    """
    Tries to convert Tensor-like object to numpy array.
    Adjust this helper if your Tensor uses a different field/method.
    """
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


def test_tensor_creation_from_list():
    x = Tensor([1.0, 2.0, 3.0])

    arr = to_numpy(x)

    assert arr.shape == (3,)
    assert np.allclose(arr, np.array([1.0, 2.0, 3.0], dtype=arr.dtype))


def test_tensor_addition():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])

    c = a + b
    arr = to_numpy(c)

    assert arr.shape == (3,)
    assert np.allclose(arr, np.array([5.0, 7.0, 9.0], dtype=arr.dtype))


def test_tensor_scalar_multiplication():
    x = Tensor([1.0, 2.0, 3.0])

    y = x * 2.0
    arr = to_numpy(y)

    assert arr.shape == (3,)
    assert np.allclose(arr, np.array([2.0, 4.0, 6.0], dtype=arr.dtype))


def test_tensor_getitem_vector():
    x = Tensor([10.0, 20.0, 30.0])

    y = x[1]
    arr = to_numpy(y)

    # scalar or shape () is acceptable here
    assert np.allclose(arr, np.array(20.0))


def test_tensor_getitem_matrix_row():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])

    y = x[1]
    arr = to_numpy(y)

    assert arr.shape == (2,)
    assert np.allclose(arr, np.array([3.0, 4.0], dtype=arr.dtype))


def test_tensor_concat_axis0():
    a = Tensor([[1.0, 2.0]])
    b = Tensor([[3.0, 4.0]])

    if not hasattr(a, "concat"):
        pytest.skip("concat is not available on Tensor")

    c = a.concat([b], axis=0)
    arr = to_numpy(c)

    assert arr.shape == (2, 2)
    assert np.allclose(
        arr,
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=arr.dtype),
    )


def test_tensor_concat_axis1():
    a = Tensor([[1.0], [2.0]])
    b = Tensor([[3.0], [4.0]])

    if not hasattr(a, "concat"):
        pytest.skip("concat is not available on Tensor")

    c = a.concat([b], axis=1)
    arr = to_numpy(c)

    assert arr.shape == (2, 2)
    assert np.allclose(
        arr,
        np.array([[1.0, 3.0], [2.0, 4.0]], dtype=arr.dtype),
    )

def test_tensor_masked_fill():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = np.array([[True, False], [False, True]])

    if not hasattr(x, "masked_fill"):
        pytest.skip("masked_fill is not available on Tensor")

    y = x.masked_fill(mask, -1.0)
    arr = to_numpy(y)

    expected = np.array([[-1.0, 2.0], [3.0, -1.0]], dtype=arr.dtype)

    assert arr.shape == (2, 2)
    assert np.allclose(arr, expected)


def test_tensor_add_preserves_shape():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[10.0, 20.0], [30.0, 40.0]])

    c = a + b
    arr = to_numpy(c)

    assert arr.shape == (2, 2)