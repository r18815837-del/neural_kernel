import numpy as np

from kernel import Tensor
from kernel.nn.layers import Embedding


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


def test_embedding_forward_shape_2d_indices():
    emb = Embedding(10, 4)
    x = Tensor([[1, 2, 3], [4, 5, 6]])

    y = emb(x)
    arr = to_numpy(y)

    assert arr.shape == (2, 3, 4)


def test_embedding_forward_shape_1d_indices():
    emb = Embedding(8, 5)
    x = Tensor([1, 3, 4])

    y = emb(x)
    arr = to_numpy(y)

    assert arr.shape == (3, 5)


def test_embedding_same_index_same_vector():
    emb = Embedding(10, 4)
    x = Tensor([2, 2, 2])

    y = emb(x)
    arr = to_numpy(y)

    assert np.allclose(arr[0], arr[1])
    assert np.allclose(arr[1], arr[2])


def test_embedding_different_indices_usually_different_vectors():
    emb = Embedding(10, 4)
    x = Tensor([1, 2])

    y = emb(x)
    arr = to_numpy(y)

    assert arr.shape == (2, 4)
    assert not np.allclose(arr[0], arr[1])


def test_embedding_has_weight():
    emb = Embedding(10, 4)

    assert hasattr(emb, "weight")
    assert emb.weight is not None

    w = to_numpy(emb.weight)
    assert w.ndim == 2
    assert w.shape == (10, 4)


def test_embedding_backward_produces_weight_grad():
    emb = Embedding(10, 4)
    x = Tensor([1, 2, 3])

    y = emb(x)
    loss = y.sum()
    loss.backward()

    grad = get_grad(emb.weight)

    assert grad is not None
    assert grad.shape == (10, 4)
    assert np.isfinite(grad).all()


def test_embedding_forward_outputs_finite_values():
    emb = Embedding(10, 4)
    x = Tensor([[0, 1], [2, 3]])

    y = emb(x)
    arr = to_numpy(y)

    assert np.isfinite(arr).all()