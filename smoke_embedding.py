import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.layers.embedding import Embedding


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_embedding_forward_shape():
    emb = Embedding(num_embeddings=10, embedding_dim=4)
    idx = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)

    out = emb(idx)
    assert out.shape == (2, 3, 4)


def smoke_embedding_forward_tensor_indices():
    emb = Embedding(num_embeddings=10, embedding_dim=4)
    idx = Tensor(np.array([[1, 2], [3, 4]], dtype=np.int64), requires_grad=False)

    out = emb(idx)
    assert out.shape == (2, 2, 4)


def smoke_embedding_backward():
    emb = Embedding(num_embeddings=10, embedding_dim=4)
    idx = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.int64)

    out = emb(idx)
    loss = out.mean()
    loss.backward()

    assert emb.weight.grad is not None
    assert emb.weight.grad.shape == emb.weight.shape


def smoke_embedding_repeated_indices():
    emb = Embedding(num_embeddings=10, embedding_dim=4)
    idx = np.array([[1, 1, 1]], dtype=np.int64)

    out = emb(idx)
    loss = out.sum()
    loss.backward()

    assert emb.weight.grad is not None
    # только индекс 1 должен получить заметный град
    grad = emb.weight.grad
    nonzero_rows = np.where(np.abs(grad).sum(axis=1) > 0)[0]
    assert np.all(nonzero_rows == np.array([1]))


def smoke_embedding_out_of_range():
    emb = Embedding(num_embeddings=5, embedding_dim=4)

    failed = False
    try:
        _ = emb(np.array([[0, 1, 5]], dtype=np.int64))
    except ValueError:
        failed = True

    assert failed, "Expected ValueError for out-of-range indices"


def main():
    check("embedding forward shape", smoke_embedding_forward_shape)
    check("embedding forward tensor indices", smoke_embedding_forward_tensor_indices)
    check("embedding backward", smoke_embedding_backward)
    check("embedding repeated indices", smoke_embedding_repeated_indices)
    check("embedding out of range", smoke_embedding_out_of_range)


if __name__ == "__main__":
    main()