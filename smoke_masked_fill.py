import numpy as np

from kernel.core.tensor import Tensor


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_masked_fill_basic():
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    mask = [[0, 1, 0]]

    y = x.masked_fill(mask, -1.0)

    expected = np.array([[1.0, -1.0, 3.0]])
    assert y.shape == (1, 3)
    assert np.allclose(y.numpy(), expected)


def smoke_masked_fill_bool_mask():
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    mask = [[False, True, True]]

    y = x.masked_fill(mask, 0.0)

    expected = np.array([[1.0, 0.0, 0.0]])
    assert np.allclose(y.numpy(), expected)


def smoke_masked_fill_broadcast():
    x = Tensor(np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4), requires_grad=True)
    mask = Tensor(np.array([[[0, 1, 0, 1]]], dtype=np.float64), requires_grad=False)

    y = x.masked_fill(mask, -5.0)

    expected = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4).copy()
    expected[:, :, 1] = -5.0
    expected[:, :, 3] = -5.0

    assert y.shape == (2, 3, 4)
    assert np.allclose(y.numpy(), expected)


def smoke_masked_fill_backward():
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    mask = [[0, 1, 0]]

    y = x.masked_fill(mask, 0.0)
    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def smoke_masked_fill_attention_like():
    scores = Tensor(
        np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5),
        requires_grad=True,
    )

    mask = Tensor(np.zeros((2, 1, 1, 5), dtype=np.float64), requires_grad=False)
    mask = mask.masked_fill(np.array([[[[0, 0, 0, 0, 1]]]], dtype=np.float64), 1.0)

    out = scores.masked_fill(mask, -1e9)

    assert out.shape == (2, 3, 4, 5)
    arr = out.numpy()
    assert np.allclose(arr[..., -1], -1e9)


def main():
    check("masked_fill basic", smoke_masked_fill_basic)
    check("masked_fill bool mask", smoke_masked_fill_bool_mask)
    check("masked_fill broadcast", smoke_masked_fill_broadcast)
    check("masked_fill backward", smoke_masked_fill_backward)
    check("masked_fill attention-like", smoke_masked_fill_attention_like)


if __name__ == "__main__":
    main()