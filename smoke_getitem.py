import numpy as np

from kernel.core.tensor import Tensor


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_getitem_basic():
    x = Tensor(np.arange(24, dtype=np.float64).reshape(2, 3, 4), requires_grad=True)
    y = x[:, 1, :]

    assert y.shape == (2, 4)
    expected = np.arange(24, dtype=np.float64).reshape(2, 3, 4)[:, 1, :]
    assert np.allclose(y.numpy(), expected)


def smoke_getitem_single_index():
    x = Tensor(np.arange(24, dtype=np.float64).reshape(2, 3, 4), requires_grad=True)
    y = x[0]

    assert y.shape == (3, 4)
    expected = np.arange(24, dtype=np.float64).reshape(2, 3, 4)[0]
    assert np.allclose(y.numpy(), expected)


def smoke_getitem_last_token():
    x = Tensor(np.arange(2 * 5 * 8, dtype=np.float64).reshape(2, 5, 8), requires_grad=True)
    y = x[:, -1, :]

    assert y.shape == (2, 8)
    expected = np.arange(2 * 5 * 8, dtype=np.float64).reshape(2, 5, 8)[:, -1, :]
    assert np.allclose(y.numpy(), expected)


def smoke_getitem_backward():
    x = Tensor(np.arange(2 * 5 * 8, dtype=np.float64).reshape(2, 5, 8), requires_grad=True)
    y = x[:, 0, :]
    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def smoke_getitem_scalar_like():
    x = Tensor(np.arange(6, dtype=np.float64).reshape(2, 3), requires_grad=True)
    y = x[1, 2]

    assert y.shape == ()
    expected = np.array(5.0)
    assert np.allclose(y.numpy(), expected)


def main():
    check("getitem basic", smoke_getitem_basic)
    check("getitem single index", smoke_getitem_single_index)
    check("getitem last token", smoke_getitem_last_token)
    check("getitem backward", smoke_getitem_backward)
    check("getitem scalar-like", smoke_getitem_scalar_like)


if __name__ == "__main__":
    main()