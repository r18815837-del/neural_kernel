from kernel.core.tensor import Tensor
import numpy as np


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_transpose_2d():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.transpose(1, 0)

    assert y.shape == (2, 2)
    assert np.allclose(y.numpy(), np.array([[1.0, 3.0], [2.0, 4.0]]))


def smoke_T_2d():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.T

    assert y.shape == (2, 2)
    assert np.allclose(y.numpy(), np.array([[1.0, 3.0], [2.0, 4.0]]))


def smoke_transpose_3d():
    x = Tensor(np.arange(24).reshape(2, 3, 4), requires_grad=True)
    y = x.transpose(0, 2, 1)

    assert y.shape == (2, 4, 3)
    expected = np.transpose(np.arange(24).reshape(2, 3, 4), (0, 2, 1))
    assert np.allclose(y.numpy(), expected)


def smoke_permute_alias():
    x = Tensor(np.arange(24).reshape(2, 3, 4), requires_grad=True)
    y = x.permute(0, 2, 1)

    assert y.shape == (2, 4, 3)
    expected = np.transpose(np.arange(24).reshape(2, 3, 4), (0, 2, 1))
    assert np.allclose(y.numpy(), expected)


def smoke_transpose_backward():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.transpose(1, 0)
    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (2, 2)


def smoke_attention_like_layout():
    x = Tensor(np.arange(2 * 5 * 3 * 4).reshape(2, 5, 3, 4), requires_grad=True)
    y = x.transpose(0, 2, 1, 3)

    assert y.shape == (2, 3, 5, 4)
    expected = np.transpose(np.arange(2 * 5 * 3 * 4).reshape(2, 5, 3, 4), (0, 2, 1, 3))
    assert np.allclose(y.numpy(), expected)


def main():
    check("transpose 2d", smoke_transpose_2d)
    check("T 2d", smoke_T_2d)
    check("transpose 3d", smoke_transpose_3d)
    check("permute alias", smoke_permute_alias)
    check("transpose backward", smoke_transpose_backward)
    check("attention-like layout", smoke_attention_like_layout)


if __name__ == "__main__":
    main()