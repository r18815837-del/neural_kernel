import numpy as np

from kernel.core.tensor import Tensor


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_concat_axis0():
    a = Tensor([[1.0, 2.0]], requires_grad=True)
    b = Tensor([[3.0, 4.0]], requires_grad=True)

    y = Tensor.cat([a, b], axis=0)

    assert y.shape == (2, 2)
    assert np.allclose(y.numpy(), np.array([[1.0, 2.0], [3.0, 4.0]]))


def smoke_concat_axis1():
    a = Tensor([[1.0], [2.0]], requires_grad=True)
    b = Tensor([[3.0], [4.0]], requires_grad=True)

    y = Tensor.cat([a, b], axis=1)

    assert y.shape == (2, 2)
    assert np.allclose(y.numpy(), np.array([[1.0, 3.0], [2.0, 4.0]]))


def smoke_concat_method():
    a = Tensor([[1.0, 2.0]], requires_grad=True)
    b = Tensor([[3.0, 4.0]], requires_grad=True)

    y = a.concat([b], axis=0)

    assert y.shape == (2, 2)
    assert np.allclose(y.numpy(), np.array([[1.0, 2.0], [3.0, 4.0]]))


def smoke_concat_backward():
    a = Tensor([[1.0, 2.0]], requires_grad=True)
    b = Tensor([[3.0, 4.0]], requires_grad=True)

    y = Tensor.cat([a, b], axis=0)
    loss = y.mean()
    loss.backward()

    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape


def smoke_concat_attention_like():
    a = Tensor(np.zeros((2, 1, 8), dtype=np.float64), requires_grad=True)
    b = Tensor(np.ones((2, 5, 8), dtype=np.float64), requires_grad=True)

    y = Tensor.cat([a, b], axis=1)

    assert y.shape == (2, 6, 8)


def main():
    check("concat axis=0", smoke_concat_axis0)
    check("concat axis=1", smoke_concat_axis1)
    check("concat method", smoke_concat_method)
    check("concat backward", smoke_concat_backward)
    check("concat attention-like", smoke_concat_attention_like)


if __name__ == "__main__":
    main()