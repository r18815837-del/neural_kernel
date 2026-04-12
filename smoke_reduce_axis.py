from kernel.core.tensor import Tensor
import numpy as np


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_sum_all():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.sum()

    assert y.shape == ()
    assert np.allclose(y.numpy(), np.array(10.0))

    y.backward()
    assert x.grad is not None
    assert x.grad.shape == (2, 2)


def smoke_sum_axis1():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.sum(axis=1)

    assert y.shape == (2,)
    assert np.allclose(y.numpy(), np.array([3.0, 7.0]))

    loss = y.mean()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (2, 2)


def smoke_sum_axis1_keepdims():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.sum(axis=1, keepdims=True)

    assert y.shape == (2, 1)
    assert np.allclose(y.numpy(), np.array([[3.0], [7.0]]))

    loss = y.mean()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (2, 2)


def smoke_mean_axis0():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.mean(axis=0)

    assert y.shape == (2,)
    assert np.allclose(y.numpy(), np.array([2.0, 3.0]))

    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (2, 2)


def smoke_mean_multi_axis():
    x = Tensor(np.arange(24, dtype=np.float64).reshape(2, 3, 4), requires_grad=True)
    y = x.mean(axis=(1, 2))

    assert y.shape == (2,)
    expected = np.arange(24, dtype=np.float64).reshape(2, 3, 4).mean(axis=(1, 2))
    assert np.allclose(y.numpy(), expected)

    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (2, 3, 4)


def smoke_mean_negative_axis():
    x = Tensor(np.arange(12, dtype=np.float64).reshape(3, 4), requires_grad=True)
    y = x.mean(axis=-1)

    assert y.shape == (3,)
    expected = np.arange(12, dtype=np.float64).reshape(3, 4).mean(axis=-1)
    assert np.allclose(y.numpy(), expected)


def main():
    check("sum all", smoke_sum_all)
    check("sum axis=1", smoke_sum_axis1)
    check("sum axis=1 keepdims", smoke_sum_axis1_keepdims)
    check("mean axis=0", smoke_mean_axis0)
    check("mean multi axis", smoke_mean_multi_axis)
    check("mean negative axis", smoke_mean_negative_axis)


if __name__ == "__main__":
    main()