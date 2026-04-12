import numpy as np

from kernel.core.tensor import Tensor


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_detach_basic():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x.detach()

    assert y.shape == x.shape
    assert y.device == x.device
    assert y.requires_grad is False
    assert np.allclose(y.numpy(), x.numpy())


def smoke_detach_breaks_graph():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    z = x * 2
    y = z.detach()

    assert y.requires_grad is False
    assert y._ctx is None
    assert y._backward is None


def smoke_detach_independent_data():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x.detach()

    y.data[0] = 999.0

    assert not np.allclose(y.numpy(), x.numpy())


def smoke_detach_numpy():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x.detach()
    arr = y.numpy()

    assert arr.shape == (3,)
    assert np.allclose(arr, np.array([1.0, 2.0, 3.0]))


def main():
    check("detach basic", smoke_detach_basic)
    check("detach breaks graph", smoke_detach_breaks_graph)
    check("detach independent data", smoke_detach_independent_data)
    check("detach numpy", smoke_detach_numpy)


if __name__ == "__main__":
    main()