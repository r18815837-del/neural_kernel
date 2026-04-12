import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.layers.linear import Linear
from kernel.nn.dropout import Dropout
from kernel.nn.modules.container import Sequential


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_sequential_forward():
    x = Tensor(np.ones((2, 8), dtype=np.float64), requires_grad=True)

    seq = Sequential(
        Linear(8, 16),
        Dropout(0.0),
        Linear(16, 3),
    )

    y = seq(x)
    assert y.shape == (2, 3)


def smoke_sequential_backward():
    x = Tensor(np.ones((2, 8), dtype=np.float64), requires_grad=True)

    seq = Sequential(
        Linear(8, 16),
        Dropout(0.0),
        Linear(16, 3),
    )

    y = seq(x)
    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert seq[0].weight.grad is not None
    assert seq[2].weight.grad is not None


def smoke_sequential_len_iter_getitem():
    seq = Sequential(
        Linear(8, 16),
        Dropout(0.1),
        Linear(16, 3),
    )

    assert len(seq) == 3
    assert seq[0] is not None
    assert seq[1] is not None
    assert seq[2] is not None

    count = 0
    for _ in seq:
        count += 1
    assert count == 3


def smoke_sequential_invalid_type():
    failed = False
    try:
        _ = Sequential(123)
    except TypeError:
        failed = True

    assert failed, "Expected TypeError for invalid Sequential item"


def main():
    check("sequential forward", smoke_sequential_forward)
    check("sequential backward", smoke_sequential_backward)
    check("sequential len/iter/getitem", smoke_sequential_len_iter_getitem)
    check("sequential invalid type", smoke_sequential_invalid_type)


if __name__ == "__main__":
    main()