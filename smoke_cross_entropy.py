import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.losses import CrossEntropyLoss


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_ce_forward():
    logits = Tensor(
        np.array([[2.0, 1.0, 0.1],
                  [0.5, 2.5, 0.3]], dtype=np.float64),
        requires_grad=True,
    )
    targets = np.array([0, 1], dtype=np.int64)

    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, targets)

    assert loss.shape == ()
    assert float(loss.detach().numpy()) > 0.0


def smoke_ce_backward():
    logits = Tensor(
        np.array([[2.0, 1.0, 0.1],
                  [0.5, 2.5, 0.3]], dtype=np.float64),
        requires_grad=True,
    )
    targets = np.array([0, 1], dtype=np.int64)

    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


def smoke_ce_tensor_targets():
    logits = Tensor(
        np.array([[1.0, 2.0],
                  [2.0, 1.0]], dtype=np.float64),
        requires_grad=True,
    )
    targets = Tensor(np.array([1, 0], dtype=np.int64), requires_grad=False)

    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, targets)

    assert loss.shape == ()


def smoke_ce_out_of_range():
    logits = Tensor(
        np.array([[1.0, 2.0],
                  [2.0, 1.0]], dtype=np.float64),
        requires_grad=True,
    )
    targets = np.array([1, 2], dtype=np.int64)

    loss_fn = CrossEntropyLoss()

    failed = False
    try:
        _ = loss_fn(logits, targets)
    except ValueError:
        failed = True

    assert failed, "Expected ValueError for out-of-range target index"


def main():
    check("ce forward", smoke_ce_forward)
    check("ce backward", smoke_ce_backward)
    check("ce tensor targets", smoke_ce_tensor_targets)
    check("ce out of range", smoke_ce_out_of_range)


if __name__ == "__main__":
    main()