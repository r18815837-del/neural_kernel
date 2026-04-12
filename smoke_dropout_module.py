import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.dropout import Dropout


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_dropout_forward_train():
    np.random.seed(42)

    x = Tensor(np.ones((4, 4), dtype=np.float64), requires_grad=True)
    drop = Dropout(p=0.5)
    drop.train()

    y = drop(x)

    assert y.shape == x.shape
    assert y.numpy().shape == (4, 4)


def smoke_dropout_forward_eval():
    x = Tensor(np.ones((4, 4), dtype=np.float64), requires_grad=True)
    drop = Dropout(p=0.5)
    drop.eval()

    y = drop(x)

    assert y.shape == x.shape
    assert np.allclose(y.numpy(), x.numpy())


def smoke_dropout_backward():
    np.random.seed(42)

    x = Tensor(np.ones((4, 4), dtype=np.float64), requires_grad=True)
    drop = Dropout(p=0.5)
    drop.train()

    y = drop(x)
    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def smoke_dropout_invalid_p():
    failed = False
    try:
        _ = Dropout(p=1.0)
    except ValueError:
        failed = True

    assert failed, "Expected ValueError for invalid dropout probability"


def main():
    check("dropout module forward train", smoke_dropout_forward_train)
    check("dropout module forward eval", smoke_dropout_forward_eval)
    check("dropout module backward", smoke_dropout_backward)
    check("dropout module invalid p", smoke_dropout_invalid_p)


if __name__ == "__main__":
    main()