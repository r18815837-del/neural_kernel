from kernel.core.tensor import Tensor
import numpy as np


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_unsqueeze_axis0():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x.unsqueeze(0)

    assert y.shape == (1, 3)
    assert np.allclose(y.numpy(), np.array([[1.0, 2.0, 3.0]]))


def smoke_unsqueeze_axis1():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x.unsqueeze(1)

    assert y.shape == (3, 1)
    assert np.allclose(y.numpy(), np.array([[1.0], [2.0], [3.0]]))


def smoke_unsqueeze_negative_axis():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x.unsqueeze(-1)

    assert y.shape == (3, 1)


def smoke_squeeze_axis():
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = x.squeeze(0)

    assert y.shape == (3,)
    assert np.allclose(y.numpy(), np.array([1.0, 2.0, 3.0]))


def smoke_squeeze_all():
    x = Tensor([[[1.0], [2.0], [3.0]]], requires_grad=True)
    y = x.squeeze()

    assert y.shape == (3,)
    assert np.allclose(y.numpy(), np.array([1.0, 2.0, 3.0]))


def smoke_unsqueeze_backward():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x.unsqueeze(0)
    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (3,)


def smoke_squeeze_backward():
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = x.squeeze(0)
    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (1, 3)


def smoke_attention_like_mask_shape():
    x = Tensor([[1.0, 0.0, 1.0]], requires_grad=True)
    m = x.unsqueeze(1).unsqueeze(2)

    assert m.shape == (1, 1, 1, 3)


def smoke_invalid_squeeze():
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)

    failed = False
    try:
        _ = x.squeeze(1)
    except ValueError:
        failed = True

    assert failed, "Expected ValueError when squeezing non-singleton axis"


def main():
    check("unsqueeze axis=0", smoke_unsqueeze_axis0)
    check("unsqueeze axis=1", smoke_unsqueeze_axis1)
    check("unsqueeze negative axis", smoke_unsqueeze_negative_axis)
    check("squeeze axis", smoke_squeeze_axis)
    check("squeeze all", smoke_squeeze_all)
    check("unsqueeze backward", smoke_unsqueeze_backward)
    check("squeeze backward", smoke_squeeze_backward)
    check("attention-like mask shape", smoke_attention_like_mask_shape)
    check("invalid squeeze", smoke_invalid_squeeze)


if __name__ == "__main__":
    main()