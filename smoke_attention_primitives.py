from kernel.core.tensor import Tensor
import numpy as np


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def softmax_tensor(x, axis=-1):
    from kernel.autograd.ops.math_ops import softmax
    return softmax(x, axis=axis)


def smoke_softmax_2d():
    x = Tensor([[1.0, 2.0, 3.0],
                [0.0, 0.0, 0.0]], requires_grad=True)
    y = softmax_tensor(x, axis=-1)

    assert y.shape == (2, 3)
    row_sums = y.numpy().sum(axis=-1)
    assert np.allclose(row_sums, np.ones(2))


def smoke_softmax_3d():
    x = Tensor(np.arange(24, dtype=np.float64).reshape(2, 3, 4), requires_grad=True)
    y = softmax_tensor(x, axis=-1)

    assert y.shape == (2, 3, 4)
    sums = y.numpy().sum(axis=-1)
    assert np.allclose(sums, np.ones((2, 3)))


def smoke_softmax_4d():
    x = Tensor(np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5), requires_grad=True)
    y = softmax_tensor(x, axis=-1)

    assert y.shape == (2, 3, 4, 5)
    sums = y.numpy().sum(axis=-1)
    assert np.allclose(sums, np.ones((2, 3, 4)))


def smoke_scalar_scaling():
    x = Tensor([[2.0, 4.0], [6.0, 8.0]], requires_grad=True)
    y = x / 2.0

    assert y.shape == (2, 2)
    assert np.allclose(y.numpy(), np.array([[1.0, 2.0], [3.0, 4.0]]))


def smoke_mask_like_broadcast_add():
    scores = Tensor(np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5), requires_grad=True)
    mask = Tensor(np.zeros((2, 1, 1, 5), dtype=np.float64), requires_grad=False)

    out = scores + mask
    assert out.shape == (2, 3, 4, 5)


def smoke_batched_attention_shape_path():
    B, T, H, D = 2, 5, 3, 4

    q = Tensor(np.arange(B * T * H * D, dtype=np.float64).reshape(B, T, H, D), requires_grad=True)
    k = Tensor(np.arange(B * T * H * D, dtype=np.float64).reshape(B, T, H, D), requires_grad=True)
    v = Tensor(np.arange(B * T * H * D, dtype=np.float64).reshape(B, T, H, D), requires_grad=True)

    qh = q.transpose(0, 2, 1, 3)      # (B, H, T, D)
    kh = k.transpose(0, 2, 3, 1)      # (B, H, D, T)
    vh = v.transpose(0, 2, 1, 3)      # (B, H, T, D)

    scores = qh @ kh                  # (B, H, T, T)
    scores = scores / np.sqrt(D)

    mask = Tensor(np.zeros((B, 1, 1, T), dtype=np.float64), requires_grad=False)
    scores = scores + mask

    attn = softmax_tensor(scores, axis=-1)   # (B, H, T, T)
    out = attn @ vh                           # (B, H, T, D)

    assert qh.shape == (B, H, T, D)
    assert kh.shape == (B, H, D, T)
    assert scores.shape == (B, H, T, T)
    assert attn.shape == (B, H, T, T)
    assert out.shape == (B, H, T, D)

    sums = attn.numpy().sum(axis=-1)
    assert np.allclose(sums, np.ones((B, H, T)))


def smoke_attention_like_backward():
    B, T, H, D = 2, 4, 2, 3

    q = Tensor(np.arange(B * T * H * D, dtype=np.float64).reshape(B, T, H, D), requires_grad=True)
    k = Tensor(np.arange(B * T * H * D, dtype=np.float64).reshape(B, T, H, D), requires_grad=True)
    v = Tensor(np.arange(B * T * H * D, dtype=np.float64).reshape(B, T, H, D), requires_grad=True)

    qh = q.transpose(0, 2, 1, 3)
    kh = k.transpose(0, 2, 3, 1)
    vh = v.transpose(0, 2, 1, 3)

    scores = (qh @ kh) / np.sqrt(D)
    attn = softmax_tensor(scores, axis=-1)
    out = attn @ vh
    loss = out.mean()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert q.grad.shape == q.shape
    assert k.grad.shape == k.shape
    assert v.grad.shape == v.shape


def main():
    check("softmax 2d", smoke_softmax_2d)
    check("softmax 3d", smoke_softmax_3d)
    check("softmax 4d", smoke_softmax_4d)
    check("scalar scaling", smoke_scalar_scaling)
    check("mask-like broadcast add", smoke_mask_like_broadcast_add)
    check("batched attention shape path", smoke_batched_attention_shape_path)
    check("attention-like backward", smoke_attention_like_backward)


if __name__ == "__main__":
    main()