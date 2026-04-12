import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.functional.attention import scaled_dot_product_attention


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_attention_forward():
    B, H, T, D = 2, 3, 4, 5

    q = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    k = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    v = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)

    out, attn = scaled_dot_product_attention(q, k, v)

    assert out.shape == (B, H, T, D)
    assert attn.shape == (B, H, T, T)

    sums = attn.numpy().sum(axis=-1)
    assert np.allclose(sums, np.ones((B, H, T)))


def smoke_attention_backward():
    B, H, T, D = 2, 2, 4, 3

    q = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    k = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    v = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)

    out, attn = scaled_dot_product_attention(q, k, v)
    loss = out.mean()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert q.grad.shape == q.shape
    assert k.grad.shape == k.shape
    assert v.grad.shape == v.shape


def smoke_attention_with_mask():
    B, H, T, D = 1, 2, 4, 3

    q = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    k = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    v = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)

    mask_np = np.zeros((B, 1, 1, T), dtype=np.float64)
    mask_np[..., -1] = -1e9
    mask = Tensor(mask_np, requires_grad=False)

    out, attn = scaled_dot_product_attention(q, k, v, mask=mask)

    assert out.shape == (B, H, T, D)
    assert attn.shape == (B, H, T, T)

    probs_last_col = attn.numpy()[..., -1]
    assert np.allclose(probs_last_col, 0.0, atol=1e-6)


def smoke_attention_with_dropout():
    B, H, T, D = 1, 2, 4, 3

    q = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    k = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    v = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)

    out, attn = scaled_dot_product_attention(q, k, v, dropout_p=0.1, training=True)

    assert out.shape == (B, H, T, D)
    assert attn.shape == (B, H, T, T)


def main():
    check("attention forward", smoke_attention_forward)
    check("attention backward", smoke_attention_backward)
    check("attention with mask", smoke_attention_with_mask)
    check("attention with dropout", smoke_attention_with_dropout)


if __name__ == "__main__":
    main()