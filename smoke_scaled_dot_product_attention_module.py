import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.modules.attention import ScaledDotProductAttention


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_module_forward():
    B, H, T, D = 2, 3, 4, 5

    q = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    k = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    v = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)

    attn = ScaledDotProductAttention(dropout_p=0.0)
    out, weights = attn(q, k, v)

    assert out.shape == (B, H, T, D)
    assert weights.shape == (B, H, T, T)


def smoke_module_backward():
    B, H, T, D = 2, 2, 4, 3

    q = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    k = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    v = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)

    attn = ScaledDotProductAttention(dropout_p=0.0)
    out, weights = attn(q, k, v)

    loss = out.mean()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


def smoke_module_eval_mode():
    B, H, T, D = 1, 2, 4, 3

    q = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    k = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    v = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)

    attn = ScaledDotProductAttention(dropout_p=0.2)
    attn.eval()

    out, weights = attn(q, k, v)

    assert out.shape == (B, H, T, D)
    assert weights.shape == (B, H, T, T)
    assert attn.training is False


def smoke_module_with_mask():
    B, H, T, D = 1, 2, 4, 3

    q = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    k = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    v = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)

    mask_np = np.zeros((B, 1, 1, T), dtype=np.float64)
    mask_np[..., -1] = -1e9
    mask = Tensor(mask_np, requires_grad=False)

    attn = ScaledDotProductAttention(dropout_p=0.0)
    out, weights = attn(q, k, v, mask=mask)

    assert out.shape == (B, H, T, D)
    assert weights.shape == (B, H, T, T)

    probs_last_col = weights.numpy()[..., -1]
    assert np.allclose(probs_last_col, 0.0, atol=1e-6)


def main():
    check("module forward", smoke_module_forward)
    check("module backward", smoke_module_backward)
    check("module eval mode", smoke_module_eval_mode)
    check("module with mask", smoke_module_with_mask)


if __name__ == "__main__":
    main()