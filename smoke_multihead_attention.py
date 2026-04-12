import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.modules.multihead_attention import MultiHeadAttention


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_mha_forward():
    B, T, D, H = 2, 5, 8, 2

    x = Tensor(
        np.arange(B * T * D, dtype=np.float64).reshape(B, T, D),
        requires_grad=True,
    )

    mha = MultiHeadAttention(d_model=D, num_heads=H, dropout_p=0.0)
    out, attn = mha(x)

    assert out.shape == (B, T, D)
    assert attn.shape == (B, H, T, T)

    sums = attn.numpy().sum(axis=-1)
    assert np.allclose(sums, np.ones((B, H, T)))


def smoke_mha_backward():
    B, T, D, H = 2, 4, 8, 2

    x = Tensor(
        np.arange(B * T * D, dtype=np.float64).reshape(B, T, D),
        requires_grad=True,
    )

    mha = MultiHeadAttention(d_model=D, num_heads=H, dropout_p=0.0)
    out, attn = mha(x)

    loss = out.mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape

    assert mha.q_proj.weight.grad is not None
    assert mha.k_proj.weight.grad is not None
    assert mha.v_proj.weight.grad is not None
    assert mha.out_proj.weight.grad is not None


def smoke_mha_with_mask():
    B, T, D, H = 1, 4, 8, 2

    x = Tensor(
        np.arange(B * T * D, dtype=np.float64).reshape(B, T, D),
        requires_grad=True,
    )

    mask_np = np.zeros((B, 1, 1, T), dtype=np.float64)
    mask_np[..., -1] = -1e9
    mask = Tensor(mask_np, requires_grad=False)

    mha = MultiHeadAttention(d_model=D, num_heads=H, dropout_p=0.0)
    out, attn = mha(x, mask=mask)

    assert out.shape == (B, T, D)
    assert attn.shape == (B, H, T, T)

    probs_last_col = attn.numpy()[..., -1]
    assert np.allclose(probs_last_col, 0.0, atol=1e-6)


def smoke_mha_eval_mode():
    B, T, D, H = 1, 4, 8, 2

    x = Tensor(
        np.arange(B * T * D, dtype=np.float64).reshape(B, T, D),
        requires_grad=True,
    )

    mha = MultiHeadAttention(d_model=D, num_heads=H, dropout_p=0.1)
    mha.eval()

    out, attn = mha(x)

    assert out.shape == (B, T, D)
    assert attn.shape == (B, H, T, T)
    assert mha.training is False


def main():
    check("mha forward", smoke_mha_forward)
    check("mha backward", smoke_mha_backward)
    check("mha with mask", smoke_mha_with_mask)
    check("mha eval mode", smoke_mha_eval_mode)


if __name__ == "__main__":
    main()