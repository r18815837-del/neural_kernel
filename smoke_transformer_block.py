import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.modules.transformer import TransformerBlock


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_transformer_forward():
    B, T, D, H, FF = 2, 5, 8, 2, 16

    x = Tensor(
        np.arange(B * T * D, dtype=np.float64).reshape(B, T, D),
        requires_grad=True,
    )

    block = TransformerBlock(
        d_model=D,
        num_heads=H,
        d_ff=FF,
        dropout_p=0.0,
    )

    out, attn = block(x)

    assert out.shape == (B, T, D)
    assert attn.shape == (B, H, T, T)

    sums = attn.numpy().sum(axis=-1)
    assert np.allclose(sums, np.ones((B, H, T)))


def smoke_transformer_backward():
    B, T, D, H, FF = 2, 4, 8, 2, 16

    x = Tensor(
        np.arange(B * T * D, dtype=np.float64).reshape(B, T, D),
        requires_grad=True,
    )

    block = TransformerBlock(
        d_model=D,
        num_heads=H,
        d_ff=FF,
        dropout_p=0.0,
    )

    out, attn = block(x)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape

    assert block.attn.q_proj.weight.grad is not None
    assert block.attn.k_proj.weight.grad is not None
    assert block.attn.v_proj.weight.grad is not None
    assert block.attn.out_proj.weight.grad is not None

    assert block.ffn.fc1.weight.grad is not None
    assert block.ffn.fc2.weight.grad is not None


def smoke_transformer_with_mask():
    B, T, D, H, FF = 1, 4, 8, 2, 16

    x = Tensor(
        np.arange(B * T * D, dtype=np.float64).reshape(B, T, D),
        requires_grad=True,
    )

    mask_np = np.zeros((B, 1, 1, T), dtype=np.float64)
    mask_np[..., -1] = -1e9
    mask = Tensor(mask_np, requires_grad=False)

    block = TransformerBlock(
        d_model=D,
        num_heads=H,
        d_ff=FF,
        dropout_p=0.0,
    )

    out, attn = block(x, mask=mask)

    assert out.shape == (B, T, D)
    assert attn.shape == (B, H, T, T)

    probs_last_col = attn.numpy()[..., -1]
    assert np.allclose(probs_last_col, 0.0, atol=1e-6)


def smoke_transformer_eval_mode():
    B, T, D, H, FF = 1, 4, 8, 2, 16

    x = Tensor(
        np.arange(B * T * D, dtype=np.float64).reshape(B, T, D),
        requires_grad=True,
    )

    block = TransformerBlock(
        d_model=D,
        num_heads=H,
        d_ff=FF,
        dropout_p=0.1,
    )
    block.eval()

    out, attn = block(x)

    assert out.shape == (B, T, D)
    assert attn.shape == (B, H, T, T)
    assert block.training is False


def main():
    check("transformer forward", smoke_transformer_forward)
    check("transformer backward", smoke_transformer_backward)
    check("transformer with mask", smoke_transformer_with_mask)
    check("transformer eval mode", smoke_transformer_eval_mode)


if __name__ == "__main__":
    main()