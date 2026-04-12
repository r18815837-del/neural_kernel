import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.functional.masks import make_causal_mask
from kernel.nn.modules.encoder import TransformerEncoder


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_encoder_forward():
    B, T, D = 2, 5, 8

    x = Tensor(
        np.arange(B * T * D, dtype=np.float64).reshape(B, T, D),
        requires_grad=True,
    )

    encoder = TransformerEncoder(
        d_model=D,
        num_heads=2,
        d_ff=16,
        num_layers=3,
        dropout_p=0.0,
        max_len=32,
        use_positional_encoding=True,
    )

    out, attn_all = encoder(x)

    assert out.shape == (B, T, D)
    assert len(attn_all) == 3
    for attn in attn_all:
        assert attn.shape == (B, 2, T, T)


def smoke_encoder_backward():
    B, T, D = 2, 4, 8

    x = Tensor(
        np.arange(B * T * D, dtype=np.float64).reshape(B, T, D),
        requires_grad=True,
    )

    encoder = TransformerEncoder(
        d_model=D,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout_p=0.0,
        max_len=32,
        use_positional_encoding=True,
    )

    out, attn_all = encoder(x)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def smoke_encoder_with_mask():
    B, T, D = 1, 4, 8

    x = Tensor(
        np.arange(B * T * D, dtype=np.float64).reshape(B, T, D),
        requires_grad=True,
    )

    mask = make_causal_mask(T, device=x.device)

    encoder = TransformerEncoder(
        d_model=D,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout_p=0.0,
        max_len=32,
        use_positional_encoding=True,
    )

    out, attn_all = encoder(x, mask=mask)

    assert out.shape == (B, T, D)
    assert len(attn_all) == 2

    for attn in attn_all:
        assert attn.shape == (B, 2, T, T)


def smoke_encoder_eval_mode():
    B, T, D = 1, 4, 8

    x = Tensor(
        np.arange(B * T * D, dtype=np.float64).reshape(B, T, D),
        requires_grad=True,
    )

    encoder = TransformerEncoder(
        d_model=D,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout_p=0.1,
        max_len=32,
        use_positional_encoding=True,
    )
    encoder.eval()

    out, attn_all = encoder(x)

    assert out.shape == (B, T, D)
    assert encoder.training is False


def main():
    check("encoder forward", smoke_encoder_forward)
    check("encoder backward", smoke_encoder_backward)
    check("encoder with mask", smoke_encoder_with_mask)
    check("encoder eval mode", smoke_encoder_eval_mode)


if __name__ == "__main__":
    main()