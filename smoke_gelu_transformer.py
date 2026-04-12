import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.modules.transformer import FeedForward, TransformerBlock
from kernel.nn.modules.encoder import TransformerEncoder
from kernel.nn.modules.classifier import TransformerEncoderClassifier


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_gelu_tensor():
    x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    y = x.gelu()

    assert y.shape == (3,)
    loss = y.mean()
    loss.backward()
    assert x.grad is not None


def smoke_ffn_gelu():
    x = Tensor(np.arange(2 * 4 * 8, dtype=np.float64).reshape(2, 4, 8), requires_grad=True)
    ffn = FeedForward(d_model=8, d_ff=16, dropout_p=0.0, activation="gelu")

    y = ffn(x)
    assert y.shape == (2, 4, 8)

    loss = y.mean()
    loss.backward()
    assert x.grad is not None


def smoke_transformer_block_gelu():
    x = Tensor(np.arange(2 * 4 * 8, dtype=np.float64).reshape(2, 4, 8), requires_grad=True)
    block = TransformerBlock(d_model=8, num_heads=2, d_ff=16, dropout_p=0.0, activation="gelu")

    y, attn = block(x)
    assert y.shape == (2, 4, 8)
    assert attn.shape == (2, 2, 4, 4)

    loss = y.mean()
    loss.backward()
    assert x.grad is not None


def smoke_encoder_gelu():
    x = Tensor(np.arange(2 * 5 * 8, dtype=np.float64).reshape(2, 5, 8), requires_grad=True)
    enc = TransformerEncoder(
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout_p=0.0,
        max_len=32,
        use_positional_encoding=True,
        activation="gelu",
    )

    y, attn_all = enc(x)
    assert y.shape == (2, 5, 8)
    assert len(attn_all) == 2


def smoke_classifier_gelu():
    x = Tensor(np.arange(2 * 5 * 8, dtype=np.float64).reshape(2, 5, 8), requires_grad=True)
    model = TransformerEncoderClassifier(
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        num_classes=3,
        dropout_p=0.0,
        max_len=32,
        use_positional_encoding=True,
        activation="gelu",
    )

    logits, attn_all = model(x)
    assert logits.shape == (2, 3)
    assert len(attn_all) == 2

    loss = logits.mean()
    loss.backward()
    assert x.grad is not None
    assert model.classifier.weight.grad is not None


def main():
    check("gelu tensor", smoke_gelu_tensor)
    check("ffn gelu", smoke_ffn_gelu)
    check("transformer block gelu", smoke_transformer_block_gelu)
    check("encoder gelu", smoke_encoder_gelu)
    check("classifier gelu", smoke_classifier_gelu)


if __name__ == "__main__":
    main()