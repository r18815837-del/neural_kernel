import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.functional.masks import make_causal_mask
from kernel.nn.modules.classifier import TransformerEncoderClassifier


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_classifier_forward():
    B, T, D, C = 2, 5, 8, 3

    x = Tensor(
        np.arange(B * T * D, dtype=np.float64).reshape(B, T, D),
        requires_grad=True,
    )

    model = TransformerEncoderClassifier(
        d_model=D,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        num_classes=C,
        dropout_p=0.0,
        max_len=32,
        use_positional_encoding=True,
    )

    logits, attn_all = model(x)

    assert logits.shape == (B, C)
    assert len(attn_all) == 2


def smoke_classifier_backward():
    B, T, D, C = 2, 4, 8, 3

    x = Tensor(
        np.arange(B * T * D, dtype=np.float64).reshape(B, T, D),
        requires_grad=True,
    )

    model = TransformerEncoderClassifier(
        d_model=D,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        num_classes=C,
        dropout_p=0.0,
        max_len=32,
        use_positional_encoding=True,
    )

    logits, attn_all = model(x)
    loss = logits.mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert model.classifier.weight.grad is not None
    assert model.encoder.layers[0].attn.q_proj.weight.grad is not None


def smoke_classifier_with_mask():
    B, T, D, C = 1, 4, 8, 3

    x = Tensor(
        np.arange(B * T * D, dtype=np.float64).reshape(B, T, D),
        requires_grad=True,
    )

    mask = make_causal_mask(T, device=x.device)

    model = TransformerEncoderClassifier(
        d_model=D,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        num_classes=C,
        dropout_p=0.0,
        max_len=32,
        use_positional_encoding=True,
    )

    logits, attn_all = model(x, mask=mask)

    assert logits.shape == (B, C)
    assert len(attn_all) == 2


def smoke_classifier_eval_mode():
    B, T, D, C = 1, 4, 8, 3

    x = Tensor(
        np.arange(B * T * D, dtype=np.float64).reshape(B, T, D),
        requires_grad=True,
    )

    model = TransformerEncoderClassifier(
        d_model=D,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        num_classes=C,
        dropout_p=0.1,
        max_len=32,
        use_positional_encoding=True,
    )
    model.eval()

    logits, attn_all = model(x)

    assert logits.shape == (B, C)
    assert model.training is False


def main():
    check("classifier forward", smoke_classifier_forward)
    check("classifier backward", smoke_classifier_backward)
    check("classifier with mask", smoke_classifier_with_mask)
    check("classifier eval mode", smoke_classifier_eval_mode)


if __name__ == "__main__":
    main()