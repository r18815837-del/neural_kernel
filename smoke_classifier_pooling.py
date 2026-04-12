import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.modules.classifier import TransformerEncoderClassifier


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def make_input(B=2, T=5, D=8):
    return Tensor(
        np.arange(B * T * D, dtype=np.float64).reshape(B, T, D),
        requires_grad=True,
    )


def smoke_classifier_mean_pool():
    x = make_input()
    model = TransformerEncoderClassifier(
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        num_classes=3,
        pooling="mean",
    )

    logits, attn_all = model(x)
    assert logits.shape == (2, 3)
    assert len(attn_all) == 2


def smoke_classifier_cls_pool():
    x = make_input()
    model = TransformerEncoderClassifier(
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        num_classes=3,
        pooling="cls",
    )

    logits, attn_all = model(x)
    assert logits.shape == (2, 3)
    assert len(attn_all) == 2


def smoke_classifier_last_pool():
    x = make_input()
    model = TransformerEncoderClassifier(
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        num_classes=3,
        pooling="last",
    )

    logits, attn_all = model(x)
    assert logits.shape == (2, 3)
    assert len(attn_all) == 2


def smoke_classifier_cls_backward():
    x = make_input()
    model = TransformerEncoderClassifier(
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        num_classes=3,
        pooling="cls",
    )

    logits, _ = model(x)
    loss = logits.mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert model.classifier.weight.grad is not None


def main():
    check("classifier mean pool", smoke_classifier_mean_pool)
    check("classifier cls pool", smoke_classifier_cls_pool)
    check("classifier last pool", smoke_classifier_last_pool)
    check("classifier cls backward", smoke_classifier_cls_backward)


if __name__ == "__main__":
    main()