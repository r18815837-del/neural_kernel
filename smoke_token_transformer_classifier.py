import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.functional.masks import make_padding_mask
from kernel.nn.modules.token_classifier import TokenTransformerClassifier


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_token_classifier_forward():
    token_ids = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int64)

    model = TokenTransformerClassifier(
        vocab_size=10,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        num_classes=3,
        dropout_p=0.0,
        max_len=16,
        activation="gelu",
        pooling="mean",
    )

    logits, attn_all = model(token_ids)
    assert logits.shape == (2, 3)
    assert len(attn_all) == 2


def smoke_token_classifier_backward():
    token_ids = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int64)

    model = TokenTransformerClassifier(
        vocab_size=10,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        num_classes=3,
        dropout_p=0.0,
        max_len=16,
        activation="gelu",
        pooling="cls",
    )

    logits, _ = model(token_ids)
    loss = logits.mean()
    loss.backward()

    assert model.embedding.weight.grad is not None
    assert model.classifier.weight.grad is not None


def smoke_token_classifier_with_padding_mask():
    token_ids = np.array([[1, 2, 3, 4], [4, 3, 0, 0]], dtype=np.int64)
    mask = make_padding_mask([4, 2], max_len=4)

    model = TokenTransformerClassifier(
        vocab_size=10,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        num_classes=3,
        dropout_p=0.0,
        max_len=16,
        activation="gelu",
        pooling="last",
    )

    logits, attn_all = model(token_ids, mask=mask)
    assert logits.shape == (2, 3)
    assert len(attn_all) == 2


def main():
    check("token classifier forward", smoke_token_classifier_forward)
    check("token classifier backward", smoke_token_classifier_backward)
    check("token classifier with padding mask", smoke_token_classifier_with_padding_mask)


if __name__ == "__main__":
    main()