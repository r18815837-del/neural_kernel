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


def smoke_cls_forward():
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
        use_cls_token=True,
    )

    logits, attn_all = model(token_ids)

    assert logits.shape == (2, 3)
    assert len(attn_all) == 2
    for attn in attn_all:
        assert attn.shape == (2, 2, 5, 5)  # T=4 -> T+1=5


def smoke_cls_backward():
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
        use_cls_token=True,
    )

    logits, _ = model(token_ids)
    loss = logits.mean()
    loss.backward()

    assert model.embedding.weight.grad is not None
    assert model.cls_token.grad is not None
    assert model.classifier.weight.grad is not None


def smoke_cls_with_padding_mask():
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
        pooling="cls",
        use_cls_token=True,
    )

    logits, attn_all = model(token_ids, mask=mask)

    assert logits.shape == (2, 3)
    assert len(attn_all) == 2
    for attn in attn_all:
        assert attn.shape == (2, 2, 5, 5)


def smoke_invalid_cls_config():
    failed = False
    try:
        _ = TokenTransformerClassifier(
            vocab_size=10,
            d_model=8,
            num_heads=2,
            d_ff=16,
            num_layers=2,
            num_classes=3,
            pooling="mean",
            use_cls_token=True,
        )
    except ValueError:
        failed = True

    assert failed, "Expected ValueError for invalid cls token config"


def main():
    check("cls forward", smoke_cls_forward)
    check("cls backward", smoke_cls_backward)
    check("cls with padding mask", smoke_cls_with_padding_mask)
    check("invalid cls config", smoke_invalid_cls_config)


if __name__ == "__main__":
    main()