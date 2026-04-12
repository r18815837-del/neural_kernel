import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.modules.classifier import TransformerEncoderClassifier
from kernel.nn.modules.token_lm import TokenTransformerLM


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_classifier_checkpoint_roundtrip():
    np.random.seed(42)

    model_a = TransformerEncoderClassifier(
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        num_classes=3,
        dropout_p=0.0,
        max_len=32,
        use_positional_encoding=True,
        activation="gelu",
        pooling="cls",
    )
    model_a.eval()

    x = Tensor(
        np.arange(2 * 5 * 8, dtype=np.float64).reshape(2, 5, 8),
        requires_grad=False,
    )

    logits_a, attn_a = model_a(x)
    state = model_a.state_dict()

    model_b = TransformerEncoderClassifier(
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        num_classes=3,
        dropout_p=0.0,
        max_len=32,
        use_positional_encoding=True,
        activation="gelu",
        pooling="cls",
    )
    model_b.load_state_dict(state)
    model_b.eval()

    logits_b, attn_b = model_b(x)

    assert logits_a.shape == logits_b.shape
    assert np.allclose(logits_a.numpy(), logits_b.numpy(), atol=1e-8)

    assert len(attn_a) == len(attn_b)
    for wa, wb in zip(attn_a, attn_b):
        assert wa.shape == wb.shape
        assert np.allclose(wa.numpy(), wb.numpy(), atol=1e-8)


def smoke_token_lm_checkpoint_roundtrip():
    np.random.seed(42)

    model_a = TokenTransformerLM(
        vocab_size=20,
        d_model=16,
        num_heads=4,
        d_ff=32,
        num_layers=2,
        dropout_p=0.0,
        max_len=32,
        activation="gelu",
    )
    model_a.eval()

    token_ids = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int64)

    logits_a, attn_a = model_a(token_ids, use_causal_mask=True)
    state = model_a.state_dict()

    model_b = TokenTransformerLM(
        vocab_size=20,
        d_model=16,
        num_heads=4,
        d_ff=32,
        num_layers=2,
        dropout_p=0.0,
        max_len=32,
        activation="gelu",
    )
    model_b.load_state_dict(state)
    model_b.eval()

    logits_b, attn_b = model_b(token_ids, use_causal_mask=True)

    assert logits_a.shape == logits_b.shape
    assert np.allclose(logits_a.numpy(), logits_b.numpy(), atol=1e-8)

    assert len(attn_a) == len(attn_b)
    for wa, wb in zip(attn_a, attn_b):
        assert wa.shape == wb.shape
        assert np.allclose(wa.numpy(), wb.numpy(), atol=1e-8)


def smoke_classifier_state_dict_keys_nonempty():
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
        pooling="mean",
    )

    state = model.state_dict()
    assert isinstance(state, dict)
    assert len(state) > 0


def smoke_token_lm_state_dict_keys_nonempty():
    model = TokenTransformerLM(
        vocab_size=20,
        d_model=16,
        num_heads=4,
        d_ff=32,
        num_layers=2,
        dropout_p=0.0,
        max_len=32,
        activation="gelu",
    )

    state = model.state_dict()
    assert isinstance(state, dict)
    assert len(state) > 0


def main():
    check("classifier checkpoint roundtrip", smoke_classifier_checkpoint_roundtrip)
    check("token lm checkpoint roundtrip", smoke_token_lm_checkpoint_roundtrip)
    check("classifier state_dict nonempty", smoke_classifier_state_dict_keys_nonempty)
    check("token lm state_dict nonempty", smoke_token_lm_state_dict_keys_nonempty)


if __name__ == "__main__":
    main()