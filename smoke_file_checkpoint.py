import os
import tempfile

import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.modules.classifier import TransformerEncoderClassifier
from kernel.nn.modules.token_lm import TokenTransformerLM
from kernel.utils.checkpoint import save_checkpoint, load_checkpoint


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_classifier_file_checkpoint():
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
    logits_a, _ = model_a(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "classifier_checkpoint.npz")
        save_checkpoint(model_a, path)

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
        load_checkpoint(model_b, path)
        model_b.eval()

        logits_b, _ = model_b(x)

    assert logits_a.shape == logits_b.shape
    assert np.allclose(logits_a.numpy(), logits_b.numpy(), atol=1e-8)


def smoke_token_lm_file_checkpoint():
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
    logits_a, _ = model_a(token_ids, use_causal_mask=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "token_lm_checkpoint.npz")
        save_checkpoint(model_a, path)

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
        load_checkpoint(model_b, path)
        model_b.eval()

        logits_b, _ = model_b(token_ids, use_causal_mask=True)

    assert logits_a.shape == logits_b.shape
    assert np.allclose(logits_a.numpy(), logits_b.numpy(), atol=1e-8)


def smoke_missing_checkpoint():
    np.random.seed(42)

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

    failed = False
    try:
        load_checkpoint(model, "definitely_missing_checkpoint_123456.npz")
    except FileNotFoundError:
        failed = True

    assert failed, "Expected FileNotFoundError for missing checkpoint"


def main():
    check("classifier file checkpoint", smoke_classifier_file_checkpoint)
    check("token lm file checkpoint", smoke_token_lm_file_checkpoint)
    check("missing checkpoint", smoke_missing_checkpoint)


if __name__ == "__main__":
    main()