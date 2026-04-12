import numpy as np

from kernel import Tensor
from kernel.nn.modules import (
    TransformerEncoderClassifier,
    TokenTransformerClassifier,
    TokenTransformerLM,
)


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "data"):
        data = x.data
        if isinstance(data, np.ndarray):
            return data
        try:
            return np.array(data)
        except Exception:
            pass
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.array(x)


def get_grad(x):
    grad = getattr(x, "grad", None)
    if grad is None:
        return None
    return to_numpy(grad)


def maybe_first(x):
    if isinstance(x, tuple):
        return x[0]
    return x


def test_transformer_encoder_classifier_output_shape():
    model = TransformerEncoderClassifier(
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        num_classes=3,
    )

    x = Tensor(np.random.randn(2, 4, 8).astype(np.float32))
    out = model(x)
    out = maybe_first(out)
    arr = to_numpy(out)

    assert arr.shape == (2, 3)
    assert np.isfinite(arr).all()


def test_transformer_encoder_classifier_backward():
    model = TransformerEncoderClassifier(
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        num_classes=3,
    )

    x = Tensor(np.random.randn(2, 4, 8).astype(np.float32), requires_grad=True)
    out = model(x)
    out = maybe_first(out)
    loss = out.sum()
    loss.backward()

    grad = get_grad(x)

    assert grad is not None
    assert grad.shape == (2, 4, 8)
    assert np.isfinite(grad).all()


def test_token_transformer_classifier_output_shape():
    model = TokenTransformerClassifier(
        vocab_size=20,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        num_classes=4,
        max_len=8,
    )

    tokens = Tensor(np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int64))
    out = model(tokens)
    out = maybe_first(out)
    arr = to_numpy(out)

    assert arr.shape == (2, 4)
    assert np.isfinite(arr).all()


def test_token_transformer_lm_output_shape():
    model = TokenTransformerLM(
        vocab_size=30,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        max_len=8,
    )

    tokens = Tensor(np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int64))
    out = model(tokens)
    out = maybe_first(out)
    arr = to_numpy(out)

    assert arr.shape == (2, 4, 30)
    assert np.isfinite(arr).all()


def test_token_transformer_lm_backward():
    model = TokenTransformerLM(
        vocab_size=30,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        max_len=8,
    )

    tokens = Tensor(np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int64))
    out = model(tokens)
    out = maybe_first(out)
    loss = out.sum()
    loss.backward()

    assert np.isfinite(to_numpy(loss)).all()


def test_token_transformer_classifier_outputs_finite_values():
    model = TokenTransformerClassifier(
        vocab_size=20,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        num_classes=4,
        max_len=8,
    )

    tokens = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.int64))
    out = model(tokens)
    out = maybe_first(out)
    arr = to_numpy(out)

    assert np.isfinite(arr).all()