import numpy as np

from kernel import Tensor
from kernel.nn.functional import make_causal_mask
from kernel.nn.modules import TransformerEncoder


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


def test_transformer_encoder_preserves_shape():
    encoder = TransformerEncoder(
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
    )

    x = Tensor(np.random.randn(2, 4, 8).astype(np.float32))
    out = encoder(x)
    out = maybe_first(out)
    arr = to_numpy(out)

    assert arr.shape == (2, 4, 8)
    assert np.isfinite(arr).all()


def test_transformer_encoder_with_mask_preserves_shape():
    encoder = TransformerEncoder(
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
    )

    x = Tensor(np.random.randn(2, 4, 8).astype(np.float32))
    mask = make_causal_mask(4)

    out = encoder(x, mask)
    out = maybe_first(out)
    arr = to_numpy(out)

    assert arr.shape == (2, 4, 8)
    assert np.isfinite(arr).all()


def test_transformer_encoder_backward_produces_input_grad():
    encoder = TransformerEncoder(
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
    )

    x = Tensor(np.random.randn(2, 4, 8).astype(np.float32), requires_grad=True)
    out = encoder(x)
    out = maybe_first(out)
    loss = out.sum()
    loss.backward()

    grad = get_grad(x)

    assert grad is not None
    assert grad.shape == (2, 4, 8)
    assert np.isfinite(grad).all()


def test_transformer_encoder_multiple_layers_preserve_batch_seq_hidden():
    encoder = TransformerEncoder(
        d_model=12,
        num_heads=3,
        d_ff=24,
        num_layers=3,
    )

    x = Tensor(np.random.randn(3, 5, 12).astype(np.float32))
    out = encoder(x)
    out = maybe_first(out)
    arr = to_numpy(out)

    assert arr.shape[0] == 3
    assert arr.shape[1] == 5
    assert arr.shape[2] == 12


def test_transformer_encoder_outputs_finite_values():
    encoder = TransformerEncoder(
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
    )

    x = Tensor(np.random.randn(2, 4, 8).astype(np.float32))
    out = encoder(x)
    out = maybe_first(out)
    arr = to_numpy(out)

    assert np.isfinite(arr).all()