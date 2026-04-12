import numpy as np

from kernel import Tensor
from kernel.nn.functional import make_causal_mask
from kernel.nn.modules import TransformerBlock


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


def test_transformer_block_preserves_shape():
    block = TransformerBlock(d_model=8, num_heads=2, d_ff=16)

    x = Tensor(np.random.randn(2, 4, 8).astype(np.float32))
    out = block(x)
    out = maybe_first(out)
    arr = to_numpy(out)

    assert arr.shape == (2, 4, 8)
    assert np.isfinite(arr).all()


def test_transformer_block_with_mask_preserves_shape():
    block = TransformerBlock(d_model=8, num_heads=2, d_ff=16)

    x = Tensor(np.random.randn(2, 4, 8).astype(np.float32))
    mask = make_causal_mask(4)

    out = block(x, mask)
    out = maybe_first(out)
    arr = to_numpy(out)

    assert arr.shape == (2, 4, 8)
    assert np.isfinite(arr).all()


def test_transformer_block_backward_produces_input_grad():
    block = TransformerBlock(d_model=8, num_heads=2, d_ff=16)

    x = Tensor(np.random.randn(2, 4, 8).astype(np.float32), requires_grad=True)
    out = block(x)
    out = maybe_first(out)
    loss = out.sum()
    loss.backward()

    grad = get_grad(x)

    assert grad is not None
    assert grad.shape == (2, 4, 8)
    assert np.isfinite(grad).all()


def test_transformer_block_preserves_batch_seq_hidden_dims():
    block = TransformerBlock(d_model=12, num_heads=3, d_ff=24)

    x = Tensor(np.random.randn(3, 5, 12).astype(np.float32))
    out = block(x)
    out = maybe_first(out)
    arr = to_numpy(out)

    assert arr.shape[0] == 3
    assert arr.shape[1] == 5
    assert arr.shape[2] == 12


def test_transformer_block_outputs_finite_values():
    block = TransformerBlock(d_model=8, num_heads=2, d_ff=16)

    x = Tensor(np.random.randn(2, 4, 8).astype(np.float32))
    out = block(x)
    out = maybe_first(out)
    arr = to_numpy(out)

    assert np.isfinite(arr).all()