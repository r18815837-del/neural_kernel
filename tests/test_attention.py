import numpy as np

from kernel import Tensor
from kernel.nn.functional import scaled_dot_product_attention, make_causal_mask
from kernel.nn.modules import ScaledDotProductAttention, MultiHeadAttention


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


def test_functional_scaled_dot_product_attention_shape():
    q = Tensor(np.random.randn(2, 1, 4, 8).astype(np.float32))
    k = Tensor(np.random.randn(2, 1, 4, 8).astype(np.float32))
    v = Tensor(np.random.randn(2, 1, 4, 8).astype(np.float32))

    out = scaled_dot_product_attention(q, k, v)
    out = maybe_first(out)
    arr = to_numpy(out)

    assert arr.shape == (2, 1, 4, 8)
    assert np.isfinite(arr).all()


def test_functional_scaled_dot_product_attention_with_causal_mask():
    q = Tensor(np.random.randn(2, 1, 4, 8).astype(np.float32))
    k = Tensor(np.random.randn(2, 1, 4, 8).astype(np.float32))
    v = Tensor(np.random.randn(2, 1, 4, 8).astype(np.float32))
    mask = make_causal_mask(4)

    out = scaled_dot_product_attention(q, k, v, mask=mask)
    out = maybe_first(out)
    arr = to_numpy(out)

    assert arr.shape == (2, 1, 4, 8)
    assert np.isfinite(arr).all()


def test_scaled_dot_product_attention_module_shape():
    attn = ScaledDotProductAttention()

    q = Tensor(np.random.randn(2, 1, 4, 8).astype(np.float32))
    k = Tensor(np.random.randn(2, 1, 4, 8).astype(np.float32))
    v = Tensor(np.random.randn(2, 1, 4, 8).astype(np.float32))

    out = attn(q, k, v)
    out = maybe_first(out)
    arr = to_numpy(out)

    assert arr.shape == (2, 1, 4, 8)
    assert np.isfinite(arr).all()


def test_scaled_dot_product_attention_module_with_mask():
    attn = ScaledDotProductAttention()

    q = Tensor(np.random.randn(2, 1, 4, 8).astype(np.float32))
    k = Tensor(np.random.randn(2, 1, 4, 8).astype(np.float32))
    v = Tensor(np.random.randn(2, 1, 4, 8).astype(np.float32))
    mask = make_causal_mask(4)

    out = attn(q, k, v, mask=mask)
    out = maybe_first(out)
    arr = to_numpy(out)

    assert arr.shape == (2, 1, 4, 8)
    assert np.isfinite(arr).all()
def test_multihead_attention_shape():
    mha = MultiHeadAttention(d_model=8, num_heads=2)

    x = Tensor(np.random.randn(2, 4, 8).astype(np.float32))
    out = mha(x)
    out = maybe_first(out)
    arr = to_numpy(out)

    assert arr.shape == (2, 4, 8)
    assert np.isfinite(arr).all()


def test_multihead_attention_with_mask_shape():
    mha = MultiHeadAttention(d_model=8, num_heads=2)

    x = Tensor(np.random.randn(2, 4, 8).astype(np.float32))
    mask = make_causal_mask(4)

    out = mha(x, mask)
    out = maybe_first(out)
    arr = to_numpy(out)

    assert arr.shape == (2, 4, 8)
    assert np.isfinite(arr).all()


def test_multihead_attention_backward_produces_input_grad():
    mha = MultiHeadAttention(d_model=8, num_heads=2)

    x = Tensor(np.random.randn(2, 4, 8).astype(np.float32), requires_grad=True)
    out = mha(x)
    out = maybe_first(out)
    loss = out.sum()
    loss.backward()

    grad = get_grad(x)

    assert grad is not None
    assert grad.shape == (2, 4, 8)
    assert np.isfinite(grad).all()


def test_multihead_attention_preserves_batch_and_seq_dims():
    mha = MultiHeadAttention(d_model=12, num_heads=3)

    x = Tensor(np.random.randn(3, 5, 12).astype(np.float32))
    out = mha(x)
    out = maybe_first(out)
    arr = to_numpy(out)

    assert arr.shape == (3, 5, 12)