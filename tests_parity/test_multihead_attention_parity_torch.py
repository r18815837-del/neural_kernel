import math

import numpy as np
import torch

from kernel import Tensor
from kernel.nn.functional import make_causal_mask
from kernel.nn.modules import MultiHeadAttention


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


def copy_linear_like(kernel_linear, torch_weight, torch_bias):
    """
    kernel Linear stores:
        weight: (in_features, out_features)
        bias:   (1, out_features)

    torch Linear/reference uses:
        weight: (out_features, in_features)
        bias:   (out_features,)
    """
    kw = to_numpy(kernel_linear.weight)
    kb = to_numpy(kernel_linear.bias) if getattr(kernel_linear, "bias", None) is not None else None

    expected_kw = torch_weight.T
    if kw.shape != expected_kw.shape:
        raise AssertionError(
            f"Unexpected kernel weight shape: kernel {kw.shape}, expected {expected_kw.shape}"
        )
    kernel_linear.weight.data = expected_kw.copy()

    if kb is not None and torch_bias is not None:
        expected_kb = torch_bias.reshape(1, -1)
        if kb.shape != expected_kb.shape:
            raise AssertionError(
                f"Unexpected kernel bias shape: kernel {kb.shape}, expected {expected_kb.shape}"
            )
        kernel_linear.bias.data = expected_kb.copy()

def assign_kernel_mha_weights_from_reference(
    kernel_mha,
    w_q,
    b_q,
    w_k,
    b_k,
    w_v,
    b_v,
    w_o,
    b_o,
):
    """
    Assumes kernel MHA has q_proj, k_proj, v_proj, out_proj style submodules.
    If your names differ, adapt only this function.
    """
    copy_linear_like(kernel_mha.q_proj, w_q, b_q)
    copy_linear_like(kernel_mha.k_proj, w_k, b_k)
    copy_linear_like(kernel_mha.v_proj, w_v, b_v)
    copy_linear_like(kernel_mha.out_proj, w_o, b_o)


def torch_mha_reference(x, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o, num_heads, mask=None):
    """
    x: (B, T, D)
    All weights use torch Linear convention: (out, in)
    """
    bsz, seq_len, d_model = x.shape
    head_dim = d_model // num_heads

    q = torch.nn.functional.linear(x, w_q, b_q)
    k = torch.nn.functional.linear(x, w_k, b_k)
    v = torch.nn.functional.linear(x, w_v, b_v)

    q = q.view(bsz, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    k = k.view(bsz, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    v = v.view(bsz, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    if mask is not None:
        scores = scores + mask

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)

    out = out.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, d_model)
    out = torch.nn.functional.linear(out, w_o, b_o)
    return out


def make_reference_weights(d_model):
    rng = np.random.default_rng(42)

    def w():
        return rng.standard_normal((d_model, d_model), dtype=np.float32)

    def b():
        return rng.standard_normal((d_model,), dtype=np.float32)

    return w(), b(), w(), b(), w(), b(), w(), b()


def test_multihead_attention_forward_parity_with_torch():
    np.random.seed(42)
    torch.manual_seed(42)

    d_model = 8
    num_heads = 2
    x_np = np.random.randn(2, 4, d_model).astype(np.float32)

    w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o = make_reference_weights(d_model)

    x_torch = torch.tensor(x_np, dtype=torch.float32)
    out_torch = torch_mha_reference(
        x_torch,
        torch.tensor(w_q),
        torch.tensor(b_q),
        torch.tensor(w_k),
        torch.tensor(b_k),
        torch.tensor(w_v),
        torch.tensor(b_v),
        torch.tensor(w_o),
        torch.tensor(b_o),
        num_heads=num_heads,
        mask=None,
    ).detach().cpu().numpy()

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    assign_kernel_mha_weights_from_reference(
        mha, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o
    )

    x_kernel = Tensor(x_np, requires_grad=False)
    out_kernel = mha(x_kernel)
    out_kernel = maybe_first(out_kernel)
    out_kernel_np = to_numpy(out_kernel)

    assert out_kernel_np.shape == out_torch.shape
    assert np.allclose(out_kernel_np, out_torch, atol=1e-5, rtol=1e-5)


def test_multihead_attention_forward_parity_with_torch_causal_mask():
    np.random.seed(42)
    torch.manual_seed(42)

    d_model = 8
    num_heads = 2
    x_np = np.random.randn(2, 4, d_model).astype(np.float32)

    w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o = make_reference_weights(d_model)

    mask_kernel = make_causal_mask(4)
    mask_np = to_numpy(mask_kernel)
    mask_torch = torch.tensor(mask_np, dtype=torch.float32)

    x_torch = torch.tensor(x_np, dtype=torch.float32)
    out_torch = torch_mha_reference(
        x_torch,
        torch.tensor(w_q),
        torch.tensor(b_q),
        torch.tensor(w_k),
        torch.tensor(b_k),
        torch.tensor(w_v),
        torch.tensor(b_v),
        torch.tensor(w_o),
        torch.tensor(b_o),
        num_heads=num_heads,
        mask=mask_torch,
    ).detach().cpu().numpy()

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    assign_kernel_mha_weights_from_reference(
        mha, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o
    )

    x_kernel = Tensor(x_np, requires_grad=False)
    out_kernel = mha(x_kernel, mask_kernel)
    out_kernel = maybe_first(out_kernel)
    out_kernel_np = to_numpy(out_kernel)

    assert out_kernel_np.shape == out_torch.shape
    assert np.allclose(out_kernel_np, out_torch, atol=1e-5, rtol=1e-5)


def test_multihead_attention_backward_input_grad_parity_with_torch():
    np.random.seed(42)
    torch.manual_seed(42)

    d_model = 8
    num_heads = 2
    x_np = np.random.randn(2, 4, d_model).astype(np.float32)

    w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o = make_reference_weights(d_model)

    x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    out_torch = torch_mha_reference(
        x_torch,
        torch.tensor(w_q),
        torch.tensor(b_q),
        torch.tensor(w_k),
        torch.tensor(b_k),
        torch.tensor(w_v),
        torch.tensor(b_v),
        torch.tensor(w_o),
        torch.tensor(b_o),
        num_heads=num_heads,
        mask=None,
    )
    loss_torch = out_torch.sum()
    loss_torch.backward()
    torch_x_grad = x_torch.grad.detach().cpu().numpy()

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    assign_kernel_mha_weights_from_reference(
        mha, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o
    )

    x_kernel = Tensor(x_np, requires_grad=True)
    out_kernel = mha(x_kernel)
    out_kernel = maybe_first(out_kernel)
    loss_kernel = out_kernel.sum()
    loss_kernel.backward()
    kernel_x_grad = get_grad(x_kernel)

    assert kernel_x_grad is not None
    assert kernel_x_grad.shape == torch_x_grad.shape
    assert np.allclose(kernel_x_grad, torch_x_grad, atol=1e-5, rtol=1e-5)


def test_multihead_attention_preserves_shape():
    mha = MultiHeadAttention(d_model=12, num_heads=3)

    x = Tensor(np.random.randn(3, 5, 12).astype(np.float32))
    out = mha(x)
    out = maybe_first(out)
    arr = to_numpy(out)

    assert arr.shape == (3, 5, 12)
    assert np.isfinite(arr).all()