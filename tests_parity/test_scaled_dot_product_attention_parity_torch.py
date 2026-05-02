import math

import numpy as np
import torch

from kernel import Tensor
from kernel.nn.functional import scaled_dot_product_attention, make_causal_mask
from kernel.nn.modules import ScaledDotProductAttention


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


def torch_scaled_dot_product_reference(q, k, v, mask=None):
    """
    Reference implementation using raw PyTorch ops so behavior is explicit.
    q, k, v: (B, H, T, D)
    mask: broadcastable to (B, H, T, T), additive mask with 0 / -1e9 style values
    """
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out


def test_functional_scaled_dot_product_attention_forward_parity():
    np.random.seed(42)
    torch.manual_seed(42)

    q_np = np.random.randn(2, 1, 4, 8).astype(np.float32)
    k_np = np.random.randn(2, 1, 4, 8).astype(np.float32)
    v_np = np.random.randn(2, 1, 4, 8).astype(np.float32)

    q_torch = torch.tensor(q_np, dtype=torch.float32, requires_grad=True)
    k_torch = torch.tensor(k_np, dtype=torch.float32, requires_grad=True)
    v_torch = torch.tensor(v_np, dtype=torch.float32, requires_grad=True)

    out_torch = torch_scaled_dot_product_reference(q_torch, k_torch, v_torch)
    out_torch_np = out_torch.detach().cpu().numpy()

    q_kernel = Tensor(q_np, requires_grad=True)
    k_kernel = Tensor(k_np, requires_grad=True)
    v_kernel = Tensor(v_np, requires_grad=True)

    out_kernel = scaled_dot_product_attention(q_kernel, k_kernel, v_kernel)
    out_kernel = maybe_first(out_kernel)
    out_kernel_np = to_numpy(out_kernel)

    assert out_kernel_np.shape == out_torch_np.shape
    assert np.allclose(out_kernel_np, out_torch_np, atol=1e-5, rtol=1e-5)


def test_functional_scaled_dot_product_attention_forward_parity_with_causal_mask():
    np.random.seed(42)
    torch.manual_seed(42)

    q_np = np.random.randn(2, 1, 4, 8).astype(np.float32)
    k_np = np.random.randn(2, 1, 4, 8).astype(np.float32)
    v_np = np.random.randn(2, 1, 4, 8).astype(np.float32)

    q_torch = torch.tensor(q_np, dtype=torch.float32, requires_grad=True)
    k_torch = torch.tensor(k_np, dtype=torch.float32, requires_grad=True)
    v_torch = torch.tensor(v_np, dtype=torch.float32, requires_grad=True)

    mask_kernel = make_causal_mask(4)
    mask_np = to_numpy(mask_kernel)
    mask_torch = torch.tensor(mask_np, dtype=torch.float32)

    out_torch = torch_scaled_dot_product_reference(q_torch, k_torch, v_torch, mask=mask_torch)
    out_torch_np = out_torch.detach().cpu().numpy()

    q_kernel = Tensor(q_np, requires_grad=True)
    k_kernel = Tensor(k_np, requires_grad=True)
    v_kernel = Tensor(v_np, requires_grad=True)

    out_kernel = scaled_dot_product_attention(q_kernel, k_kernel, v_kernel, mask=mask_kernel)
    out_kernel = maybe_first(out_kernel)
    out_kernel_np = to_numpy(out_kernel)

    assert out_kernel_np.shape == out_torch_np.shape
    assert np.allclose(out_kernel_np, out_torch_np, atol=1e-5, rtol=1e-5)


def test_functional_scaled_dot_product_attention_backward_q_grad_parity():
    np.random.seed(42)
    torch.manual_seed(42)

    q_np = np.random.randn(2, 1, 4, 8).astype(np.float32)
    k_np = np.random.randn(2, 1, 4, 8).astype(np.float32)
    v_np = np.random.randn(2, 1, 4, 8).astype(np.float32)

    q_torch = torch.tensor(q_np, dtype=torch.float32, requires_grad=True)
    k_torch = torch.tensor(k_np, dtype=torch.float32, requires_grad=True)
    v_torch = torch.tensor(v_np, dtype=torch.float32, requires_grad=True)

    out_torch = torch_scaled_dot_product_reference(q_torch, k_torch, v_torch)
    loss_torch = out_torch.sum()
    loss_torch.backward()
    torch_q_grad = q_torch.grad.detach().cpu().numpy()

    q_kernel = Tensor(q_np, requires_grad=True)
    k_kernel = Tensor(k_np, requires_grad=True)
    v_kernel = Tensor(v_np, requires_grad=True)

    out_kernel = scaled_dot_product_attention(q_kernel, k_kernel, v_kernel)
    out_kernel = maybe_first(out_kernel)
    loss_kernel = out_kernel.sum()
    loss_kernel.backward()
    kernel_q_grad = get_grad(q_kernel)

    assert kernel_q_grad is not None
    assert kernel_q_grad.shape == torch_q_grad.shape
    assert np.allclose(kernel_q_grad, torch_q_grad, atol=1e-5, rtol=1e-5)


def test_module_scaled_dot_product_attention_forward_parity():
    np.random.seed(42)
    torch.manual_seed(42)

    q_np = np.random.randn(2, 1, 4, 8).astype(np.float32)
    k_np = np.random.randn(2, 1, 4, 8).astype(np.float32)
    v_np = np.random.randn(2, 1, 4, 8).astype(np.float32)

    q_torch = torch.tensor(q_np, dtype=torch.float32)
    k_torch = torch.tensor(k_np, dtype=torch.float32)
    v_torch = torch.tensor(v_np, dtype=torch.float32)

    out_torch = torch_scaled_dot_product_reference(q_torch, k_torch, v_torch)
    out_torch_np = out_torch.detach().cpu().numpy()

    attn = ScaledDotProductAttention()

    q_kernel = Tensor(q_np, requires_grad=False)
    k_kernel = Tensor(k_np, requires_grad=False)
    v_kernel = Tensor(v_np, requires_grad=False)

    out_kernel = attn(q_kernel, k_kernel, v_kernel)
    out_kernel = maybe_first(out_kernel)
    out_kernel_np = to_numpy(out_kernel)

    assert out_kernel_np.shape == out_torch_np.shape
    assert np.allclose(out_kernel_np, out_torch_np, atol=1e-5, rtol=1e-5)