import numpy as np
import torch

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


def copy_linear_kernel_from_torch(kernel_linear, torch_linear):
    torch_w = torch_linear.weight.detach().cpu().numpy()
    torch_b = torch_linear.bias.detach().cpu().numpy()

    kernel_linear.weight.data = torch_w.T.copy()
    kernel_linear.bias.data = torch_b.reshape(1, -1).copy()


def copy_layernorm_kernel_from_torch(kernel_ln, torch_ln):
    kernel_ln.weight.data = torch_ln.weight.detach().cpu().numpy().reshape(1, -1).copy()
    kernel_ln.bias.data = torch_ln.bias.detach().cpu().numpy().reshape(1, -1).copy()


def copy_mha_kernel_from_torch(kernel_mha, torch_mha):
    d_model = torch_mha.embed_dim

    w_q = torch_mha.in_proj_weight[:d_model].detach().cpu().numpy()
    w_k = torch_mha.in_proj_weight[d_model:2 * d_model].detach().cpu().numpy()
    w_v = torch_mha.in_proj_weight[2 * d_model:].detach().cpu().numpy()

    b_q = torch_mha.in_proj_bias[:d_model].detach().cpu().numpy()
    b_k = torch_mha.in_proj_bias[d_model:2 * d_model].detach().cpu().numpy()
    b_v = torch_mha.in_proj_bias[2 * d_model:].detach().cpu().numpy()

    class Tmp:
        def __init__(self, w, b):
            self.weight = torch.tensor(w)
            self.bias = torch.tensor(b)

    copy_linear_kernel_from_torch(kernel_mha.q_proj, Tmp(w_q, b_q))
    copy_linear_kernel_from_torch(kernel_mha.k_proj, Tmp(w_k, b_k))
    copy_linear_kernel_from_torch(kernel_mha.v_proj, Tmp(w_v, b_v))
    copy_linear_kernel_from_torch(kernel_mha.out_proj, torch_mha.out_proj)


class TorchTransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ln2 = torch.nn.LayerNorm(d_model)

        self.attn = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

        self.ff1 = torch.nn.Linear(d_model, d_ff)
        self.ff2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x, attn_mask=None):
        attn_in = self.ln1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out

        ffn_in = self.ln2(x)
        ffn_out = self.ff2(torch.relu(self.ff1(ffn_in)))
        x = x + ffn_out

        return x


def build_torch_attn_mask_from_kernel(mask_kernel):
    mask_np = to_numpy(mask_kernel)
    return torch.tensor(mask_np[0, 0], dtype=torch.float32)


def sync_kernel_block_from_torch(kernel_block, torch_block):
    copy_mha_kernel_from_torch(kernel_block.attn, torch_block.attn)
    copy_layernorm_kernel_from_torch(kernel_block.ln1, torch_block.ln1)
    copy_layernorm_kernel_from_torch(kernel_block.ln2, torch_block.ln2)
    copy_linear_kernel_from_torch(kernel_block.ffn.fc1, torch_block.ff1)
    copy_linear_kernel_from_torch(kernel_block.ffn.fc2, torch_block.ff2)


def test_transformer_block_forward_parity_with_torch():
    np.random.seed(42)
    torch.manual_seed(42)

    d_model = 8
    num_heads = 2
    d_ff = 16

    x_np = np.random.randn(2, 4, d_model).astype(np.float32)

    torch_block = TorchTransformerBlock(d_model, num_heads, d_ff)
    kernel_block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)

    sync_kernel_block_from_torch(kernel_block, torch_block)

    x_torch = torch.tensor(x_np, dtype=torch.float32)
    out_torch = torch_block(x_torch).detach().cpu().numpy()

    x_kernel = Tensor(x_np)
    out_kernel = maybe_first(kernel_block(x_kernel))
    out_kernel_np = to_numpy(out_kernel)

    assert out_kernel_np.shape == out_torch.shape
    assert np.allclose(out_kernel_np, out_torch, atol=1e-5, rtol=1e-5)


def test_transformer_block_forward_parity_with_torch_causal_mask():
    np.random.seed(42)
    torch.manual_seed(42)

    d_model = 8
    num_heads = 2
    d_ff = 16

    x_np = np.random.randn(2, 4, d_model).astype(np.float32)

    torch_block = TorchTransformerBlock(d_model, num_heads, d_ff)
    kernel_block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)

    sync_kernel_block_from_torch(kernel_block, torch_block)

    kernel_mask = make_causal_mask(4)
    torch_mask = build_torch_attn_mask_from_kernel(kernel_mask)

    x_torch = torch.tensor(x_np, dtype=torch.float32)
    out_torch = torch_block(x_torch, attn_mask=torch_mask).detach().cpu().numpy()

    x_kernel = Tensor(x_np)
    out_kernel = maybe_first(kernel_block(x_kernel, kernel_mask))
    out_kernel_np = to_numpy(out_kernel)

    assert out_kernel_np.shape == out_torch.shape
    assert np.allclose(out_kernel_np, out_torch, atol=1e-5, rtol=1e-5)


def test_transformer_block_backward_input_grad_parity_with_torch():
    np.random.seed(42)
    torch.manual_seed(42)

    d_model = 8
    num_heads = 2
    d_ff = 16

    x_np = np.random.randn(2, 4, d_model).astype(np.float32)

    torch_block = TorchTransformerBlock(d_model, num_heads, d_ff)
    kernel_block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)

    sync_kernel_block_from_torch(kernel_block, torch_block)

    x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    out_torch = torch_block(x_torch)
    loss_torch = out_torch.sum()
    loss_torch.backward()
    torch_x_grad = x_torch.grad.detach().cpu().numpy()

    x_kernel = Tensor(x_np, requires_grad=True)
    out_kernel = maybe_first(kernel_block(x_kernel))
    loss_kernel = out_kernel.sum()
    loss_kernel.backward()
    kernel_x_grad = get_grad(x_kernel)

    assert kernel_x_grad is not None
    assert kernel_x_grad.shape == torch_x_grad.shape
    assert np.allclose(kernel_x_grad, torch_x_grad, atol=1e-5, rtol=1e-5)