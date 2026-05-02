import numpy as np
import torch

from kernel import Tensor
from kernel.nn.layers import Embedding


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


def assign_kernel_embedding_from_torch(kernel_emb, torch_emb):
    torch_w = torch_emb.weight.detach().cpu().numpy()
    kernel_w = to_numpy(kernel_emb.weight)

    if kernel_w.shape == torch_w.shape:
        kernel_emb.weight.data = torch_w.copy()
    else:
        raise AssertionError(
            f"Unsupported embedding weight shape mapping: kernel {kernel_w.shape}, torch {torch_w.shape}"
        )


def test_embedding_forward_parity_with_torch_1d():
    np.random.seed(42)
    torch.manual_seed(42)

    vocab_size = 10
    emb_dim = 4

    ids_np = np.array([1, 3, 5, 7], dtype=np.int64)

    torch_emb = torch.nn.Embedding(vocab_size, emb_dim)
    kernel_emb = Embedding(vocab_size, emb_dim)

    assign_kernel_embedding_from_torch(kernel_emb, torch_emb)

    ids_torch = torch.tensor(ids_np, dtype=torch.long)
    out_torch = torch_emb(ids_torch).detach().cpu().numpy()

    ids_kernel = Tensor(ids_np)
    out_kernel = to_numpy(kernel_emb(ids_kernel))

    assert out_kernel.shape == out_torch.shape
    assert np.allclose(out_kernel, out_torch, atol=1e-5, rtol=1e-5)


def test_embedding_forward_parity_with_torch_2d():
    np.random.seed(42)
    torch.manual_seed(42)

    vocab_size = 12
    emb_dim = 5

    ids_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)

    torch_emb = torch.nn.Embedding(vocab_size, emb_dim)
    kernel_emb = Embedding(vocab_size, emb_dim)

    assign_kernel_embedding_from_torch(kernel_emb, torch_emb)

    ids_torch = torch.tensor(ids_np, dtype=torch.long)
    out_torch = torch_emb(ids_torch).detach().cpu().numpy()

    ids_kernel = Tensor(ids_np)
    out_kernel = to_numpy(kernel_emb(ids_kernel))

    assert out_kernel.shape == out_torch.shape
    assert np.allclose(out_kernel, out_torch, atol=1e-5, rtol=1e-5)


def test_embedding_repeated_indices_match_torch():
    np.random.seed(42)
    torch.manual_seed(42)

    vocab_size = 8
    emb_dim = 3

    ids_np = np.array([2, 2, 2, 5], dtype=np.int64)

    torch_emb = torch.nn.Embedding(vocab_size, emb_dim)
    kernel_emb = Embedding(vocab_size, emb_dim)

    assign_kernel_embedding_from_torch(kernel_emb, torch_emb)

    ids_torch = torch.tensor(ids_np, dtype=torch.long)
    out_torch = torch_emb(ids_torch).detach().cpu().numpy()

    ids_kernel = Tensor(ids_np)
    out_kernel = to_numpy(kernel_emb(ids_kernel))

    assert np.allclose(out_kernel, out_torch, atol=1e-5, rtol=1e-5)
    assert np.allclose(out_kernel[0], out_kernel[1], atol=1e-5, rtol=1e-5)


def test_embedding_backward_weight_grad_parity_with_torch():
    np.random.seed(42)
    torch.manual_seed(42)

    vocab_size = 10
    emb_dim = 4

    ids_np = np.array([[1, 2, 1], [3, 4, 2]], dtype=np.int64)

    torch_emb = torch.nn.Embedding(vocab_size, emb_dim)
    kernel_emb = Embedding(vocab_size, emb_dim)

    assign_kernel_embedding_from_torch(kernel_emb, torch_emb)

    ids_torch = torch.tensor(ids_np, dtype=torch.long)
    out_torch = torch_emb(ids_torch)
    loss_torch = out_torch.sum()
    loss_torch.backward()
    torch_w_grad = torch_emb.weight.grad.detach().cpu().numpy()

    ids_kernel = Tensor(ids_np)
    out_kernel = kernel_emb(ids_kernel)
    loss_kernel = out_kernel.sum()
    loss_kernel.backward()
    kernel_w_grad = get_grad(kernel_emb.weight)

    assert kernel_w_grad is not None
    assert kernel_w_grad.shape == torch_w_grad.shape
    assert np.allclose(kernel_w_grad, torch_w_grad, atol=1e-5, rtol=1e-5)


def test_embedding_forward_outputs_finite_values():
    np.random.seed(42)
    torch.manual_seed(42)

    vocab_size = 10
    emb_dim = 4
    ids_np = np.array([[0, 1], [2, 3]], dtype=np.int64)

    torch_emb = torch.nn.Embedding(vocab_size, emb_dim)
    kernel_emb = Embedding(vocab_size, emb_dim)

    assign_kernel_embedding_from_torch(kernel_emb, torch_emb)

    ids_kernel = Tensor(ids_np)
    out_kernel = to_numpy(kernel_emb(ids_kernel))

    assert np.isfinite(out_kernel).all()