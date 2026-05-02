import numpy as np
import torch

from kernel import CrossEntropyLoss, Tensor


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


def test_cross_entropy_forward_parity_with_torch_single_example():
    np.random.seed(42)
    torch.manual_seed(42)

    logits_np = np.array([[2.0, 1.0, 0.1]], dtype=np.float32)
    targets_np = np.array([0], dtype=np.int64)

    torch_loss_fn = torch.nn.CrossEntropyLoss()
    kernel_loss_fn = CrossEntropyLoss()

    logits_torch = torch.tensor(logits_np, dtype=torch.float32, requires_grad=True)
    targets_torch = torch.tensor(targets_np, dtype=torch.long)
    loss_torch = torch_loss_fn(logits_torch, targets_torch).detach().cpu().numpy()

    logits_kernel = Tensor(logits_np, requires_grad=True)
    targets_kernel = Tensor(targets_np)
    loss_kernel = to_numpy(kernel_loss_fn(logits_kernel, targets_kernel))

    assert np.allclose(loss_kernel, loss_torch, atol=1e-5, rtol=1e-5)


def test_cross_entropy_forward_parity_with_torch_batch():
    np.random.seed(42)
    torch.manual_seed(42)

    logits_np = np.random.randn(5, 4).astype(np.float32)
    targets_np = np.array([0, 1, 2, 3, 1], dtype=np.int64)

    torch_loss_fn = torch.nn.CrossEntropyLoss()
    kernel_loss_fn = CrossEntropyLoss()

    logits_torch = torch.tensor(logits_np, dtype=torch.float32, requires_grad=True)
    targets_torch = torch.tensor(targets_np, dtype=torch.long)
    loss_torch = torch_loss_fn(logits_torch, targets_torch).detach().cpu().numpy()

    logits_kernel = Tensor(logits_np, requires_grad=True)
    targets_kernel = Tensor(targets_np)
    loss_kernel = to_numpy(kernel_loss_fn(logits_kernel, targets_kernel))

    assert np.allclose(loss_kernel, loss_torch, atol=1e-5, rtol=1e-5)


def test_cross_entropy_backward_logits_grad_parity_with_torch():
    np.random.seed(42)
    torch.manual_seed(42)

    logits_np = np.random.randn(6, 5).astype(np.float32)
    targets_np = np.array([0, 1, 2, 3, 4, 1], dtype=np.int64)

    torch_loss_fn = torch.nn.CrossEntropyLoss()
    kernel_loss_fn = CrossEntropyLoss()

    logits_torch = torch.tensor(logits_np, dtype=torch.float32, requires_grad=True)
    targets_torch = torch.tensor(targets_np, dtype=torch.long)
    loss_torch = torch_loss_fn(logits_torch, targets_torch)
    loss_torch.backward()
    torch_grad = logits_torch.grad.detach().cpu().numpy()

    logits_kernel = Tensor(logits_np, requires_grad=True)
    targets_kernel = Tensor(targets_np)
    loss_kernel = kernel_loss_fn(logits_kernel, targets_kernel)
    loss_kernel.backward()
    kernel_grad = get_grad(logits_kernel)

    assert kernel_grad is not None
    assert kernel_grad.shape == torch_grad.shape
    assert np.allclose(kernel_grad, torch_grad, atol=1e-5, rtol=1e-5)


def test_cross_entropy_good_logits_lower_than_bad_logits():
    kernel_loss_fn = CrossEntropyLoss()

    good_logits = Tensor(np.array([[5.0, 1.0, 0.1]], dtype=np.float32), requires_grad=True)
    bad_logits = Tensor(np.array([[0.1, 1.0, 5.0]], dtype=np.float32), requires_grad=True)
    targets = Tensor(np.array([0], dtype=np.int64))

    good_loss = to_numpy(kernel_loss_fn(good_logits, targets))
    bad_loss = to_numpy(kernel_loss_fn(bad_logits, targets))

    assert good_loss < bad_loss