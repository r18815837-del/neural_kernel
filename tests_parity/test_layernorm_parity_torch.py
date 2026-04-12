import numpy as np
import torch

from kernel import LayerNorm, Tensor


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


def assign_kernel_layernorm_from_torch(kernel_ln, torch_ln):
    torch_w = torch_ln.weight.detach().cpu().numpy()
    torch_b = torch_ln.bias.detach().cpu().numpy()

    kernel_w = to_numpy(kernel_ln.weight)
    kernel_b = to_numpy(kernel_ln.bias)

    if kernel_w.shape == torch_w.shape:
        kernel_ln.weight.data = torch_w.copy()
    elif kernel_w.shape == (1, torch_w.shape[0]):
        kernel_ln.weight.data = torch_w.reshape(1, -1).copy()
    else:
        raise AssertionError(
            f"Unsupported weight shape mapping: kernel {kernel_w.shape}, torch {torch_w.shape}"
        )

    if kernel_b.shape == torch_b.shape:
        kernel_ln.bias.data = torch_b.copy()
    elif kernel_b.shape == (1, torch_b.shape[0]):
        kernel_ln.bias.data = torch_b.reshape(1, -1).copy()
    else:
        raise AssertionError(
            f"Unsupported bias shape mapping: kernel {kernel_b.shape}, torch {torch_b.shape}"
        )


def test_layernorm_forward_parity_with_torch_2d():
    np.random.seed(42)
    torch.manual_seed(42)

    x_np = np.random.randn(5, 4).astype(np.float32)

    torch_ln = torch.nn.LayerNorm(4, elementwise_affine=True)
    kernel_ln = LayerNorm(4)

    assign_kernel_layernorm_from_torch(kernel_ln, torch_ln)

    x_torch = torch.tensor(x_np, dtype=torch.float32)
    y_torch = torch_ln(x_torch).detach().cpu().numpy()

    x_kernel = Tensor(x_np, requires_grad=False)
    y_kernel = to_numpy(kernel_ln(x_kernel))

    assert y_kernel.shape == y_torch.shape
    assert np.allclose(y_kernel, y_torch, atol=1e-5, rtol=1e-5)


def test_layernorm_forward_parity_with_torch_3d():
    np.random.seed(42)
    torch.manual_seed(42)

    x_np = np.random.randn(2, 3, 4).astype(np.float32)

    torch_ln = torch.nn.LayerNorm(4, elementwise_affine=True)
    kernel_ln = LayerNorm(4)

    assign_kernel_layernorm_from_torch(kernel_ln, torch_ln)

    x_torch = torch.tensor(x_np, dtype=torch.float32)
    y_torch = torch_ln(x_torch).detach().cpu().numpy()

    x_kernel = Tensor(x_np, requires_grad=False)
    y_kernel = to_numpy(kernel_ln(x_kernel))

    assert y_kernel.shape == y_torch.shape
    assert np.allclose(y_kernel, y_torch, atol=1e-5, rtol=1e-5)


def test_layernorm_backward_input_grad_parity_with_torch():
    np.random.seed(42)
    torch.manual_seed(42)

    x_np = np.random.randn(3, 4).astype(np.float32)

    torch_ln = torch.nn.LayerNorm(4, elementwise_affine=True)
    kernel_ln = LayerNorm(4)

    assign_kernel_layernorm_from_torch(kernel_ln, torch_ln)

    x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    y_torch = torch_ln(x_torch)
    loss_torch = y_torch.sum()
    loss_torch.backward()
    torch_x_grad = x_torch.grad.detach().cpu().numpy()

    x_kernel = Tensor(x_np, requires_grad=True)
    y_kernel = kernel_ln(x_kernel)
    loss_kernel = y_kernel.sum()
    loss_kernel.backward()
    kernel_x_grad = get_grad(x_kernel)

    assert kernel_x_grad is not None
    assert kernel_x_grad.shape == torch_x_grad.shape
    assert np.allclose(kernel_x_grad, torch_x_grad, atol=1e-5, rtol=1e-5)


def test_layernorm_backward_weight_grad_parity_with_torch():
    np.random.seed(42)
    torch.manual_seed(42)

    x_np = np.random.randn(3, 4).astype(np.float32)

    torch_ln = torch.nn.LayerNorm(4, elementwise_affine=True)
    kernel_ln = LayerNorm(4)

    assign_kernel_layernorm_from_torch(kernel_ln, torch_ln)

    x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    y_torch = torch_ln(x_torch)
    loss_torch = y_torch.sum()
    loss_torch.backward()
    torch_w_grad = torch_ln.weight.grad.detach().cpu().numpy()

    x_kernel = Tensor(x_np, requires_grad=True)
    y_kernel = kernel_ln(x_kernel)
    loss_kernel = y_kernel.sum()
    loss_kernel.backward()
    kernel_w_grad = get_grad(kernel_ln.weight)

    assert kernel_w_grad is not None

    if kernel_w_grad.shape == torch_w_grad.shape:
        assert np.allclose(kernel_w_grad, torch_w_grad, atol=1e-5, rtol=1e-5)
    elif kernel_w_grad.shape == (1, torch_w_grad.shape[0]):
        assert np.allclose(
            kernel_w_grad,
            torch_w_grad.reshape(1, -1),
            atol=1e-5,
            rtol=1e-5,
        )
    else:
        raise AssertionError(
            f"Unsupported weight grad shape mapping: kernel {kernel_w_grad.shape}, torch {torch_w_grad.shape}"
        )


def test_layernorm_backward_bias_grad_parity_with_torch():
    np.random.seed(42)
    torch.manual_seed(42)

    x_np = np.random.randn(3, 4).astype(np.float32)

    torch_ln = torch.nn.LayerNorm(4, elementwise_affine=True)
    kernel_ln = LayerNorm(4)

    assign_kernel_layernorm_from_torch(kernel_ln, torch_ln)

    x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    y_torch = torch_ln(x_torch)
    loss_torch = y_torch.sum()
    loss_torch.backward()
    torch_b_grad = torch_ln.bias.grad.detach().cpu().numpy()

    x_kernel = Tensor(x_np, requires_grad=True)
    y_kernel = kernel_ln(x_kernel)
    loss_kernel = y_kernel.sum()
    loss_kernel.backward()
    kernel_b_grad = get_grad(kernel_ln.bias)

    assert kernel_b_grad is not None

    if kernel_b_grad.shape == torch_b_grad.shape:
        assert np.allclose(kernel_b_grad, torch_b_grad, atol=1e-5, rtol=1e-5)
    elif kernel_b_grad.shape == (1, torch_b_grad.shape[0]):
        assert np.allclose(
            kernel_b_grad,
            torch_b_grad.reshape(1, -1),
            atol=1e-5,
            rtol=1e-5,
        )
    else:
        raise AssertionError(
            f"Unsupported bias grad shape mapping: kernel {kernel_b_grad.shape}, torch {torch_b_grad.shape}"
        )