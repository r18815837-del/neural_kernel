import numpy as np
import pytest

from kernel.nn.layers.linear import Linear
from kernel.core.tensor import Tensor


def to_numpy(x):
    import numpy as np

    try:
        import cupy as cp
    except Exception:
        cp = None

    if isinstance(x, np.ndarray):
        return x

    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)

    if hasattr(x, "data"):
        data = x.data

        if isinstance(data, np.ndarray):
            return data

        if cp is not None and isinstance(data, cp.ndarray):
            return cp.asnumpy(data)

        try:
            return np.array(data)
        except Exception:
            pass

    if hasattr(x, "numpy"):
        out = x.numpy()
        if cp is not None and isinstance(out, cp.ndarray):
            return cp.asnumpy(out)
        return np.array(out)

    return np.array(x)


def get_grad(x):
    grad = getattr(x, "grad", None)
    if grad is None:
        return None
    return to_numpy(grad)


def cuda_available():
    try:
        import cupy as cp
        return cp.cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not cuda_available(),
    reason="CUDA/CuPy is not available",
)


def clone_linear_to_cuda(layer_cpu):
    in_features = to_numpy(layer_cpu.weight).shape[0]
    out_features = to_numpy(layer_cpu.weight).shape[1]

    layer_gpu = Linear(in_features, out_features)
    layer_gpu.weight.data = to_numpy(layer_cpu.weight).copy()
    layer_gpu.bias.data = to_numpy(layer_cpu.bias).copy()
    return layer_gpu.cuda()


def test_linear_forward_parity_cpu_vs_cuda():
    np.random.seed(42)

    x_np = np.random.randn(5, 4).astype(np.float32)

    layer_cpu = Linear(4, 3)
    layer_gpu = clone_linear_to_cuda(layer_cpu)

    x_cpu = Tensor(x_np, requires_grad=False)
    y_cpu = layer_cpu(x_cpu)

    x_gpu = Tensor(x_np, requires_grad=False).cuda()
    y_gpu = layer_gpu(x_gpu)

    y_cpu_np = to_numpy(y_cpu)
    y_gpu_np = to_numpy(y_gpu)

    assert y_cpu_np.shape == y_gpu_np.shape
    assert np.allclose(y_cpu_np, y_gpu_np, atol=1e-5, rtol=1e-5)


def test_linear_backward_input_grad_parity_cpu_vs_cuda():
    np.random.seed(42)

    x_np = np.random.randn(5, 4).astype(np.float32)

    layer_cpu = Linear(4, 3)
    layer_gpu = clone_linear_to_cuda(layer_cpu)

    x_cpu = Tensor(x_np, requires_grad=True)
    y_cpu = layer_cpu(x_cpu)
    loss_cpu = y_cpu.sum()
    loss_cpu.backward()
    grad_x_cpu = get_grad(x_cpu)

    x_gpu = Tensor(x_np, requires_grad=True).cuda()
    y_gpu = layer_gpu(x_gpu)
    loss_gpu = y_gpu.sum()
    loss_gpu.backward()
    grad_x_gpu = get_grad(x_gpu)

    assert grad_x_cpu is not None
    assert grad_x_gpu is not None
    assert grad_x_cpu.shape == grad_x_gpu.shape
    assert np.allclose(grad_x_cpu, grad_x_gpu, atol=1e-5, rtol=1e-5)


def test_linear_backward_weight_grad_parity_cpu_vs_cuda():
    np.random.seed(42)

    x_np = np.random.randn(5, 4).astype(np.float32)

    layer_cpu = Linear(4, 3)
    layer_gpu = clone_linear_to_cuda(layer_cpu)

    x_cpu = Tensor(x_np, requires_grad=True)
    y_cpu = layer_cpu(x_cpu)
    loss_cpu = y_cpu.sum()
    loss_cpu.backward()
    grad_w_cpu = get_grad(layer_cpu.weight)

    x_gpu = Tensor(x_np, requires_grad=True).cuda()
    y_gpu = layer_gpu(x_gpu)
    loss_gpu = y_gpu.sum()
    loss_gpu.backward()
    grad_w_gpu = get_grad(layer_gpu.weight)

    assert grad_w_cpu is not None
    assert grad_w_gpu is not None
    assert grad_w_cpu.shape == grad_w_gpu.shape
    assert np.allclose(grad_w_cpu, grad_w_gpu, atol=1e-5, rtol=1e-5)


def test_linear_backward_bias_grad_parity_cpu_vs_cuda():
    np.random.seed(42)

    x_np = np.random.randn(5, 4).astype(np.float32)

    layer_cpu = Linear(4, 3)
    layer_gpu = clone_linear_to_cuda(layer_cpu)

    x_cpu = Tensor(x_np, requires_grad=True)
    y_cpu = layer_cpu(x_cpu)
    loss_cpu = y_cpu.sum()
    loss_cpu.backward()
    grad_b_cpu = get_grad(layer_cpu.bias)

    x_gpu = Tensor(x_np, requires_grad=True).cuda()
    y_gpu = layer_gpu(x_gpu)
    loss_gpu = y_gpu.sum()
    loss_gpu.backward()
    grad_b_gpu = get_grad(layer_gpu.bias)

    assert grad_b_cpu is not None
    assert grad_b_gpu is not None
    assert grad_b_cpu.shape == grad_b_gpu.shape
    assert np.allclose(grad_b_cpu, grad_b_gpu, atol=1e-5, rtol=1e-5)