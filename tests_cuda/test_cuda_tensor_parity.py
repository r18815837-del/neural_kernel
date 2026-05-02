import numpy as np
import pytest

from kernel.core.tensor import Tensor


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "data"):
        data = x.data
        if isinstance(data, np.ndarray):
            return data
        try:
            # cupy array -> numpy
            import cupy as cp

            if isinstance(data, cp.ndarray):
                return cp.asnumpy(data)
        except Exception:
            pass
        try:
            return np.array(data)
        except Exception:
            pass
    if hasattr(x, "numpy"):
        out = x.numpy()
        try:
            import cupy as cp

            if isinstance(out, cp.ndarray):
                return cp.asnumpy(out)
        except Exception:
            pass
        return np.array(out)
    return np.array(x)


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


def test_tensor_cuda_and_cpu_roundtrip():
    x = Tensor([1.0, 2.0, 3.0])
    y = x.cuda()

    assert x.device == "cpu"
    assert y.device == "cuda"

    z = y.cpu()
    assert z.device == "cpu"
    assert np.allclose(to_numpy(z), np.array([1.0, 2.0, 3.0]))


def test_tensor_add_parity_cpu_vs_cuda():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)

    a_cpu = Tensor(a_np)
    b_cpu = Tensor(b_np)
    out_cpu = a_cpu + b_cpu

    a_gpu = Tensor(a_np).cuda()
    b_gpu = Tensor(b_np).cuda()
    out_gpu = a_gpu + b_gpu

    assert np.allclose(to_numpy(out_cpu), to_numpy(out_gpu), atol=1e-5, rtol=1e-5)


def test_tensor_mul_parity_cpu_vs_cuda():
    x_np = np.random.randn(4, 5).astype(np.float32)

    x_cpu = Tensor(x_np)
    out_cpu = x_cpu * 2.5

    x_gpu = Tensor(x_np).cuda()
    out_gpu = x_gpu * 2.5

    assert np.allclose(to_numpy(out_cpu), to_numpy(out_gpu), atol=1e-5, rtol=1e-5)


def test_tensor_sum_parity_cpu_vs_cuda():
    x_np = np.random.randn(6, 7).astype(np.float32)

    x_cpu = Tensor(x_np)
    out_cpu = x_cpu.sum(axis=1)

    x_gpu = Tensor(x_np).cuda()
    out_gpu = x_gpu.sum(axis=1)

    assert np.allclose(to_numpy(out_cpu), to_numpy(out_gpu), atol=1e-5, rtol=1e-5)


def test_tensor_mean_parity_cpu_vs_cuda():
    x_np = np.random.randn(6, 7).astype(np.float32)

    x_cpu = Tensor(x_np)
    out_cpu = x_cpu.mean(axis=0)

    x_gpu = Tensor(x_np).cuda()
    out_gpu = x_gpu.mean(axis=0)

    assert np.allclose(to_numpy(out_cpu), to_numpy(out_gpu), atol=1e-5, rtol=1e-5)


def test_tensor_reshape_parity_cpu_vs_cuda():
    x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    x_cpu = Tensor(x_np)
    out_cpu = x_cpu.reshape(6, 4)

    x_gpu = Tensor(x_np).cuda()
    out_gpu = x_gpu.reshape(6, 4)

    assert to_numpy(out_cpu).shape == to_numpy(out_gpu).shape
    assert np.allclose(to_numpy(out_cpu), to_numpy(out_gpu), atol=1e-5, rtol=1e-5)