import numpy as np
import pytest

from kernel.core.tensor import Tensor
from kernel.nn.functional import make_causal_mask
from kernel.nn.modules import TransformerBlock


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


def maybe_first(x):
    if isinstance(x, tuple):
        return x[0]
    return x


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


def clone_block_to_cuda(block_cpu, d_model, num_heads, d_ff):
    block_gpu = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout_p=0.0,
    )

    # LayerNorms
    block_gpu.ln1.weight.data = to_numpy(block_cpu.ln1.weight).copy()
    block_gpu.ln1.bias.data = to_numpy(block_cpu.ln1.bias).copy()

    block_gpu.ln2.weight.data = to_numpy(block_cpu.ln2.weight).copy()
    block_gpu.ln2.bias.data = to_numpy(block_cpu.ln2.bias).copy()

    # Attention projections
    for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
        src = getattr(block_cpu.attn, name)
        dst = getattr(block_gpu.attn, name)
        dst.weight.data = to_numpy(src.weight).copy()
        dst.bias.data = to_numpy(src.bias).copy()

    # FeedForward
    block_gpu.ffn.fc1.weight.data = to_numpy(block_cpu.ffn.fc1.weight).copy()
    block_gpu.ffn.fc1.bias.data = to_numpy(block_cpu.ffn.fc1.bias).copy()
    block_gpu.ffn.fc2.weight.data = to_numpy(block_cpu.ffn.fc2.weight).copy()
    block_gpu.ffn.fc2.bias.data = to_numpy(block_cpu.ffn.fc2.bias).copy()

    return block_gpu.cuda()


def test_transformer_block_forward_parity_cpu_vs_cuda():
    np.random.seed(42)

    d_model = 8
    num_heads = 2
    d_ff = 16
    x_np = np.random.randn(2, 4, d_model).astype(np.float32)

    block_cpu = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout_p=0.0,
    )
    block_gpu = clone_block_to_cuda(block_cpu, d_model, num_heads, d_ff)

    x_cpu = Tensor(x_np, requires_grad=False)
    out_cpu = maybe_first(block_cpu(x_cpu))

    x_gpu = Tensor(x_np, requires_grad=False).cuda()
    out_gpu = maybe_first(block_gpu(x_gpu))

    out_cpu_np = to_numpy(out_cpu)
    out_gpu_np = to_numpy(out_gpu)

    assert out_cpu_np.shape == out_gpu_np.shape
    assert np.allclose(out_cpu_np, out_gpu_np, atol=1e-5, rtol=1e-5)


def test_transformer_block_forward_parity_cpu_vs_cuda_with_causal_mask():
    np.random.seed(42)

    d_model = 8
    num_heads = 2
    d_ff = 16
    x_np = np.random.randn(2, 4, d_model).astype(np.float32)

    block_cpu = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout_p=0.0,
    )
    block_gpu = clone_block_to_cuda(block_cpu, d_model, num_heads, d_ff)

    mask_cpu = make_causal_mask(4)
    mask_gpu = mask_cpu.cuda()

    x_cpu = Tensor(x_np, requires_grad=False)
    out_cpu = maybe_first(block_cpu(x_cpu, mask_cpu))

    x_gpu = Tensor(x_np, requires_grad=False).cuda()
    out_gpu = maybe_first(block_gpu(x_gpu, mask_gpu))

    out_cpu_np = to_numpy(out_cpu)
    out_gpu_np = to_numpy(out_gpu)

    assert out_cpu_np.shape == out_gpu_np.shape
    assert np.allclose(out_cpu_np, out_gpu_np, atol=1e-5, rtol=1e-5)


def test_transformer_block_backward_input_grad_parity_cpu_vs_cuda():
    np.random.seed(42)

    d_model = 8
    num_heads = 2
    d_ff = 16
    x_np = np.random.randn(2, 4, d_model).astype(np.float32)

    block_cpu = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout_p=0.0,
    )
    block_gpu = clone_block_to_cuda(block_cpu, d_model, num_heads, d_ff)

    x_cpu = Tensor(x_np, requires_grad=True)
    out_cpu = maybe_first(block_cpu(x_cpu))
    loss_cpu = out_cpu.sum()
    loss_cpu.backward()
    grad_x_cpu = get_grad(x_cpu)

    x_gpu = Tensor(x_np, requires_grad=True).cuda()
    out_gpu = maybe_first(block_gpu(x_gpu))
    loss_gpu = out_gpu.sum()
    loss_gpu.backward()
    grad_x_gpu = get_grad(x_gpu)

    assert grad_x_cpu is not None
    assert grad_x_gpu is not None
    assert grad_x_cpu.shape == grad_x_gpu.shape
    assert np.allclose(grad_x_cpu, grad_x_gpu, atol=1e-5, rtol=1e-5)