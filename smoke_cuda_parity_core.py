import numpy as np

from kernel.core.tensor import Tensor
from kernel.backend import is_cuda_available
from kernel.nn.layers.embedding import Embedding
from kernel.nn.losses import CrossEntropyLoss


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def assert_close(a, b, atol=1e-6, rtol=1e-6, msg=""):
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(msg or f"Arrays are not close.\nA={a}\nB={b}")


def parity_basic_math():
    x_cpu = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, device="cpu")
    y_cpu = ((x_cpu + 1.0) * 2.0 / 3.0).mean()
    y_cpu.backward()

    x_cuda = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, device="cuda")
    y_cuda = ((x_cuda + 1.0) * 2.0 / 3.0).mean()
    y_cuda.backward()

    assert_close(y_cpu.detach().numpy(), y_cuda.detach().numpy(), msg="basic math forward mismatch")
    assert_close(x_cpu.grad, x_cuda.grad.get() if hasattr(x_cuda.grad, "get") else x_cuda.grad,
                 msg="basic math grad mismatch")


def parity_matmul():
    a_np = np.arange(12, dtype=np.float64).reshape(3, 4)
    b_np = np.arange(8, dtype=np.float64).reshape(4, 2)

    a_cpu = Tensor(a_np, requires_grad=True, device="cpu")
    b_cpu = Tensor(b_np, requires_grad=True, device="cpu")
    out_cpu = (a_cpu @ b_cpu).mean()
    out_cpu.backward()

    a_cuda = Tensor(a_np, requires_grad=True, device="cuda")
    b_cuda = Tensor(b_np, requires_grad=True, device="cuda")
    out_cuda = (a_cuda @ b_cuda).mean()
    out_cuda.backward()

    assert_close(out_cpu.detach().numpy(), out_cuda.detach().numpy(), msg="matmul forward mismatch")
    assert_close(a_cpu.grad, a_cuda.grad.get() if hasattr(a_cuda.grad, "get") else a_cuda.grad,
                 msg="matmul grad a mismatch")
    assert_close(b_cpu.grad, b_cuda.grad.get() if hasattr(b_cuda.grad, "get") else b_cuda.grad,
                 msg="matmul grad b mismatch")


def parity_shape_ops():
    x_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

    x_cpu = Tensor(x_np, requires_grad=True, device="cpu")
    y_cpu = x_cpu.transpose(0, 2, 1).reshape(2, 12).mean()
    y_cpu.backward()

    x_cuda = Tensor(x_np, requires_grad=True, device="cuda")
    y_cuda = x_cuda.transpose(0, 2, 1).reshape(2, 12).mean()
    y_cuda.backward()

    assert_close(y_cpu.detach().numpy(), y_cuda.detach().numpy(), msg="shape ops forward mismatch")
    assert_close(x_cpu.grad, x_cuda.grad.get() if hasattr(x_cuda.grad, "get") else x_cuda.grad,
                 msg="shape ops grad mismatch")


def parity_reduce_ops():
    x_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

    x_cpu = Tensor(x_np, requires_grad=True, device="cpu")
    y_cpu = x_cpu.mean(axis=(1, 2)).sum()
    y_cpu.backward()

    x_cuda = Tensor(x_np, requires_grad=True, device="cuda")
    y_cuda = x_cuda.mean(axis=(1, 2)).sum()
    y_cuda.backward()

    assert_close(y_cpu.detach().numpy(), y_cuda.detach().numpy(), msg="reduce forward mismatch")
    assert_close(x_cpu.grad, x_cuda.grad.get() if hasattr(x_cuda.grad, "get") else x_cuda.grad,
                 msg="reduce grad mismatch")


def parity_softmax():
    from kernel.autograd.ops.math_ops import softmax

    x_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

    x_cpu = Tensor(x_np, requires_grad=True, device="cpu")
    y_cpu = softmax(x_cpu, axis=-1).mean()
    y_cpu.backward()

    x_cuda = Tensor(x_np, requires_grad=True, device="cuda")
    y_cuda = softmax(x_cuda, axis=-1).mean()
    y_cuda.backward()

    assert_close(y_cpu.detach().numpy(), y_cuda.detach().numpy(), msg="softmax forward mismatch")
    assert_close(x_cpu.grad, x_cuda.grad.get() if hasattr(x_cuda.grad, "get") else x_cuda.grad,
                 atol=1e-5, rtol=1e-5, msg="softmax grad mismatch")


def parity_embedding():
    idx = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.int64)

    np.random.seed(42)
    emb_cpu = Embedding(num_embeddings=10, embedding_dim=4)
    state = emb_cpu.state_dict()

    emb_cuda = Embedding(num_embeddings=10, embedding_dim=4)
    emb_cuda.load_state_dict(state)
    emb_cuda.cuda()

    out_cpu = emb_cpu(idx).mean()
    out_cpu.backward()

    out_cuda = emb_cuda(idx).mean()
    out_cuda.backward()

    grad_cuda = emb_cuda.weight.grad.get() if hasattr(emb_cuda.weight.grad, "get") else emb_cuda.weight.grad

    assert_close(out_cpu.detach().numpy(), out_cuda.detach().numpy(), msg="embedding forward mismatch")
    assert_close(emb_cpu.weight.grad, grad_cuda, msg="embedding grad mismatch")


def parity_cross_entropy():
    np.random.seed(42)
    logits_np = np.random.randn(6, 5).astype(np.float64)
    targets_np = np.array([0, 1, 2, 3, 4, 0], dtype=np.int64)

    loss_fn = CrossEntropyLoss()

    logits_cpu = Tensor(logits_np, requires_grad=True, device="cpu")
    loss_cpu = loss_fn(logits_cpu, targets_np)
    loss_cpu.backward()

    logits_cuda = Tensor(logits_np, requires_grad=True, device="cuda")
    loss_cuda = loss_fn(logits_cuda, targets_np)
    loss_cuda.backward()

    grad_cuda = logits_cuda.grad.get() if hasattr(logits_cuda.grad, "get") else logits_cuda.grad

    assert_close(loss_cpu.detach().numpy(), loss_cuda.detach().numpy(), atol=1e-5, rtol=1e-5,
                 msg="cross entropy forward mismatch")
    assert_close(logits_cpu.grad, grad_cuda, atol=1e-5, rtol=1e-5,
                 msg="cross entropy grad mismatch")


def main():
    if not is_cuda_available():
        print("[SKIP] CUDA is not available")
        return

    check("parity basic math", parity_basic_math)
    check("parity matmul", parity_matmul)
    check("parity shape ops", parity_shape_ops)
    check("parity reduce ops", parity_reduce_ops)
    check("parity softmax", parity_softmax)
    check("parity embedding", parity_embedding)
    check("parity cross entropy", parity_cross_entropy)


if __name__ == "__main__":
    main()