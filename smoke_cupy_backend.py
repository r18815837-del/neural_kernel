from kernel.core.tensor import Tensor
from kernel.backend import is_cuda_available


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_cpu():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True, device="cpu")
    y = x + 1
    assert y.device == "cpu"
    assert y.shape == (3,)
    assert (y.numpy() == [2.0, 3.0, 4.0]).all()

    z = (x * 2).mean()
    z.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def smoke_cuda_tensor_creation():
    x = Tensor([1.0, 2.0, 3.0], device="cuda")
    assert x.device == "cuda"
    arr = x.numpy()
    assert arr.shape == (3,)


def smoke_cuda_scalar_promotion():
    x = Tensor([1.0, 2.0, 3.0], device="cuda")
    y = x + 1
    z = 2 * x
    w = x / 2.0

    assert y.device == "cuda"
    assert z.device == "cuda"
    assert w.device == "cuda"


def smoke_cuda_backward():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True, device="cuda")
    y = (x * 2).mean()
    y.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def smoke_cuda_matmul():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, device="cuda")
    b = Tensor([[5.0], [6.0]], requires_grad=True, device="cuda")
    out = a @ b
    assert out.device == "cuda"
    assert out.shape == (2, 1)

    loss = out.mean()
    loss.backward()

    assert a.grad is not None
    assert b.grad is not None


def smoke_cuda_reshape():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, device="cuda")
    y = x.reshape(4)
    assert y.device == "cuda"
    assert y.shape == (4,)


def smoke_cuda_activations():
    x = Tensor([-1.0, 0.0, 2.0], requires_grad=True, device="cuda")

    y = x.relu()
    s = x.sigmoid()

    assert y.device == "cuda"
    assert s.device == "cuda"
    assert y.shape == x.shape
    assert s.shape == x.shape


def smoke_cuda_sum_mean():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True, device="cuda")

    s = x.sum()
    m = x.mean()

    assert s.device == "cuda"
    assert m.device == "cuda"

    s.backward()
    assert x.grad is not None


def main():
    check("cpu basic", smoke_cpu)

    if not is_cuda_available():
        print("[SKIP] CUDA is not available")
        return

    check("cuda tensor creation", smoke_cuda_tensor_creation)
    check("cuda scalar promotion", smoke_cuda_scalar_promotion)
    check("cuda backward", smoke_cuda_backward)
    check("cuda matmul", smoke_cuda_matmul)
    check("cuda reshape", smoke_cuda_reshape)
    check("cuda activations", smoke_cuda_activations)
    check("cuda sum/mean", smoke_cuda_sum_mean)


if __name__ == "__main__":
    main()