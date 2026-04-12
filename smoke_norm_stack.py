from kernel.core.tensor import Tensor
from kernel.backend import is_cuda_available

# поправь импорт под свою структуру файлов
# например:
# from kernel.nn.norm import BatchNorm1d, BatchNorm2d, LayerNorm
# или:
# from kernel.nn.modules.normalization import BatchNorm1d, BatchNorm2d, LayerNorm
from kernel.nn.normalization import BatchNorm1d, BatchNorm2d, LayerNorm


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_bn1d_cpu():
    bn = BatchNorm1d(4)
    x = Tensor([[1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0]], requires_grad=True, device="cpu")

    y = bn(x)
    assert y.shape == (2, 4)
    assert bn.running_mean is not None
    assert bn.running_var is not None
    assert bn.running_mean.device == "cpu"
    assert bn.running_var.device == "cpu"

    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert bn.weight.grad is not None
    assert bn.bias.grad is not None


def smoke_bn1d_eval_cpu():
    bn = BatchNorm1d(4)
    x = Tensor([[1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0]], requires_grad=True, device="cpu")

    _ = bn(x)   # обновляем running stats
    bn.eval()
    y = bn(x)

    assert y.shape == (2, 4)
    assert bn.training is False


def smoke_bn2d_cpu():
    import numpy as np

    bn = BatchNorm2d(3)
    x = Tensor(np.random.randn(2, 3, 4, 4), requires_grad=True, device="cpu")

    y = bn(x)
    assert y.shape == (2, 3, 4, 4)
    assert bn.running_mean.device == "cpu"
    assert bn.running_var.device == "cpu"

    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert bn.weight.grad is not None
    assert bn.bias.grad is not None


def smoke_layernorm_cpu():
    ln = LayerNorm(4)
    x = Tensor([[1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0]], requires_grad=True, device="cpu")

    y = ln(x)
    assert y.shape == (2, 4)

    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None


def smoke_state_dict_cpu():
    bn = BatchNorm1d(4)
    x = Tensor([[1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0]], requires_grad=True, device="cpu")
    _ = bn(x)

    sd = bn.state_dict()

    assert "weight" in sd
    assert "bias" in sd
    assert "running_mean" in sd
    assert "running_var" in sd
    assert sd["running_mean"].shape == (4,)
    assert sd["running_var"].shape == (4,)


def smoke_module_to_cpu():
    bn = BatchNorm1d(4)
    bn.to("cpu")

    assert bn.weight.device == "cpu"
    assert bn.bias.device == "cpu"
    assert bn.running_mean.device == "cpu"
    assert bn.running_var.device == "cpu"


def smoke_bn1d_cuda():
    bn = BatchNorm1d(4).cuda()
    x = Tensor([[1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0]], requires_grad=True, device="cuda")

    y = bn(x)
    assert y.device == "cuda"
    assert bn.weight.device == "cuda"
    assert bn.bias.device == "cuda"
    assert bn.running_mean.device == "cuda"
    assert bn.running_var.device == "cuda"

    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert bn.weight.grad is not None
    assert bn.bias.grad is not None


def smoke_bn2d_cuda():
    import numpy as np

    bn = BatchNorm2d(3).cuda()
    x = Tensor(np.random.randn(2, 3, 4, 4), requires_grad=True, device="cuda")

    y = bn(x)
    assert y.device == "cuda"
    assert y.shape == (2, 3, 4, 4)

    loss = y.mean()
    loss.backward()

    assert x.grad is not None


def smoke_layernorm_cuda():
    ln = LayerNorm(4).cuda()
    x = Tensor([[1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0]], requires_grad=True, device="cuda")

    y = ln(x)
    assert y.device == "cuda"

    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None


def main():
    check("BatchNorm1d CPU", smoke_bn1d_cpu)
    check("BatchNorm1d eval CPU", smoke_bn1d_eval_cpu)
    check("BatchNorm2d CPU", smoke_bn2d_cpu)
    check("LayerNorm CPU", smoke_layernorm_cpu)
    check("state_dict CPU", smoke_state_dict_cpu)
    check("Module.to(cpu)", smoke_module_to_cpu)

    if not is_cuda_available():
        print("[SKIP] CUDA is not available")
        return

    check("BatchNorm1d CUDA", smoke_bn1d_cuda)
    check("BatchNorm2d CUDA", smoke_bn2d_cuda)
    check("LayerNorm CUDA", smoke_layernorm_cuda)


if __name__ == "__main__":
    main()