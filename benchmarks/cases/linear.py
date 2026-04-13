from __future__ import annotations

from kernel import Linear, Tensor

from benchmarks.utils.timing import benchmark_function


def run_linear_forward_benchmark() -> dict:
    layer = Linear(512, 512)
    x = Tensor([[1.0] * 512 for _ in range(32)])

    def case():
        _ = layer(x)

    result = benchmark_function(case, warmup=3, runs=10)
    result["case"] = "linear_forward"
    result["framework"] = "neural_kernel"
    result["batch_size"] = 32
    result["in_features"] = 512
    result["out_features"] = 512
    return result


def run_linear_backward_benchmark() -> dict:
    layer = Linear(512, 512)
    x = Tensor([[1.0] * 512 for _ in range(32)], requires_grad=True)

    def case():
        out = layer(x)
        loss = out.sum()
        layer.zero_grad()
        x.grad = None
        loss.backward()

    result = benchmark_function(case, warmup=3, runs=10)
    result["case"] = "linear_backward"
    result["framework"] = "neural_kernel"
    result["batch_size"] = 32
    result["in_features"] = 512
    result["out_features"] = 512
    return result