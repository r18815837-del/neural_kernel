from __future__ import annotations

from kernel.core.tensor import Tensor

from benchmarks.utils.timing import benchmark_function


def run_tensor_add_benchmark() -> dict:
    a = Tensor([[1.0] * 512 for _ in range(512)])
    b = Tensor([[2.0] * 512 for _ in range(512)])

    def case():
        _ = a + b

    result = benchmark_function(case, warmup=3, runs=10)
    result["case"] = "tensor_add"
    result["framework"] = "neural_kernel"
    result["shape"] = [512, 512]
    return result


def run_tensor_mul_benchmark() -> dict:
    a = Tensor([[1.0] * 512 for _ in range(512)])
    b = Tensor([[2.0] * 512 for _ in range(512)])

    def case():
        _ = a * b

    result = benchmark_function(case, warmup=3, runs=10)
    result["case"] = "tensor_mul"
    result["framework"] = "neural_kernel"
    result["shape"] = [512, 512]
    return result