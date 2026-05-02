from __future__ import annotations

import kernel as K

from benchmarks.utils.timing import benchmark_function


def run_layernorm_forward_benchmark() -> dict:
    layer = K.LayerNorm(512)
    x = K.Tensor([[1.0] * 512 for _ in range(32)])

    def case():
        _ = layer(x)

    result = benchmark_function(case, warmup=3, runs=10)
    result["case"] = "layernorm_forward"
    result["framework"] = "neural_kernel"
    result["batch_size"] = 32
    result["normalized_shape"] = 512
    return result


def run_layernorm_backward_benchmark() -> dict:
    layer = K.LayerNorm(512)
    x = K.Tensor([[1.0] * 512 for _ in range(32)], requires_grad=True)

    def case():
        out = layer(x)
        loss = out.sum()
        layer.zero_grad()
        x.grad = None
        loss.backward()

    result = benchmark_function(case, warmup=3, runs=10)
    result["case"] = "layernorm_backward"
    result["framework"] = "neural_kernel"
    result["batch_size"] = 32
    result["normalized_shape"] = 512
    return result