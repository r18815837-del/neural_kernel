from __future__ import annotations

import numpy as np
import kernel as K

from benchmarks.utils.timing import benchmark_function


def run_multihead_attention_forward_benchmark() -> dict:
    layer = K.MultiHeadAttention(d_model=128, num_heads=4)
    x = K.Tensor(np.random.randn(8, 32, 128).astype(np.float64), requires_grad=False)

    def case():
        _ = layer(x)

    result = benchmark_function(case, warmup=3, runs=10)
    result["case"] = "multihead_attention_forward"
    result["framework"] = "neural_kernel"
    result["batch_size"] = 8
    result["seq_len"] = 32
    result["d_model"] = 128
    result["num_heads"] = 4
    return result


def run_multihead_attention_backward_benchmark() -> dict:
    layer = K.MultiHeadAttention(d_model=128, num_heads=4)
    x = K.Tensor(np.random.randn(8, 32, 128).astype(np.float64), requires_grad=True)

    def case():
        out = layer(x)
        if isinstance(out, tuple):
            out = out[0]
        loss = out.sum()
        layer.zero_grad()
        x.grad = None
        loss.backward()

    result = benchmark_function(case, warmup=3, runs=10)
    result["case"] = "multihead_attention_backward"
    result["framework"] = "neural_kernel"
    result["batch_size"] = 8
    result["seq_len"] = 32
    result["d_model"] = 128
    result["num_heads"] = 4
    return result