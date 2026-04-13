from __future__ import annotations

import numpy as np
import kernel as K

from benchmarks.utils.timing import benchmark_function


def run_transformer_block_forward_benchmark() -> dict:
    layer = K.TransformerBlock(d_model=128, num_heads=4, d_ff=256)
    x = K.Tensor(np.random.randn(8, 32, 128).astype(np.float64), requires_grad=False)

    def case():
        _ = layer(x)

    result = benchmark_function(case, warmup=3, runs=10)
    result["case"] = "transformer_block_forward"
    result["framework"] = "neural_kernel"
    result["batch_size"] = 8
    result["seq_len"] = 32
    result["d_model"] = 128
    result["num_heads"] = 4
    result["d_ff"] = 256
    return result


def run_transformer_block_backward_benchmark() -> dict:
    layer = K.TransformerBlock(d_model=128, num_heads=4, d_ff=256)
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
    result["case"] = "transformer_block_backward"
    result["framework"] = "neural_kernel"
    result["batch_size"] = 8
    result["seq_len"] = 32
    result["d_model"] = 128
    result["num_heads"] = 4
    result["d_ff"] = 256
    return result