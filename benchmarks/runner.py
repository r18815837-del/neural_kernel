from __future__ import annotations

import json
from pathlib import Path

from benchmarks.cases.attention import (
    run_multihead_attention_backward_benchmark,
    run_multihead_attention_forward_benchmark,
)
from benchmarks.cases.layernorm import (
    run_layernorm_backward_benchmark,
    run_layernorm_forward_benchmark,
)
from benchmarks.cases.linear import (
    run_linear_backward_benchmark,
    run_linear_forward_benchmark,
)
from benchmarks.cases.tensor_ops import (
    run_tensor_add_benchmark,
    run_tensor_mul_benchmark,
)
from benchmarks.cases.transformer_block import (
    run_transformer_block_backward_benchmark,
    run_transformer_block_forward_benchmark,
)
from benchmarks.utils.env import get_environment_info


def run_all_benchmarks() -> dict:
    return {
        "environment": get_environment_info(),
        "results": [
            run_tensor_add_benchmark(),
            run_tensor_mul_benchmark(),
            run_linear_forward_benchmark(),
            run_linear_backward_benchmark(),
            run_layernorm_forward_benchmark(),
            run_layernorm_backward_benchmark(),
            run_multihead_attention_forward_benchmark(),
            run_multihead_attention_backward_benchmark(),
            run_transformer_block_forward_benchmark(),
            run_transformer_block_backward_benchmark(),
        ],
    }


def main() -> None:
    payload = run_all_benchmarks()

    results_dir = Path("benchmarks/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"\nSaved benchmark results to: {output_path}")


if __name__ == "__main__":
    main()