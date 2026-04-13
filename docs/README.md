## Benchmarks

Preliminary benchmark coverage currently includes:

- tensor add / mul
- linear forward / backward
- layernorm forward / backward
- multi-head attention forward / backward
- transformer block forward / backward

Benchmarks are collected with a reproducible runner that records:
- environment metadata
- warmup count
- timed runs
- mean / median / min / max timings

Raw outputs are written to:

```text
benchmarks/results/benchmark_results.json