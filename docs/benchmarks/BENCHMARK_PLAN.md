# Benchmark Plan

## Purpose
This document defines the benchmark foundation for `neural_kernel`.

The goal is not to produce flattering numbers, but to produce honest, reproducible, technically defensible measurements for core execution paths.

## Benchmark Goals

### Primary Goals
- Measure core execution paths of the framework
- Compare `neural_kernel` against PyTorch where appropriate
- Cover both CPU and CUDA execution paths
- Make benchmark runs reproducible and extensible

### Secondary Goals
- Identify obvious performance bottlenecks
- Provide a public benchmark structure for future release notes
- Support future optimization work with consistent measurement methodology

## Non-Goals
- Winning every comparison against PyTorch
- Exhaustive coverage of all possible operations in Sprint 1
- Publishing polished benchmark tables in Sprint 1
- Micro-optimizing benchmark cases before methodology is stable

---

## Benchmark Scope

### Initial Benchmark Areas
The first benchmark suite should cover the most important and representative framework paths.

#### 1. Tensor Ops
Examples:
- elementwise add
- elementwise multiply
- matmul
- reduction ops where relevant

#### 2. Linear Layer
- forward
- backward

#### 3. LayerNorm
- forward
- backward

#### 4. Attention
- scaled dot-product attention forward
- scaled dot-product attention backward

#### 5. Transformer Block
- forward
- backward

#### 6. Tiny Generation Loop
- autoregressive token generation loop
- representative small decoding run

---

## Comparison Targets

### CPU
- `neural_kernel` CPU vs PyTorch CPU

### CUDA
- `neural_kernel` CUDA vs PyTorch CUDA

### Internal Comparisons
- CPU vs CUDA for selected cases
- small vs medium vs larger shapes where relevant

> The benchmark suite should remain fair and transparent. Results must clearly describe shape choices, batch sizes, device, and run conditions.

---

## Benchmark Principles

### Principle 1 — Reproducibility
A benchmark result is useful only if it can be rerun and interpreted under known conditions.

### Principle 2 — Fairness
Comparison setups should be as equivalent as possible in:
- tensor shapes
- dtypes
- device
- warmup behavior
- number of timed iterations

### Principle 3 — Transparency
All benchmark outputs should include enough metadata to explain:
- environment
- device
- framework versions
- benchmark case parameters

### Principle 4 — Separation of Infrastructure and Results
Sprint 1 builds the benchmark framework.
Sprint 2 produces and publishes benchmark results.

---

## Benchmark Methodology

### General Approach
For each benchmark case:
1. Construct input tensors/modules
2. Run warmup iterations
3. Run timed iterations
4. Record timing result
5. Export result with metadata

### Timing Strategy
- Use warmup runs before timing
- Use multiple timed runs
- Prefer median reporting for robustness
- Report iteration count
- Record benchmark parameters explicitly

### CUDA Timing Rules
- Synchronize before ending timing windows
- Avoid accidental asynchronous timing artifacts
- Ensure setup overhead is not mixed with execution timing unless explicitly intended

### Reporting
Each result should include:
- benchmark name
- framework
- device
- dtype if relevant
- shape/config parameters
- number of warmup runs
- number of timed runs
- timing statistic(s)

---

## Environment Capture

Each benchmark run should record environment metadata, including as many of the following as available:

- Python version
- Operating system
- CPU information
- NumPy version
- CuPy version
- CUDA availability
- CUDA version if accessible
- GPU device name if accessible
- PyTorch version
- `neural_kernel` version or git commit where possible

This metadata should be included in machine-readable output.

---

## Output Format

### Required Outputs
- JSON output for machine-readable results
- Markdown summary for human-readable inspection

### Recommended JSON Shape
A typical benchmark result structure should contain:

- suite metadata
- environment metadata
- benchmark cases
- per-case timing results

The exact schema can evolve, but must remain stable once public reporting begins.

---

## Suggested Directory Structure

```text
benchmarks/
  __init__.py
  runner.py
  cases/
    tensor_ops.py
    linear.py
    layernorm.py
    attention.py
    transformer_block.py
    generation.py
  utils/
    env.py
    timing.py
    reporting.py
  results/

This is only a suggested structure and may be adapted to the current repository layout.

Initial Benchmark Case Ideas
Tensor Ops
add on medium tensor
mul on medium tensor
matmul on representative matrix shapes
Linear
batch x feature forward
batch x feature backward
LayerNorm
standard normalized hidden dimension
backward included
Attention
representative batch / heads / seq / head_dim configuration
forward and backward separately or combined depending on runner design
Transformer Block
representative hidden size
representative sequence length
forward and backward path
Tiny Generation
decode loop over N tokens
small transformer language model configuration
timing focused on practical generation path
Parameter Strategy

The benchmark suite should define stable representative configurations rather than unlimited ad hoc shapes.

Suggested categories:

small
medium
large

This allows future benchmarking to scale without losing comparability.

Sprint 1 Deliverables

By the end of Sprint 1, the benchmark effort must produce:

benchmark directory scaffold
benchmark runner skeleton
initial case definitions
timing utilities
environment capture helper
methodology note
placeholder output path

Sprint 1 does not require:

polished benchmark charts
final public benchmark results
exhaustive case coverage
Sprint 2 Follow-Up

Sprint 2 will build on this foundation to:

run benchmark cases
collect results
verify fairness
prepare public benchmark section
integrate benchmark findings into docs
Quality Bar

A benchmark setup is acceptable only if it is:

reproducible
explicit about conditions
reasonably fair
easy to rerun
easy to extend

If a benchmark looks impressive but is not technically defensible, it should not be published.

Summary

The benchmark suite exists to make neural_kernel measurable, not merely presentable.

The end goal is a benchmark system that supports:

engineering iteration
honest public reporting
future optimization work
credibility with users and contributors

Теперь четвёртый файл: `EXAMPLES_ROADMAP.md`

```md
# Examples Roadmap

## Purpose
This document defines the flagship examples that will represent `neural_kernel` publicly in the next development phase.

The goal is to choose examples that best demonstrate the current strengths of the framework while also supporting future product and startup positioning.

## Selected Flagship Examples

The two flagship examples are:

1. `tiny_gpt`
2. `text_classification`

These are intentionally selected over a CNN-first showcase because together they better reflect:
- the current strength of the framework
- the public demo value of the project
- the future direction toward AI systems and assistants

---

## Example 1 — `tiny_gpt`

### Purpose
Demonstrate the language-modeling and generation capabilities of `neural_kernel`.

### Why It Matters
This example showcases:
- Transformer / LM path
- token generation flow
- greedy / temperature / top-k / top-p decoding
- checkpointing/resume potential
- practical sequence modeling

It is the strongest public-facing showcase for the current framework stack.

### What It Should Demonstrate
- model definition
- training loop
- checkpoint save/load
- text generation after training
- configurable decoding strategies
- basic evaluation or sample output inspection

### Minimum Deliverables
- `train.py`
- `generate.py`
- `README.md`

### Preferred Deliverables
- `eval.py`
- config file or simple argument interface
- example checkpoint usage notes
- sample outputs in docs

### Technical Themes
- Transformer stack correctness
- LM head usage
- generation utilities
- checkpoint integration

### Public Value
This example helps communicate that `neural_kernel` is not just a tensor/autograd project, but a usable mini deep learning framework with real LM capability.

---

## Example 2 — `text_classification`

### Purpose
Demonstrate a practical supervised learning workflow using `neural_kernel`.

### Why It Matters
This example showcases:
- end-to-end training
- evaluation flow
- inference flow
- practical business-facing use case
- approachable entry point for users

It balances the more impressive LM demo with a clean, understandable applied ML example.

### What It Should Demonstrate
- dataset preparation
- model training
- validation/evaluation
- simple inference path
- metric reporting
- checkpoint usage if practical

### Minimum Deliverables
- `train.py`
- `eval.py`
- `predict.py`
- `README.md`

### Preferred Deliverables
- simple config path
- expected metrics or expected output format
- example input/output samples
- checkpoint load example

### Technical Themes
- embeddings and/or encoder usage depending on architecture choice
- training/evaluation lifecycle
- practical loss optimization
- inference API clarity

### Public Value
This example helps show that `neural_kernel` can support real applied ML tasks, not only low-level framework mechanics.

---

## Why These Two Were Chosen

### `tiny_gpt`
Chosen because it highlights:
- current transformer maturity
- generation support already implemented
- strongest technical demo value
- strongest startup-facing narrative

### `text_classification`
Chosen because it highlights:
- usability
- practical supervised ML flow
- clear developer onboarding path
- easy communication to non-research users

Together they provide both:
- technical depth
- practical accessibility

---

## Why CNN Is Not a Flagship Example Right Now

CNN support is valuable and should remain part of the project, but it is not the best top-level showcase at this stage.
