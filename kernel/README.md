# Neural Kernel — ML Core

Custom deep learning framework built from scratch. Autograd engine, neural network layers, optimizers, tokenization, and model training.

## Structure

```
kernel/
  autograd/     — Automatic differentiation engine (Tensor, backward pass)
  backend/      — Compute backends (CPU, optional CUDA)
  core/         — Core abstractions and utilities
  data/         — Data loading and batching
  loss/         — Loss functions (CrossEntropy, MSE, etc.)
  nn/           — Neural network layers (Linear, Conv, RNN, Transformer, etc.)
  optim/        — Optimizers (SGD, Adam, AdamW, etc.)
  tokenization/ — BPE tokenizer with merge caching
  utils/        — Helpers and logging
```

## Related directories

- `models/` — Pre-built model architectures (language models, tokenizers)
- `benchmarks/` — Performance benchmarks and regression tests
- `checkpoints/` — Saved model weights
- `data/corpus/` — Training data
- `tests_cuda/` — GPU-specific tests
- `tests_parity/` — Numerical parity tests against reference implementations

## Quick start

```bash
pip install -r kernel/requirements.txt
python -c "from kernel.nn import Linear; print(Linear(10, 5))"
```
