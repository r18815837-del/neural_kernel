# Neural Kernel

A learning project — a small deep learning framework written from scratch in Python. Nothing production-grade, just an exercise in understanding how things work under the hood: autograd, backpropagation, transformers, tokenizers, and all that.

No PyTorch, no TensorFlow — just NumPy (and optionally CuPy for GPU).

## What's here

**Core framework** (`kernel/`) — tensors with autograd, basic layers (Linear, Conv2d, Embedding, BatchNorm, LayerNorm, Dropout), a simple Transformer stack, BPE tokenizer, SGD/Adam optimizers, and a data loader. Nothing fancy, but it works and the gradients are correct (checked against PyTorch).

**BPE tokenizer** (`kernel/tokenization/`) — byte-pair encoding trained from scratch. GPT-2 style pre-tokenization. Saves to JSON.

**Training scripts** (`scripts/`) — download Wikipedia articles, train BPE, train a small language model. Can run on CPU or GPU.

**Cognition engine** (`cognition/`) — an experimental Q&A pipeline with memory, Wikipedia search, a coding assistant that can analyze Python/Dart code, and a sandboxed code executor. More of a playground than anything serious.

**REST API** (`api/`) — FastAPI server with JWT auth, rate limiting, error handling. Endpoints for the cognition engine, code analysis, project generation, session history.

**Persistence** (`persistence/`) — SQLite storage for sessions, memory, projects, artifacts. Thread-safe, with migrations.

**Flutter client** (`flutter_client/`) — a simple mobile/desktop UI. Riverpod + GoRouter.

## Quick start

```bash
git clone https://github.com/r18815837-del/neural_kernel.git
cd neural_kernel
pip install -e .[dev]

# Run tests
pytest -q tests

# Train a small LM
python scripts/prepare_data.py --max-articles 2000
python scripts/train_lm.py --epochs 10

# GPU training (needs CuPy + CUDA)
python scripts/train_lm.py --device cuda --batch-size 64 --epochs 20

# Start the API
python run_api.py

# Or with Docker
docker compose up --build
```

## Project structure

```
kernel/              Core framework (tensors, autograd, layers, optimizers)
cognition/           Q&A engine, memory, specialists, code executor
persistence/         SQLite storage, migrations, lifecycle
api/                 FastAPI server, auth, middleware
flutter_client/      Mobile/desktop client
scripts/             Data prep + training
tests/               ~250 tests
tests_parity/        PyTorch parity checks
```

## Some numbers

These are modest — it's a learning project, not a competitor to anything:

- MLP on MNIST: ~98% accuracy
- CNN on MNIST: ~94% accuracy
- Language model: 1.85M params, 4-layer Transformer, loss 7.27 → 5.79 over 10 epochs on CPU
- ~250 tests (core, cognition, persistence, API, auth)
- PyTorch parity tests pass for the main ops

## Requirements

- Python 3.10+
- NumPy
- For API: `pip install -r requirements.txt`
- For GPU: CuPy + CUDA 12.x
- For Flutter client: Flutter SDK 3.x

## License

MIT

## Author

Raian
