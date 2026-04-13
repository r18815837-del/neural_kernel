# Neural Kernel

A mini deep learning framework with autograd, NumPy/CuPy backends, CPU/CUDA execution, CNN support, Transformer modules, language-model generation, and checkpointed training.

Neural Kernel is built from scratch to make deep learning internals understandable without giving up structure, testing discipline, or practical usability.

## Highlights

- Tensor + autograd engine
- NumPy backend and CuPy CUDA path
- Device-aware tensor API
- CNN stack: convolutions, pooling, normalization, residual block
- Transformer stack: attention, multi-head attention, encoder, classifier, LM
- Generation: greedy, temperature, top-k, top-p
- Checkpointing + resume
- PyTorch parity coverage for key modules
- Real CUDA validation on local NVIDIA GPU

---

## Features

### Core
- `Tensor`
- automatic differentiation
- dynamic computation graph
- `backward()`
- gradient accumulation
- device-aware tensor handling
- `Tensor.to(device)`, `.cpu()`, `.cuda()`, `.detach()`

### Backend
- NumPy backend
- CuPy backend
- backend abstraction layer

### Tensor ops
- `reshape`
- `transpose`
- `permute`
- `unsqueeze`
- `squeeze`
- `getitem`
- `concat`
- `masked_fill`

### Reductions
- `sum(axis, keepdims)`
- `mean(axis, keepdims)`

### Math / activations / loss
- `relu`
- `sigmoid`
- `tanh`
- `leaky_relu`
- `softmax`
- `gelu`
- `MSELoss`
- `CrossEntropyLoss`

### Neural network API
- `Module`
- parameter registration
- submodule registration
- buffers
- `parameters()`
- `named_parameters()`
- `zero_grad()`
- `train()` / `eval()`
- `state_dict()` / `load_state_dict()`

### Layers
- `Linear`
- `Embedding`
- `Conv2d`
- `Flatten`
- `Dropout`
- `ResidualBlock`

### Pooling
- `MaxPool2d`
- `AvgPool2d`
- `AdaptiveAvgPool2d`
- `AdaptiveMaxPool2d`

### Activations
- `ReLU`
- `Sigmoid`
- `Tanh`
- `LeakyReLU`
- `Identity`
- `Softmax`

### Normalization
- `BatchNorm1d`
- `BatchNorm2d`
- `LayerNorm`

### Containers
- `Sequential`
- `ModuleList`
- `ModuleDict`

### Transformer modules
- `ScaledDotProductAttention`
- `MultiHeadAttention`
- `PositionalEncoding`
- `FeedForward`
- `TransformerBlock`
- `TransformerEncoder`

### Models
- `TransformerEncoderClassifier`
- `TokenTransformerClassifier`
- `TokenTransformerLM`

### Token / sequence utilities
- causal mask
- padding mask
- mean pooling
- CLS pooling
- last-token pooling
- learnable CLS token
- tied embeddings for LM

### Generation
- greedy decoding
- temperature sampling
- top-k sampling
- top-p sampling

### Optimizers and scheduling
- `SGD`
- `Adam`
- `StepLR`

### Data utilities
- `Dataset`
- `TensorDataset`
- `DataLoader`
- `default_collate`

### Utilities
- checkpoint save/load
- training `History`
- deterministic `set_seed`
- metrics: `accuracy`, `mse`
- plotting training curves

---

## Project structure

```text
neural_kernel/
├── docs/
│   ├── architecture.md
│   ├── public_api.md
│   └── testing.md
├── examples/
│   ├── 01_linear_regression.py
│   ├── 02_digits_mlp.py
│   ├── 03_mnist_mlp.py
│   ├── 04_mnist_inference.py
│   ├── 05_mnist_cnn.py
│   ├── 06_mnist_cnn_inference.py
│   ├── 07_transformer_classifier.py
│   ├── 08_token_lm_generate.py
│   └── 09_checkpoint_resume.py
├── kernel/
│   ├── autograd/
│   ├── backend/
│   ├── core/
│   ├── data/
│   ├── loss/
│   ├── nn/
│   │   ├── functional/
│   │   ├── layers/
│   │   └── modules/
│   ├── optim/
│   └── utils/
├── tests/
├── tests_cuda/
├── tests_parity/
├── smoke_public_api_imports.py
├── smoke_canonical_imports.py
├── pyproject.toml
└── README.md

kernel/ contains the framework implementation
tests/ contains the regression suite
tests_parity/ contains PyTorch reference parity tests
tests_cuda/ contains CPU vs CUDA parity tests
examples/ contains runnable demos
docs/ contains project documentation
Installation

Clone the repository and install dependencies:

git clone <your-repo-url>
cd neural_kernel
pip install -e .[dev]

If you only want the base package:

pip install -e .
Quick start
Build a small MLP
import kernel as K

model = K.Sequential(
    K.Linear(4, 16),
    K.ReLU(),
    K.Linear(16, 3),
)

x = K.Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
y = model(x)

print(y.data)
Save and load a checkpoint
import kernel as K

K.save_checkpoint(model, "model.pkl")
K.load_checkpoint(model, "model.pkl")
Use training history
import kernel as K

history = K.History()
history.log(epoch=1, train_loss=0.5, train_acc=0.90, test_acc=0.88, lr=0.001)
print(history.as_dict())
Transformer / LM example
import numpy as np
import kernel as K

model = K.TokenTransformerLM(
    vocab_size=30,
    d_model=32,
    num_heads=4,
    d_ff=64,
    num_layers=2,
    max_len=32,
    tie_embeddings=True,
)

tokens = K.Tensor(np.array([[1, 2, 3, 4]], dtype=np.int64))
logits, attn = model(tokens)

generated = model.generate(
    tokens,
    max_new_tokens=8,
    temperature=1.0,
    top_k=5,
    do_sample=True,
)

print(logits.shape)
print(generated)
Examples
Classical / CNN examples
01_linear_regression.py
02_digits_mlp.py
03_mnist_mlp.py
04_mnist_inference.py
05_mnist_cnn.py
06_mnist_cnn_inference.py
Transformer / LM examples
07_transformer_classifier.py
08_token_lm_generate.py
09_checkpoint_resume.py
Run examples
python examples/07_transformer_classifier.py
python examples/08_token_lm_generate.py
python examples/09_checkpoint_resume.py
Testing
Run the main regression suite
pytest -q tests
Run the PyTorch parity suite
pytest -q tests_parity
Run the CUDA parity suite
pytest -q tests_cuda
Run smoke tests
pytest -q smoke_public_api_imports.py smoke_canonical_imports.py
Run everything
pytest -q tests tests_parity tests_cuda smoke_public_api_imports.py smoke_canonical_imports.py
Validation status
Regression suite

Current status:

216 passed
1 skipped
Public API

Validated with:

public API smoke imports
canonical import usage
Numerical parity with PyTorch

Validated for:

Linear
CrossEntropyLoss
LayerNorm
Embedding
ScaledDotProductAttention
MultiHeadAttention
TransformerBlock
TransformerEncoder
CUDA status

CUDA support has been validated in a real local NVIDIA GPU environment.

Confirmed:

CuPy backend is working on Windows
tensor CPU ↔ CUDA transfer works
CUDA parity is green for:
tensor ops
Linear
MultiHeadAttention
TransformerBlock
TransformerEncoder

CUDA validation was run with:

NVIDIA GeForce RTX 4060
CUDA Toolkit 12.4
CuPy 13.4.1

This means CUDA is no longer just scaffolded at the API level — key execution paths are now verified in practice.

Results so far
MLP on MNIST
best test accuracy: ~0.9807
CNN experiments on MNIST
CNN + BatchNorm2d + MaxPool2d: 0.9340
CNN + BatchNorm2d + AvgPool2d: 0.9280
CNN without BatchNorm2d: 0.9120
deeper CNN baseline: 0.9440
Transformer synthetic classifier demo

The 07_transformer_classifier.py example trains successfully and reaches strong validation accuracy on the synthetic task.

Token LM generation demo

The 08_token_lm_generate.py example successfully trains and generates the expected continuation on the toy autoregressive task.

Current best CNN architecture
Sequential(
    Conv2d(1, 8, 3, padding=1),
    BatchNorm2d(8),
    ReLU(),
    MaxPool2d(2),

    Conv2d(8, 16, 3, padding=1),
    BatchNorm2d(16),
    ReLU(),
    MaxPool2d(2),

    Conv2d(16, 32, 3, padding=1),
    BatchNorm2d(32),
    ReLU(),

    Flatten(),
    Linear(32 * 7 * 7, 128),
    ReLU(),
    Linear(128, 10),
)
Why this project exists

This project was built to deeply understand:

how autograd works
how tensors propagate gradients
how layers and parameters are structured
how normalization and residual connections behave
how attention and Transformer blocks are implemented
how checkpoints and training loops work internally
how framework internals compare numerically to PyTorch
how CPU and CUDA execution paths behave in practice

Instead of treating deep learning frameworks as black boxes, Neural Kernel rebuilds the core pieces from first principles.

What was validated

This is not forward-only code. The project was validated through:

regression tests
parity tests against PyTorch
CUDA parity tests
end-to-end training runs
checkpoint reload tests
train() / eval() behavior tests
normalization running-stat tests
Transformer and LM path tests
generation tests
public API import tests
runnable examples
Current maturity

Neural Kernel is now a structured mini-framework with:

core autograd
NumPy backend
working CuPy CUDA path
CNN support
Transformer encoder stack
token classification and LM path
generation utilities
checkpointing and resume
clean public API
regression suite
PyTorch parity on key components
CUDA parity on key execution paths

This is no longer just a toy script. It is a compact framework project with meaningful architectural depth and validation.

Roadmap
Completed
tensor + autograd engine
linear and convolution layers
pooling and adaptive pooling
normalization layers
activations
optimizers and scheduler
checkpointing
training history / seed / plots
MLP and CNN experiments
residual block
Transformer attention stack
encoder / classifier / LM path
generation
public API cleanup
regression suite
parity tests for key modules
CUDA validation for core execution paths
Next possible steps
optimizer state_dict() / load_state_dict() completion everywhere
broader CUDA coverage beyond current validated set
richer example set
packaging polish
GitHub Actions CI finalization
release flow
richer model zoo
more parity coverage
benchmark comparisons
Documentation

Additional docs live in:

docs/testing.md
docs/public_api.md
docs/architecture.md
License

MIT

Author

Built by [Your Name / GitHub handle].