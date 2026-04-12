# Architecture

This document gives a high-level overview of how Neural Kernel is structured internally.

The framework is organized as a compact but layered system:

- tensor core
- autograd
- neural network modules
- optimizers
- utilities
- transformer stack
- examples and tests

The design goal is to keep the internals understandable while still covering real deep-learning workflows.

---

## High-level layout

```text
neural_kernel/
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
├── examples/
├── tests/
└── tests_parity/

At a high level:

core/ owns the tensor object and low-level data behavior
autograd/ owns the backward graph logic and differentiable ops
nn/ owns layers, modules, normalization, attention, and models
optim/ owns parameter update rules and scheduling
utils/ owns checkpointing, metrics, plotting, seeding, and history
data/ owns datasets, collation, and data loading
tests/ validates framework behavior
tests_parity/ validates numerical agreement with PyTorch
Core tensor system
kernel/core/tensor.py

The Tensor is the central object in the framework.

It is responsible for:

holding array data
tracking gradients
participating in autograd
exposing tensor operations
carrying device information

Conceptually, the tensor layer provides the user-facing numerical object while the autograd layer provides the backward logic behind many operations.

Main responsibilities of Tensor
wraps numerical data
knows whether gradients are required
stores .grad
supports shape/view operations
supports movement between logical devices
interacts with backend abstractions
Backend abstraction
kernel/backend/

The backend layer exists so tensor computation is not hardwired to a single implementation forever.

Current status:

NumPy backend is the main implementation
CuPy backend exists as scaffold / future GPU path
device-aware API already exists at the tensor level

This gives the project a path toward broader CPU/GPU support without rewriting the full framework interface.

Autograd architecture
kernel/autograd/

Autograd provides the dynamic differentiation system.

Key ideas:

operations build graph relationships during forward execution
each node knows how to propagate gradients backward
Tensor.backward() triggers reverse traversal
gradients accumulate into tensors and parameters
kernel/autograd/ops/

The operation layer is split into thematic groups such as:

tensor ops
math ops
linalg ops
reduce ops
conv ops
pool ops
loss ops

This keeps operation logic modular instead of concentrating all backward rules in one file.

Design intent

The autograd engine is meant to be readable and explicit:

forward path computes values
backward path computes local gradients
gradients are propagated through the graph

This makes the project educational while still supporting meaningful models.

Module system
kernel/nn/module.py

The module system provides the structural abstraction used for layers and models.

A Module is responsible for:

parameter registration
submodule registration
buffer registration
mode switching via train() and eval()
serialization through state_dict() and load_state_dict()

This is the layer that turns raw differentiable functions into composable neural network objects.

Why it matters

Without Module, you can still write differentiable math.
With Module, you get:

reusable layers
nested models
optimizer-friendly parameter traversal
checkpointing support
train/eval mode control
Neural network organization

The nn/ package is split into several distinct layers.

kernel/nn/layers/

This contains standard architectural layers such as:

Linear
Conv2d
Embedding
Flatten
pooling layers
ResidualBlock

These are reusable building blocks.

kernel/nn/normalization.py

Normalization modules:

BatchNorm1d
BatchNorm2d
LayerNorm

These maintain running statistics or affine parameters depending on the module.

kernel/nn/activations.py

Activation modules:

ReLU
Sigmoid
Tanh
LeakyReLU
Identity
Softmax
kernel/nn/dropout.py

Dropout module with train() / eval() dependent behavior.

kernel/nn/containers.py

Container abstractions such as:

Sequential
kernel/nn/modules/

This is the higher-level model-components layer:

attention
multi-head attention
transformer blocks
encoder stacks
token classifiers
token language models

This split is important:

layers/ holds standard reusable pieces
modules/ holds higher-order architecture pieces
kernel/nn/functional/

This contains stateless helpers and low-level functional APIs such as:

scaled_dot_product_attention
make_causal_mask
make_padding_mask

This is useful both for internal composition and for lower-level testing.

Transformer architecture

The Transformer path is one of the strongest parts of the project.

Main building blocks
ScaledDotProductAttention
MultiHeadAttention
PositionalEncoding
FeedForward
TransformerBlock
TransformerEncoder
Design pattern

The Transformer stack is built hierarchically:

low-level attention math
multi-head self-attention
feed-forward block
normalization + residual logic
repeated encoder stack
task-specific classifier or LM heads
Token models

On top of the encoder stack, the project includes:

TransformerEncoderClassifier
TokenTransformerClassifier
TokenTransformerLM

This gives the framework:

sequence classification
token-based classification
autoregressive LM-style generation
Generation

TokenTransformerLM supports:

greedy decoding
temperature sampling
top-k sampling
top-p sampling

This pushes the framework beyond static forward inference and into actual sequence generation workflows.

CNN path

The project also has a full CNN path with:

Conv2d
pooling
normalization
residual connections
classifier pipelines

This makes the framework broader than a transformer-only experiment.

The CNN side and transformer side share:

the same tensor system
the same autograd engine
the same module and optimizer infrastructure

That reuse is a key architectural strength.

Optimizer stack
kernel/optim/

The optimizer package handles parameter updates.

Current components:

Optimizer
SGD
Adam
StepLR

Responsibilities include:

iterating parameters
reading gradients
applying update rules
clearing gradients
scheduler-driven learning-rate updates

This layer sits above autograd and below training scripts.

Loss organization

Losses appear in two related places:

lower-level differentiable loss operations in autograd/loss logic
user-facing module-style loss objects such as CrossEntropyLoss

This split keeps:

gradient math inside the differentiable engine
training-facing ergonomics inside the module/loss API
Data pipeline
kernel/data/

The data package provides:

dataset abstractions
batching
collation
dataloader iteration

This allows end-to-end examples to stay within the framework ecosystem rather than relying only on ad hoc NumPy loops.

Utilities
kernel/utils/

This package contains project-level support tools:

checkpoint save/load
History
set_seed
metrics
plotting helpers

These pieces are not part of the math core, but they matter for real training workflows.

Checkpointing

Checkpointing allows:

saving model state
loading model state
restoring forward behavior
resume flows

This is one of the features that pushes the project toward framework-style maturity.

Public API philosophy

The project uses a layered API strategy:

Root-level API

A compact surface for convenient imports.

Canonical import paths

The most explicit and stable imports come from subpackages such as:

kernel.core.tensor
kernel.nn.layers
kernel.nn.modules
kernel.optim
kernel.utils

This allows the internal structure to stay understandable while still supporting a user-facing surface.

Testing architecture

The testing strategy is part of the architecture, not an afterthought.

Regression suite

tests/ verifies:

tensor behavior
gradients
shape contracts
training/eval behavior
checkpoint behavior
generation behavior
transformer stack behavior
Parity suite

tests_parity/ verifies numerical agreement with PyTorch for key blocks:

Linear
CrossEntropyLoss
LayerNorm
Embedding
attention modules
transformer block
encoder
Smoke tests

Smoke tests validate:

public imports
canonical imports
API surface stability

Together these form a layered trust model:

regression protects behavior
parity protects correctness against a reference
smoke protects the user-facing package surface
Example workflow through the stack

A typical training step flows through the architecture like this:

input data is wrapped as Tensor
model modules run forward through layers and submodules
losses produce a scalar objective
autograd records graph relationships
backward() computes gradients
optimizer reads gradients and updates parameters
utilities may track metrics or save checkpoints

For a transformer LM generation workflow:

token IDs go into Embedding
encoded sequence passes through Transformer blocks
logits are projected to vocabulary space
generation loop selects next token
tokens are appended and decoding continues
Design strengths

At the current stage, the strongest architectural properties are:

one shared tensor/autograd foundation for CNN and Transformer paths
clear separation between low-level ops and high-level modules
module-based parameter and buffer management
support for both training and generation workflows
regression + parity validation
clean enough boundaries for future packaging and CI
Current limitations

Some areas are still intentionally lightweight or evolving:

root-level exports may still be narrower than canonical subpackage usage
optimizer checkpoint serialization is only partially generalized
CUDA parity has not yet been fully validated in a GPU-enabled environment
packaging and CI are still a future step

These are maturity gaps, not structural dead-ends.

Summary

Neural Kernel is built around a simple but strong idea:

a custom tensor object
a readable autograd engine
a module system for composition
reusable CNN and Transformer components
optimizer and checkpoint support
layered validation through regression and parity tests

That combination is what turns the project from a collection of experiments into a real mini-framework.