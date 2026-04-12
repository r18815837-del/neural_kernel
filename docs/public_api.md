# Public API

This document describes the recommended import paths and the intended public surface of Neural Kernel.

The project includes both:
- a root package API
- canonical subpackage import paths

In practice, the most stable and explicit imports are usually the subpackage-level imports shown below.

---

## Recommended import style

For framework usage in examples and training scripts, the safest approach is:

```python id="p0lxh5"
from kernel.core.tensor import Tensor
from kernel.nn import CrossEntropyLoss
from kernel.optim import Adam
from kernel.utils import set_seed

For module-based work:

from kernel.nn.modules import (
    MultiHeadAttention,
    TransformerBlock,
    TransformerEncoder,
    TransformerEncoderClassifier,
    TokenTransformerClassifier,
    TokenTransformerLM,
)

For standard layers:

from kernel.nn.layers import (
    Linear,
    Conv2d,
    Flatten,
    Embedding,
    MaxPool2d,
    AvgPool2d,
    AdaptiveAvgPool2d,
    AdaptiveMaxPool2d,
    ResidualBlock,
)

For normalization and activations:

from kernel.nn.normalization import BatchNorm1d, BatchNorm2d, LayerNorm
from kernel.nn.activations import ReLU, Sigmoid, Tanh, LeakyReLU, Identity, Softmax
from kernel.nn.dropout import Dropout

For container modules:

from kernel.nn.containers import Sequential
from kernel.nn.modules import ModuleList, ModuleDict
Root-level API

The root package may expose a compact import layer, but examples and internal docs should prefer canonical paths when clarity matters.

Example root import style:

import kernel as K

Depending on the exact export surface, this may be useful for compact demos.
However, canonical imports are often preferable for:

examples
tests
documentation
avoiding ambiguity while the API surface evolves
Core API
Tensor

Canonical import:

from kernel.core.tensor import Tensor

Main capabilities:

data storage
autograd tracking
device-aware behavior
tensor operations
reductions
view/shape manipulation

Common methods and behavior include:

.backward()
.to(device)
.cpu()
.cuda()
.detach()
Neural network API
Base module

Canonical import:

from kernel.nn.module import Module

Core behaviors:

parameter registration
submodule registration
buffers
parameters()
named_parameters()
zero_grad()
train()
eval()
state_dict()
load_state_dict()
Common module-level imports

Recommended public layer:

from kernel.nn import CrossEntropyLoss

Other common imports often used directly from their canonical locations:

BatchNorm1d
BatchNorm2d
LayerNorm
Dropout
Sequential
Layers API

Canonical path:

from kernel.nn.layers import ...

Main layer exports include:

Linear
Embedding
Conv2d
Flatten
MaxPool2d
AvgPool2d
AdaptiveAvgPool2d
AdaptiveMaxPool2d
ResidualBlock

Example:

from kernel.nn.layers import Linear, Embedding, Conv2d
Transformer API

Canonical path:

from kernel.nn.modules import ...

Main Transformer-related exports:

ScaledDotProductAttention
MultiHeadAttention
PositionalEncoding
FeedForward
TransformerBlock
TransformerEncoder
TransformerEncoderClassifier
TokenTransformerClassifier
TokenTransformerLM

Example:

from kernel.nn.modules import TransformerBlock, TransformerEncoder, TokenTransformerLM
Functional API

Canonical path:

from kernel.nn.functional import ...

Main functional exports include:

scaled_dot_product_attention
make_causal_mask
make_padding_mask

Example:

from kernel.nn.functional import scaled_dot_product_attention, make_causal_mask, make_padding_mask
Optimizer API

Canonical path:

from kernel.optim import Optimizer, SGD, Adam, StepLR

Main optimizer-related exports:

Optimizer
SGD
Adam
StepLR

Example:

from kernel.optim import Adam
Utilities API

Canonical path:

from kernel.utils import ...

Main utility exports:

save_checkpoint
load_checkpoint
History
set_seed
accuracy
mse
plot_history

Example:

from kernel.utils import save_checkpoint, load_checkpoint, set_seed
Data API

Canonical path:

from kernel.data import ...

Typical exports may include:

Dataset
TensorDataset
DataLoader
default_collate

Example:

from kernel.data import DataLoader
Example import patterns
Minimal MLP script
from kernel.core.tensor import Tensor
from kernel.nn.activations import ReLU
from kernel.nn.containers import Sequential
from kernel.nn.layers import Linear

model = Sequential(
    Linear(4, 16),
    ReLU(),
    Linear(16, 3),
)

x = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
y = model(x)
Transformer classifier
from kernel.core.tensor import Tensor
from kernel.nn import CrossEntropyLoss
from kernel.nn.modules import TokenTransformerClassifier
from kernel.optim import Adam
from kernel.utils import set_seed
Language model generation
from kernel.core.tensor import Tensor
from kernel.nn.modules import TokenTransformerLM
from kernel.utils import set_seed
Public API design notes

Neural Kernel uses a layered public API approach:

Root-level surface

Useful for compact demos, but may be intentionally small or evolving.

Canonical import paths

Preferred when:

writing examples
writing tests
writing documentation
debugging
keeping imports explicit

This keeps the project easier to maintain while still allowing a clean user-facing surface.

Public API validation

The public surface is validated with:

smoke import tests
canonical import tests
runnable examples

Examples:

smoke_public_api_imports.py
smoke_canonical_imports.py

These tests help ensure that:

exported names resolve correctly
import paths remain stable
examples use valid public entry points
Recommended rule of thumb

Use:

root imports for compact demo-style usage
subpackage canonical imports for examples, tests, and docs

If an example is meant to be robust and explicit, prefer canonical imports.

If an example is meant to be very short and user-facing, root imports are acceptable if those names are exported at the root.