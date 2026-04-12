# Testing

This project is validated with multiple layers of tests:

- regression tests
- smoke tests
- public API import tests
- canonical import tests
- numerical parity tests against PyTorch
- example script runs

The goal is not only to check that code runs, but also to verify shapes, gradients, values, API contracts, checkpoint behavior, and consistency with reference implementations.

---

## Test layout

```text
tests/
tests_parity/
smoke_public_api_imports.py
smoke_canonical_imports.py

tests/

Main regression suite for framework behavior:

tensor ops
shape ops
reductions
autograd
layers
normalization
containers
optimizers
checkpointing
masks
attention
transformer blocks
encoder stack
token models
generation
tests_parity/

Reference tests against PyTorch for key components:

Linear
CrossEntropyLoss
LayerNorm
Embedding
ScaledDotProductAttention
MultiHeadAttention
TransformerBlock
TransformerEncoder
Smoke tests

Smoke files validate:

public import surface
canonical usage patterns
root-level API behavior
Running tests
Run the main regression suite
pytest -q tests
Run the parity suite
pytest -q tests_parity
Run the smoke tests
pytest -q smoke_public_api_imports.py smoke_canonical_imports.py
Run everything
pytest -q tests tests_parity smoke_public_api_imports.py smoke_canonical_imports.py
Current status

At the current project stage, the regression suite is green:

216 passed
1 skipped

The skipped test is an expected partial case related to optimizer checkpoint serialization support.

Parity tests against PyTorch are green for the core reference modules listed above.

What is covered
Core tensor behavior

Validated for:

tensor creation
arithmetic ops
indexing
reshape / transpose / permute
unsqueeze / squeeze
concat
masked fill
reductions
Autograd

Validated for:

scalar backward paths
gradient propagation
input gradients
parameter gradients
Neural network components

Validated for:

Linear
Embedding
Dropout
BatchNorm1d
BatchNorm2d
LayerNorm
containers
losses
optimizers
Transformer stack

Validated for:

causal mask
padding mask
scaled dot-product attention
multi-head attention
transformer block
transformer encoder
classifier / token classifier / LM path
generation
Checkpointing

Validated for:

model save/load
state restoration
forward consistency after reload
checkpoint resume flow
Public API

Validated for:

root imports
package exports
canonical import style
Why both regression and parity tests exist

The project uses two complementary testing layers.

Regression tests

These verify that framework behavior stays stable over time:

shapes
gradients
outputs are finite
API contracts
train/eval behavior
checkpoint behavior
Parity tests

These verify that core implementations numerically agree with a trusted reference:

PyTorch is used as the reference implementation
forward outputs are compared
gradients are compared
tolerance-based matching is used with np.allclose(...)

Regression tests answer:

“Did we break framework behavior?”

Parity tests answer:

“Does this implementation match a known-correct reference?”

Both are important.

Conventions used in tests
Numeric comparison

Most value-based checks use:

np.allclose(actual, expected, atol=1e-5, rtol=1e-5)
Tensor-to-NumPy conversion

Because the framework uses custom Tensor objects, tests commonly use a helper like:

def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "data"):
        data = x.data
        if isinstance(data, np.ndarray):
            return data
        return np.array(data)
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.array(x)
Gradient access

Tests usually read gradients through:

def get_grad(x):
    grad = getattr(x, "grad", None)
    if grad is None:
        return None
    return to_numpy(grad)
Adding new tests

When adding a new module, it is recommended to add:

a regression test in tests/
a parity test in tests_parity/ if a PyTorch reference exists
a smoke-style check only if public API behavior changes
Minimum standard for a new regression test

A useful test should usually include:

one happy path
one shape/value check
one edge case
one backward check if the module is differentiable
Minimum standard for a parity test

A useful parity test should usually include:

forward parity
backward parity
reference weight synchronization
handling of layout differences if internal storage differs from PyTorch
Common pitfalls
Internal layout differences

Some components intentionally use a different internal layout than PyTorch.

Examples:

Linear.weight may be stored as (in_features, out_features)
Linear.bias may be stored as (1, out_features)

Parity tests should compare values correctly rather than assuming identical raw storage layout.

Tensor vs NumPy buffers

Running statistics such as BatchNorm buffers may be stored as framework Tensor objects rather than raw NumPy arrays.
Tests should convert them with to_numpy(...) before comparison.

Root API vs canonical import path

Some examples may use canonical imports from subpackages if the root package export surface is intentionally minimal or still evolving.

Example workflow
During normal development

Run:

pytest -q tests
When changing public exports

Run:

pytest -q smoke_public_api_imports.py smoke_canonical_imports.py
When modifying a reference-checked component

Run the matching parity file, for example:

pytest -q tests_parity/test_linear_parity_torch.py
Before considering a feature complete

Run everything:

pytest -q tests tests_parity smoke_public_api_imports.py smoke_canonical_imports.py
Testing philosophy

Neural Kernel is intended to be both educational and structurally sound.

The testing strategy reflects that:

regression tests protect behavior
parity tests build trust in correctness
smoke tests protect the public surface
runnable examples verify real usage paths

This makes the project easier to evolve without silently breaking core functionality.