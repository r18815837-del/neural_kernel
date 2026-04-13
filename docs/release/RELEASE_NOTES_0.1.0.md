

```md id="60310"
# Release Notes — 0.1.0

## Overview
`neural-kernel 0.1.0` is the first structured release candidate for the project as a mini deep learning framework.

This version establishes a strong baseline across:
- tensor + autograd fundamentals
- device-aware execution
- NumPy/CuPy backend support
- CNN path
- Transformer / language-model path
- generation utilities
- checkpointing and resume
- public API structure
- testing and parity validation

## Highlights

### Core framework
- Tensor implementation with autograd
- dynamic graph-based backward propagation
- device-aware tensor API
- NumPy backend
- CuPy CUDA execution path

### Model support
- MLP / CNN workflow
- normalization layers
- residual blocks
- Transformer encoder stack
- token classification
- token language modeling

### Generation
- greedy decoding
- temperature sampling
- top-k sampling
- top-p sampling

### Reliability
- checkpoint save/load
- resume training flow
- optimizer `state_dict()` / `load_state_dict()`
- optimizer checkpoint integration

### Validation
- regression suite coverage
- PyTorch parity for key modules
- CUDA parity for key paths
- smoke-tested public API imports

## Included in This Release Candidate

### Optimizer checkpoint maturity
- added optimizer serialization contract
- added optimizer state save/load
- added validation for invalid optimizer state payloads
- added optimizer checkpoint integration tests

### Packaging
- improved `pyproject.toml`
- cleaned dependency layout
- added optional dependency groups
- validated install and editable install flows

### Benchmarks
- added benchmark suite skeleton
- added benchmark runner
- added environment capture
- added initial benchmark cases for tensor ops and linear layers

## Current Validation Snapshot
- optimizer state tests: passing
- checkpoint tests: passing
- install smoke: passing
- benchmark runner: passing

## Known Next Steps
- broader benchmark coverage
- richer benchmark comparisons vs PyTorch
- stronger example polish for flagship demos
- release publishing workflow finalization
- public docs refinement

## Positioning
This release candidate marks the transition of `neural_kernel` from a strong framework experiment into a more product-grade mini framework with real validation depth.

## License
MIT