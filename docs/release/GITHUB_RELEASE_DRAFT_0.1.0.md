# neural-kernel 0.1.0

`neural-kernel 0.1.0` is the first structured release candidate of Neural Kernel as a mini deep learning framework.

## Highlights

- Tensor + autograd engine
- NumPy backend + CuPy CUDA path
- device-aware tensor API
- CNN support
- Transformer / language-model path
- generation: greedy, temperature, top-k, top-p
- checkpoint save/load + resume
- optimizer `state_dict()` / `load_state_dict()`
- clean public API
- parity coverage against PyTorch for key modules
- real CUDA validation
- runnable benchmark infrastructure

## Validation Snapshot
- full automated test suite passing
- benchmark runner operational
- package install validated
- public API smoke imports validated

## Benchmark Coverage
Current benchmark cases include:
- tensor add / mul
- linear forward / backward
- layernorm forward / backward
- multi-head attention forward / backward
- transformer block forward / backward

## Notes
This release establishes a strong engineering baseline for future work on:
- richer benchmark comparisons
- flagship examples
- broader CUDA coverage
- release flow polish
- startup-facing demo packaging

## License
MIT