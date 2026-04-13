# Changelog

## 0.1.0
- added optimizer `state_dict()` and `load_state_dict()`
- added optimizer serialization validation
- added optimizer state unit tests
- integrated optimizer state with checkpoint save/load
- added resume baseline coverage
- improved `pyproject.toml`
- added optional dependency groups
- validated install and editable install flows
- added benchmark suite skeleton
- added benchmark runner
- added environment capture
- added initial tensor and linear benchmark cases
- selected `tiny_gpt` and `text_classification` as flagship examples
- expanded benchmark coverage to LayerNorm, MultiHeadAttention, and TransformerBlock
- generated preliminary benchmark results for core Transformer execution paths