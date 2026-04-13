# Sprint 1

## Title
Sprint 1 — Reliability, Packaging, and Benchmark Foundation

## Goal
Make `neural_kernel` training-lifecycle reliable, package-ready, and benchmark-ready.

This sprint is focused on turning the project from a strong engineering repository into a product-grade framework foundation that is ready for release work in Sprint 2.

## Strategic Objective
This sprint must achieve three outcomes:

1. Close the reliability gap in optimizer/checkpoint lifecycle
2. Make the package structurally ready for PyPI/release publishing
3. Establish a reproducible benchmark framework for public comparison

In parallel, this sprint will also lock the showcase direction by selecting two flagship examples.

## Scope

### In Scope
- Optimizer `state_dict()` design and implementation
- Optimizer `load_state_dict()` design and implementation
- Resume training correctness validation
- Checkpoint API cleanup related to optimizer state
- Package metadata and install-path polish
- Release process scaffolding
- Benchmark suite skeleton
- Benchmark methodology definition
- Flagship examples shortlist

### Out of Scope
- Public PyPI publishing
- Final benchmark report
- Major docs rewrite
- New transformer features
- Expanded CUDA feature roadmap
- Software factory orchestration
- New large example implementations
- GitHub Pages / badges polish

## Deliverables

### Reliability
- Optimizer `state_dict()` implemented
- Optimizer `load_state_dict()` implemented
- Unit tests for optimizer state serialization
- Resume integration test
- Checkpoint format reviewed and cleaned up

### Packaging
- `pyproject.toml` reviewed and polished
- Package metadata aligned
- Optional dependency strategy defined
- Clean install flow validated
- Release checklist created
- Changelog scaffold created

### Benchmarking
- `benchmarks/` directory created
- Benchmark runner skeleton created
- Initial benchmark cases defined
- Environment capture helper defined
- Benchmark methodology documented

### Examples
- Two flagship examples selected
- Example scope documented
- Example goals documented for Sprint 2 implementation

## Success Criteria

Sprint 1 is successful if all of the following are true:

- Optimizer state can be serialized and restored
- Training can resume correctly from checkpoint
- Package installs cleanly from a fresh environment
- Release process is documented and repeatable
- Benchmark framework exists and can be extended systematically
- Two flagship examples are selected and documented

## Definition of Done

Sprint 1 is considered complete only when all items below are satisfied.

### Optimizer / State
- `optimizer.state_dict()` exists
- `optimizer.load_state_dict()` exists
- serialization format is documented internally
- unit tests pass
- integration resume test passes

### Package / Release
- `pyproject.toml` is reviewed and cleaned
- package metadata is complete and professional
- install path is validated
- release checklist exists
- changelog scaffold exists

### Benchmarks
- `benchmarks/` exists
- benchmark runner exists
- at least several benchmark cases are scaffolded
- environment capture is defined
- benchmark methodology note exists

### Examples
- `tiny_gpt` is selected as flagship example
- `text_classification` is selected as flagship example
- example scope is documented

## Priorities

### Priority 1
Optimizer state lifecycle reliability

### Priority 2
Package/release readiness

### Priority 3
Benchmark skeleton and measurement foundation

### Priority 4
Showcase direction lock-in

## Execution Order

1. Audit and lock sprint scope
2. Design optimizer serialization contract
3. Implement `state_dict()`
4. Implement `load_state_dict()`
5. Add resume integration test
6. Review and clean checkpoint API
7. Review package metadata and install flow
8. Define release process scaffolding
9. Build benchmark skeleton
10. Finalize flagship examples shortlist

## Constraints
- Avoid overdesigning optimizer serialization for hypothetical future features
- Prefer minimal clean extensible design over excessive abstraction
- Benchmark work in this sprint is infrastructure-focused, not result-focused
- Packaging work must prioritize reproducibility over presentation polish

## Risks

### Risk 1 — Overengineering optimizer state format
Mitigation:
Keep the schema minimal, explicit, and versioned.

### Risk 2 — Benchmark work expanding beyond sprint scope
Mitigation:
Only build the skeleton and methodology in Sprint 1. Final benchmark runs belong to Sprint 2.

### Risk 3 — Packaging polish consuming too much time
Mitigation:
Focus on install correctness, metadata completeness, and release repeatability.

### Risk 4 — Example selection drifting
Mitigation:
Lock the two flagship examples early and treat them as fixed for Sprint 2.

## Flagship Examples
The two flagship examples for this roadmap are:

1. `tiny_gpt`
2. `text_classification`

These are selected because together they demonstrate:
- Transformer / LM path
- generation capabilities
- practical supervised training flow
- training/evaluation/inference lifecycle
- strong public demo value

## Sprint Output Summary
By the end of Sprint 1, `neural_kernel` should look like a framework that:
- can be trusted in checkpoint/resume workflows
- is structurally ready for release packaging
- has a benchmark foundation for honest public measurement
- has a clear showcase direction for future demos