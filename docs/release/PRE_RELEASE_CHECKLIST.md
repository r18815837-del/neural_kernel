# Pre-Release Checklist

## Repository Hygiene
- [ ] top-level repository is clean
- [ ] no temporary artifacts in root
- [ ] no debug-only scripts left in root
- [x] README.md is updated
- [x] CHANGELOG.md is present
- [x] LICENSE is present
- [ ] examples/README.md is present
- [x] docs/README.md is present

## Package Metadata
- [ ] `pyproject.toml` has real GitHub URLs
- [x] author metadata is correct
- [x] version is correct
- [x] description is correct
- [x] classifiers are correct
- [x] optional dependencies are correct

## Installation Validation
- [x] `pip install .` passes
- [x] `pip install -e .[dev]` passes
- [x] `pip install -e .[examples]` passes
- [x] `import kernel` works
- [x] public API smoke import works

## Testing
- [x] optimizer state tests pass
- [x] checkpoint tests pass
- [x] regression suite passes
- [x] parity suite passes
- [x] smoke imports pass
- [ ] CUDA tests pass where available

## Benchmarks
- [x] benchmark runner executes
- [x] environment metadata is captured
- [x] benchmark JSON output is produced
- [x] benchmark results are documented

## Documentation
- [x] README install section is accurate
- [x] README quickstart is accurate
- [x] README validation section is accurate
- [x] README benchmark section is present
- [x] release notes draft exists
- [x] changelog is updated

## Examples
- [x] flagship examples are identified
- [x] examples/README.md exists
- [x] key examples start correctly
- [x] checkpoint example runs
- [x] transformer example runs
- [x] LM example runs

## Release Readiness
- [x] no known blocker remains
- [x] release notes are drafted
- [x] release candidate has been reviewed
- [x] publish flow is understood