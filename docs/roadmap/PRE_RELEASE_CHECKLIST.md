# Pre-Release Checklist

## Repository Hygiene
- [ ] top-level repository is clean
- [ ] no temporary artifacts in root
- [ ] no debug-only scripts left in root
- [ ] README.md is updated
- [ ] CHANGELOG.md is present
- [ ] LICENSE is present
- [ ] examples/README.md is present
- [ ] docs/README.md is present

## Package Metadata
- [ ] `pyproject.toml` has real GitHub URLs
- [ ] author metadata is correct
- [ ] version is correct
- [ ] description is correct
- [ ] classifiers are correct
- [ ] optional dependencies are correct

## Installation Validation
- [ ] `pip install .` passes
- [ ] `pip install -e .[dev]` passes
- [ ] `pip install -e .[examples]` passes
- [ ] `import kernel` works
- [ ] public API smoke import works

## Testing
- [ ] optimizer state tests pass
- [ ] checkpoint tests pass
- [ ] regression suite passes
- [ ] parity suite passes
- [ ] smoke imports pass
- [ ] CUDA tests pass where available

## Benchmarks
- [ ] benchmark runner executes
- [ ] environment metadata is captured
- [ ] benchmark JSON output is produced
- [ ] benchmark results are documented

## Documentation
- [ ] README install section is accurate
- [ ] README quickstart is accurate
- [ ] README validation section is accurate
- [ ] README benchmark section is present
- [ ] release notes draft exists
- [ ] changelog is updated

## Examples
- [ ] flagship examples are identified
- [ ] examples/README.md exists
- [ ] key examples start correctly
- [ ] checkpoint example runs
- [ ] transformer example runs
- [ ] LM example runs

## Release Readiness
- [ ] no known blocker remains
- [ ] release notes are drafted
- [ ] release candidate has been reviewed
- [ ] publish flow is understood