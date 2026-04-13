# Release Checklist

## Purpose
This checklist defines the minimum requirements for preparing and validating a release of `neural_kernel`.

It is intended to make releases repeatable, auditable, and low-friction.

---

## 1. Package Metadata

- [ ] Package name verified
- [ ] Version set correctly
- [ ] Description reviewed
- [ ] Long description / README presentation reviewed
- [ ] License metadata verified
- [ ] Author / maintainer metadata reviewed
- [ ] Project URLs verified
- [ ] Classifiers reviewed
- [ ] Keywords reviewed
- [ ] Python version support declared correctly

---

## 2. Dependency Review

- [ ] Core dependencies reviewed
- [ ] Optional dependencies reviewed
- [ ] Dev dependencies reviewed
- [ ] Test dependencies reviewed
- [ ] Example-related dependencies reviewed
- [ ] CUDA-related optional dependencies reviewed
- [ ] No accidental unnecessary dependency leaks

### Optional Extras
- [ ] `dev` extra verified
- [ ] `test` extra verified if used
- [ ] `examples` extra verified
- [ ] `cuda` extra verified if supported by current packaging strategy

---

## 3. Package Structure

- [ ] Source layout reviewed
- [ ] Public API import path verified
- [ ] Internal modules do not leak unintentionally into public API
- [ ] Top-level package import works
- [ ] Version exposure strategy is defined
- [ ] Distribution includes required package files
- [ ] Distribution excludes unintended files

---

## 4. Install Validation

### Fresh Install
- [ ] `pip install .` works in a clean environment
- [ ] Package imports successfully after install
- [ ] Minimal quickstart runs successfully

### Editable Install
- [ ] `pip install -e .` works
- [ ] `pip install -e .[dev]` works
- [ ] Development workflow install is stable

### Optional Paths
- [ ] `examples` extra install path validated
- [ ] `cuda` extra install path validated if part of current packaging plan

---

## 5. Test Validation

### Core Validation
- [ ] Regression suite passes
- [ ] Optimizer state unit tests pass
- [ ] Resume integration test passes
- [ ] Public API smoke tests pass

### Device Validation
- [ ] CPU tests pass
- [ ] CUDA tests pass where applicable

### Parity Validation
- [ ] Existing parity tests remain green

---

## 6. Benchmark Validation

- [ ] Benchmark runner exists
- [ ] Benchmark suite executes without structural errors
- [ ] Environment capture works
- [ ] Benchmark methodology note exists
- [ ] Benchmark outputs are machine-readable

> Note: final benchmark result publication is not required for Sprint 1 readiness, but benchmark infrastructure must be in place.

---

## 7. Documentation Readiness

- [ ] README install section reviewed
- [ ] README quickstart section reviewed
- [ ] Package usage examples reviewed
- [ ] Checkpoint/resume docs reviewed or scoped
- [ ] Known limitations are documented where necessary
- [ ] Release notes scaffold prepared

---

## 8. Changelog and Versioning

- [ ] Version chosen
- [ ] Changelog entry created
- [ ] Release notes summary drafted
- [ ] Versioning convention documented
- [ ] Git tag naming convention decided

---

## 9. Build Validation

- [ ] Source distribution builds successfully
- [ ] Wheel builds successfully
- [ ] Built artifacts inspected
- [ ] Artifact contents look correct
- [ ] No packaging surprises in final dist

---

## 10. Git / Repository State

- [ ] Main branch is clean
- [ ] Release commit prepared
- [ ] Release tag ready
- [ ] CI status is green
- [ ] Required release files are committed

---

## 11. Publish Steps

- [ ] Build release artifacts
- [ ] Verify artifacts locally
- [ ] Upload artifacts to package registry
- [ ] Create Git tag
- [ ] Create GitHub release
- [ ] Attach release notes

---

## 12. Post-Release Validation

- [ ] Install released version from registry
- [ ] Import package from released version
- [ ] Run minimal quickstart against released package
- [ ] Confirm release metadata is visible correctly
- [ ] Confirm release notes are correct

---

## Release Decision Gate

A release is considered publishable only if all of the following are true:

- [ ] Package installs cleanly
- [ ] Public API imports cleanly
- [ ] Tests are green
- [ ] Resume flow is validated
- [ ] Packaging metadata is complete
- [ ] Build artifacts are correct
- [ ] Changelog is prepared
- [ ] CI is green

---

## Notes
- Release quality matters more than release speed.
- Reproducibility matters more than cosmetic polish.
- Installation trust is part of product trust.