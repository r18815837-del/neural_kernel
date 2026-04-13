


```md
# Optimizer State Contract

## Purpose
This document defines the serialization contract for optimizer state in `neural_kernel`.

The goal is to support:
- checkpoint save/load
- reliable training resume
- explicit and extensible state structure
- future format evolution with versioning

## Public API

The optimizer serialization API is:

```python
state = optimizer.state_dict()
optimizer.load_state_dict(state)

This API must be stable enough to support checkpoint/resume workflows.

Design Goals
Goal 1 — Reliability

The serialized state must contain enough information to restore optimizer behavior for continued training.

Goal 2 — Explicitness

The structure should be easy to inspect and reason about.

Goal 3 — Extensibility

The format should support future optimizer features without breaking old assumptions.

Goal 4 — Minimalism

Do not add speculative complexity for unsupported future features.

Proposed Top-Level Structure

A serialized optimizer state should have the following top-level shape:

{
    "param_groups": ...,
    "state": ...,
    "defaults": ...,
    "meta": {
        "optimizer_class": ...,
        "format_version": ...
    }
}
Field Definitions
param_groups

Contains optimizer parameter grouping information and group-level hyperparameters.

This should include:

parameter references or parameter indices
learning rate
weight decay
momentum / betas / eps or equivalent fields depending on optimizer

If full parameter groups are not yet complex in the current implementation, the format should still remain compatible with future grouping.

state

Contains per-parameter optimizer state required to continue training correctly.

Examples:

step count
momentum buffers
first moment estimates
second moment estimates
any optimizer-specific accumulators

This data must be enough to preserve update continuity after loading.

defaults

Contains optimizer default hyperparameters.

This should reflect the optimizer configuration in a normalized form.

meta

Contains serialization metadata.

Required fields:

optimizer_class
format_version

Example:

{
    "meta": {
        "optimizer_class": "Adam",
        "format_version": 1
    }
}
Parameter Identification Strategy
Current Decision

Parameters should be identified in optimizer state using stable internal ordering or stable parameter indices derived from optimizer parameter registration order.

Rationale

This is simpler and more reliable in the current stage than attempting complex object identity mapping in serialized form.

Constraint

load_state_dict() should require parameter structure compatibility between:

the optimizer being loaded into
the serialized optimizer state
Validation Rules for load_state_dict()

The loading path should validate at least the following:

top-level structure is a mapping/dict
required keys exist
optimizer class compatibility where relevant
parameter count compatibility
parameter group compatibility
state structure validity

If validation fails, load_state_dict() should raise a clear, explicit error.

Error Handling Principles
Must Do

Raise informative errors when:

required keys are missing
parameter counts do not match
parameter groups are structurally incompatible
state payload is malformed
Must Avoid
silent partial loads
ambiguous fallback behavior
hidden mismatches

Reliability is more important than permissive loading.

Device Considerations
Goal

Serialized optimizer state should participate cleanly in checkpoint workflows involving CPU/CUDA where supported by the broader framework.

Current Practical Rule

If device transfer behavior is already supported by the existing tensor/checkpoint system, optimizer state should integrate with it.

If device portability is not fully guaranteed yet, current limitations must be documented explicitly rather than hidden.

Resume Correctness Requirement

The serialization contract is only valid if this scenario works correctly:

train model for N steps
save checkpoint
recreate model and optimizer
load model state
load optimizer state
continue training
obtain consistent training continuation behavior

This must be tested with an integration test.

Minimum Test Coverage
Unit Tests
state_dict() returns required structure
load_state_dict() accepts valid structure
invalid state raises clear errors
optimizer state is preserved across round-trip serialization
Integration Test
interrupted training vs uninterrupted training comparison
resumed training path behaves consistently after checkpoint restore
Format Versioning
Current Version

format_version = 1

Rule

Any future structural serialization change should explicitly consider whether:

it is backward-compatible
it requires format version bump
migration logic is necessary

Even if migration is not implemented immediately, versioning should exist from the start.

Non-Goals

This contract does not currently aim to solve:

cross-framework optimizer import/export
complex partial parameter remapping
advanced sharded/distributed optimizer checkpointing
permissive loading of structurally mismatched optimizers
Summary

The optimizer state contract should be:

simple
explicit
reliable
versioned
sufficient for resume correctness

The immediate target is not maximum generality.
The immediate target is trustworthy checkpoint/resume behavior.