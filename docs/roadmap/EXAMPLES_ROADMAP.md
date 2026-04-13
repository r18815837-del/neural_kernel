


```md
# Examples Roadmap

## Purpose
This document defines the flagship examples that will represent `neural_kernel` publicly in the next development phase.

The goal is to choose examples that best demonstrate the current strengths of the framework while also supporting future product and startup positioning.

## Selected Flagship Examples

The two flagship examples are:

1. `tiny_gpt`
2. `text_classification`

These are intentionally selected over a CNN-first showcase because together they better reflect:
- the current strength of the framework
- the public demo value of the project
- the future direction toward AI systems and assistants

---

## Example 1 — `tiny_gpt`

### Purpose
Demonstrate the language-modeling and generation capabilities of `neural_kernel`.

### Why It Matters
This example showcases:
- Transformer / LM path
- token generation flow
- greedy / temperature / top-k / top-p decoding
- checkpointing/resume potential
- practical sequence modeling

It is the strongest public-facing showcase for the current framework stack.

### What It Should Demonstrate
- model definition
- training loop
- checkpoint save/load
- text generation after training
- configurable decoding strategies
- basic evaluation or sample output inspection

### Minimum Deliverables
- `train.py`
- `generate.py`
- `README.md`

### Preferred Deliverables
- `eval.py`
- config file or simple argument interface
- example checkpoint usage notes
- sample outputs in docs

### Technical Themes
- Transformer stack correctness
- LM head usage
- generation utilities
- checkpoint integration

### Public Value
This example helps communicate that `neural_kernel` is not just a tensor/autograd project, but a usable mini deep learning framework with real LM capability.

---

## Example 2 — `text_classification`

### Purpose
Demonstrate a practical supervised learning workflow using `neural_kernel`.

### Why It Matters
This example showcases:
- end-to-end training
- evaluation flow
- inference flow
- practical business-facing use case
- approachable entry point for users

It balances the more impressive LM demo with a clean, understandable applied ML example.

### What It Should Demonstrate
- dataset preparation
- model training
- validation/evaluation
- simple inference path
- metric reporting
- checkpoint usage if practical

### Minimum Deliverables
- `train.py`
- `eval.py`
- `predict.py`
- `README.md`

### Preferred Deliverables
- simple config path
- expected metrics or expected output format
- example input/output samples
- checkpoint load example

### Technical Themes
- embeddings and/or encoder usage depending on architecture choice
- training/evaluation lifecycle
- practical loss optimization
- inference API clarity

### Public Value
This example helps show that `neural_kernel` can support real applied ML tasks, not only low-level framework mechanics.

---

## Why These Two Were Chosen

### `tiny_gpt`
Chosen because it highlights:
- current transformer maturity
- generation support already implemented
- strongest technical demo value
- strongest startup-facing narrative

### `text_classification`
Chosen because it highlights:
- usability
- practical supervised ML flow
- clear developer onboarding path
- easy communication to non-research users

Together they provide both:
- technical depth
- practical accessibility

---

## Why CNN Is Not a Flagship Example Right Now

CNN support is valuable and should remain part of the project, but it is not the best top-level showcase at this stage.

Reasons:
- less aligned with the strongest differentiators already built
- lower public demo impact compared to LM/generation
- weaker alignment with future AI assistant / software factory positioning

CNN examples can still exist as supporting examples.

---

## Suggested Example Structure

### tiny_gpt
```text
examples/
  tiny_gpt/
    README.md
    train.py
    generate.py
    eval.py        # optional in first implementation
    config.py      # optional

  text_classification/
    README.md
    train.py
    eval.py
    predict.py
    config.py      # optional
Documentation Expectations

Each flagship example should include:

README Requirements
what the example does
what framework features it demonstrates
how to run it
what outputs to expect
how checkpoints are used
how to adapt it
Execution Requirements
minimal commands to run
clear dependency expectations
clear dataset or sample data expectations
Output Expectations
sample metrics or sample text generations
expected behavior after short training runs
notes on reproducibility limitations if relevant
Sprint 2 Expectations

These examples are expected to be implemented or polished in Sprint 2.

Sprint 2 goals for flagship examples
runnable example folders
polished README files
clean CLI or minimal config flow
package/demo cleanup
alignment with updated docs
Long-Term Example Roadmap
Flagship Tier
tiny_gpt
text_classification
Supporting Tier
CNN classifier
transformer encoder example
attention-focused minimal example
checkpoint/resume example
parity demonstration example
Summary

The selected flagship examples are designed to maximize:

technical credibility
usability
public demo strength
alignment with future startup direction

The roadmap is intentionally focused.
The goal is not to have many examples first.
The goal is to have the right examples first.