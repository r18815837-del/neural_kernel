# Examples

This directory contains runnable examples for `neural_kernel`.

## Getting Started
- `01_linear_regression.py` — minimal training loop and autograd flow
- `02_digits_mlp.py` — basic MLP classification on a small dataset

## MLP / CNN
- `03_mnist_mlp.py` — MNIST classification with an MLP
- `04_mnist_inference.py` — inference using a trained MNIST MLP
- `05_mnist_cnn.py` — CNN training on MNIST
- `06_mnist_cnn_inference.py` — CNN inference on MNIST

## Transformer / Language Modeling
- `07_transformer_classifier.py` — Transformer encoder classifier example
- `08_token_lm_generate.py` — token language model training and generation
- `09_checkpoint_resume.py` — checkpoint save/load and resume workflow

## Planned Flagship Examples
- `tiny_gpt`
- `text_classification`

## Running Examples

Basic usage:

```bash
python examples/01_linear_regression.py
python examples/07_transformer_classifier.py
python examples/08_token_lm_generate.py
python examples/09_checkpoint_resume.py