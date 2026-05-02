from __future__ import annotations

import numpy as np
from tensorflow.keras.datasets import mnist

from kernel.core.tensor import Tensor
from kernel.nn import ReLU, Sequential
from kernel.nn.layers.linear import Linear
from kernel.utils.checkpoint import load_checkpoint


def build_model():
    return Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10),
    )


def predict(model, x: np.ndarray) -> np.ndarray:
    logits = model(Tensor(x)).data
    return np.argmax(logits, axis=1)


def accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    return float((preds == targets).mean())


def main():
    (_, _), (x_test, y_test) = mnist.load_data()

    x_test = x_test.astype(np.float64) / 255.0
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    y_test = y_test.astype(np.int64)

    model = build_model()
    load_checkpoint(model, "checkpoints/mnist_mlp.pkl")
    print("Model loaded from checkpoints/mnist_mlp.pkl")

    preds = predict(model, x_test)
    test_acc = accuracy(preds, y_test)

    print(f"\nTest accuracy: {test_acc:.4f}")

    print("\nSample predictions:")
    for i in range(10):
        print(
            f"sample={i:02d} "
            f"true={y_test[i]} "
            f"pred={preds[i]}"
        )


if __name__ == "__main__":
    main()