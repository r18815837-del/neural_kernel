from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

from kernel.utils import load_checkpoint, set_seed
from kernel.core.tensor import Tensor
from kernel.nn import BatchNorm2d, ReLU, Sequential
from kernel.nn.layers.conv import Conv2d
from kernel.nn.layers.flatten import Flatten
from kernel.nn.layers.linear import Linear
from kernel.nn.layers.pooling import MaxPool2d


def build_model():
    return Sequential(
        Conv2d(1, 8, 3, padding=1),
        BatchNorm2d(8),
        ReLU(),
        MaxPool2d(2),

        Conv2d(8, 16, 3, padding=1),
        BatchNorm2d(16),
        ReLU(),
        MaxPool2d(2),

        Conv2d(16, 32, 3, padding=1),
        BatchNorm2d(32),
        ReLU(),

        Flatten(),
        Linear(32 * 7 * 7, 128),
        ReLU(),
        Linear(128, 10),
    )


def accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return float((preds == targets).mean())


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def topk_info(probs_row: np.ndarray, k: int = 3):
    top_idx = np.argsort(probs_row)[-k:][::-1]
    return [(int(i), float(probs_row[i])) for i in top_idx]


def main():
    np.random.seed(42)

    (_, _), (x_test, y_test) = mnist.load_data()

    x_test = x_test.astype(np.float64) / 255.0
    y_test = y_test.astype(np.int64)

    x_test_model = x_test[:, None, :, :]
    x_test_model = x_test_model[:500]
    y_test = y_test[:500]
    x_test_vis = x_test[:500]

    model = build_model()
    load_checkpoint(model, "checkpoints/mnist_cnn_deeper.pkl")
    model.eval()

    logits = model(Tensor(x_test_model)).data
    probs = softmax(logits)
    preds = np.argmax(logits, axis=1)
    confs = probs[np.arange(len(preds)), preds]
    acc = accuracy(logits, y_test)

    print("Loaded CNN checkpoint: checkpoints/mnist_cnn_deeper.pkl")
    print(f"Test accuracy: {acc:.4f}")

    num_examples = 10
    indices = np.random.choice(len(x_test_vis), size=num_examples, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        image = x_test_vis[idx]
        true_label = y_test[idx]
        pred_label = preds[idx]
        confidence = confs[idx]
        top3 = topk_info(probs[idx], k=3)
        ok = true_label == pred_label

        ax.imshow(image, cmap="gray")
        ax.axis("off")

        top3_text = ", ".join([f"{cls}:{prob:.2f}" for cls, prob in top3])

        title = (
            f"true={true_label} pred={pred_label}\n"
            f"conf={confidence:.3f}\n"
            f"top3 {top3_text}"
        )
        if not ok:
            title += "\nWRONG"

        ax.set_title(title, fontsize=9)

        print(
            f"sample={idx:03d} "
            f"true={true_label} "
            f"pred={pred_label} "
            f"conf={confidence:.3f} "
            f"top3={top3} "
            f"{'OK' if ok else 'WRONG'}"
        )

    plt.suptitle(f"MNIST deeper CNN inference — accuracy={acc:.4f}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()