from __future__ import annotations

import numpy as np
from tensorflow.keras.datasets import mnist

from kernel.data import DataLoader, TensorDataset
from kernel.core.tensor import Tensor
from kernel.loss.classification import CrossEntropyLoss
from kernel.nn import BatchNorm1d, Dropout, ReLU, Sequential
from kernel.nn.layers.linear import Linear
from kernel.optim import Adam
from kernel.utils.checkpoint import save_checkpoint


def accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return float((preds == targets).mean())


def main():
    np.random.seed(42)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype(np.float64) / 255.0
    x_test = x_test.astype(np.float64) / 255.0

    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)

    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Dropout(0.2),
        Linear(128, 64),
        ReLU(),
        Dropout(0.2),
        Linear(64, 10),
    )

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    epochs = 20

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for xb, yb in train_loader:
            xb = Tensor(xb)

            model.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.item()
            num_batches += 1

        model.eval()
        train_logits = model(Tensor(x_train)).data
        test_logits = model(Tensor(x_test)).data

        train_acc = accuracy(train_logits, y_train)
        test_acc = accuracy(test_logits, y_test)
        avg_loss = epoch_loss / max(num_batches, 1)

        print(
            f"epoch={epoch:02d} "
            f"loss={avg_loss:.6f} "
            f"train_acc={train_acc:.4f} "
            f"test_acc={test_acc:.4f}"
        )

    model.eval()
    final_train_acc = accuracy(model(Tensor(x_train)).data, y_train)
    final_test_acc = accuracy(model(Tensor(x_test)).data, y_test)

    print("\nFinal metrics:")
    print(f"train_acc: {final_train_acc:.4f}")
    print(f"test_acc:  {final_test_acc:.4f}")

    save_checkpoint(model, "checkpoints/mnist_mlp.pkl")
    print("Model saved to checkpoints/mnist_mlp.pkl")


if __name__ == "__main__":
    main()