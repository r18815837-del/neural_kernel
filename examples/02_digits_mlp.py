from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from kernel.core.tensor import Tensor
from kernel.loss.classification import CrossEntropyLoss
from kernel.nn import ReLU, Sequential
from kernel.nn.layers.linear import Linear
from kernel.optim import Adam


def accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return float((preds == targets).mean())


def main():
    np.random.seed(42)

    data = load_digits()
    x = data.data.astype(np.float64)
    y = data.target.astype(np.int64)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    x_train = Tensor(x_train)
    x_test = Tensor(x_test)

    model = Sequential(
        Linear(64, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10),
    )

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    epochs = 300

    for epoch in range(1, epochs + 1):
        model.zero_grad()

        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            train_acc = accuracy(model(x_train).data, y_train)
            test_acc = accuracy(model(x_test).data, y_test)
            print(
                f"epoch={epoch:03d} "
                f"loss={loss.data.item():.6f} "
                f"train_acc={train_acc:.4f} "
                f"test_acc={test_acc:.4f}"
            )

    final_train_acc = accuracy(model(x_train).data, y_train)
    final_test_acc = accuracy(model(x_test).data, y_test)

    print("\nFinal metrics:")
    print(f"train_acc: {final_train_acc:.4f}")
    print(f"test_acc:  {final_test_acc:.4f}")


if __name__ == "__main__":
    main()