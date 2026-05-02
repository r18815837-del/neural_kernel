from __future__ import annotations

import numpy as np
from tensorflow.keras.datasets import mnist

from kernel.utils import History, set_seed, save_checkpoint, plot_history
from kernel.data import DataLoader, TensorDataset
from kernel.core.tensor import Tensor
from kernel.loss.classification import CrossEntropyLoss
from kernel.nn import BatchNorm2d, ReLU, Sequential, ResidualBlock
from kernel.nn.layers.conv import Conv2d
from kernel.nn.layers.flatten import Flatten
from kernel.nn.layers.linear import Linear
from kernel.nn.layers.pooling import MaxPool2d
from kernel.optim import Adam


def accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return float((preds == targets).mean())


def main():
    set_seed(42)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype(np.float64) / 255.0
    x_test = x_test.astype(np.float64) / 255.0

    x_train = x_train[:, None, :, :]
    x_test = x_test[:, None, :, :]

    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    x_train = x_train[:2000]
    y_train = y_train[:2000]
    x_test = x_test[:500]
    y_test = y_test[:500]

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = Sequential(
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

        ResidualBlock(32),

        Flatten(),
        Linear(32 * 7 * 7, 128),
        ReLU(),
        Linear(128, 10),
    )

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    history = History()

    epochs = 3
    best_test_acc = -1.0
    best_epoch = 0

    print("Training deeper CNN on MNIST...")
    print(f"Model parameters: {sum(p.data.size for p in model.parameters()):,}")

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

            epoch_loss += float(loss.data)
            num_batches += 1

        model.eval()
        train_logits = model(Tensor(x_train)).data
        test_logits = model(Tensor(x_test)).data

        train_acc = accuracy(train_logits, y_train)
        test_acc = accuracy(test_logits, y_test)
        avg_loss = epoch_loss / max(num_batches, 1)

        history.log(
            epoch=epoch,
            train_loss=avg_loss,
            train_acc=train_acc,
            test_acc=test_acc,
            lr=optimizer.lr,
        )

        print(
            f"epoch={epoch:02d} "
            f"loss={avg_loss:.6f} "
            f"train_acc={train_acc:.4f} "
            f"test_acc={test_acc:.4f} "
            f"lr={optimizer.lr:.6f}"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            save_checkpoint(model, "checkpoints/mnist_cnn_residual.pkl")
            print(
                "New best model saved to checkpoints/mnist_cnn_residual.pkl "
                f"(epoch={epoch:02d}, test_acc={test_acc:.4f})"
            )

    print("\nTraining finished.")
    print(f"Best epoch:    {best_epoch}")
    print(f"Best test_acc: {best_test_acc:.4f}")
    print(f"History keys:  {list(history.keys())}")
    print("History data:")
    print(history.as_dict())
    plot_history(history)


if __name__ == "__main__":
    main()