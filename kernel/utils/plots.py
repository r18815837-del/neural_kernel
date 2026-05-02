from __future__ import annotations

import matplotlib.pyplot as plt


def plot_history(history, show: bool = True):
    data = history.as_dict()

    epochs = data.get("epoch", list(range(1, len(next(iter(data.values()), [])) + 1)))

    fig = plt.figure(figsize=(12, 4))

    if "train_loss" in data:
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(epochs, data["train_loss"])
        ax1.set_title("Train Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

    if "train_acc" in data or "test_acc" in data:
        ax2 = fig.add_subplot(1, 3, 2)
        if "train_acc" in data:
            ax2.plot(epochs, data["train_acc"], label="train_acc")
        if "test_acc" in data:
            ax2.plot(epochs, data["test_acc"], label="test_acc")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()

    if "lr" in data:
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(epochs, data["lr"])
        ax3.set_title("Learning Rate")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("LR")

    plt.tight_layout()

    if show:
        plt.show()

    return fig