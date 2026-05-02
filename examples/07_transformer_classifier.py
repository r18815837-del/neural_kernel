import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn import CrossEntropyLoss
from kernel.nn.modules import TokenTransformerClassifier
from kernel.optim import Adam
from kernel.utils import set_seed

def make_synthetic_dataset(
    num_samples=512,
    seq_len=12,
    vocab_size=20,
    num_classes=2,
    seed=42,
):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, vocab_size, size=(num_samples, seq_len), dtype=np.int64)

    # Simple rule-based target:
    # class 1 if the count of tokens >= vocab_size // 2 is large enough, else class 0
    threshold = seq_len // 2
    y = (x >= (vocab_size // 2)).sum(axis=1) >= threshold
    y = y.astype(np.int64)

    if num_classes != 2:
        y = y % num_classes

    return x, y


def batch_iter(x, y, batch_size=32, shuffle=True, seed=42):
    indices = np.arange(len(x))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, len(x), batch_size):
        batch_idx = indices[start:start + batch_size]
        yield x[batch_idx], y[batch_idx]


def accuracy_from_logits(logits, targets):
    preds = np.argmax(logits, axis=1)
    return float((preds == targets).mean())


def main():
    set_seed(42)

    vocab_size = 20
    seq_len = 12
    num_classes = 2

    x_train, y_train = make_synthetic_dataset(
        num_samples=512,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_classes=num_classes,
        seed=42,
    )
    x_val, y_val = make_synthetic_dataset(
        num_samples=128,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_classes=num_classes,
        seed=123,
    )

    model = TokenTransformerClassifier(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        num_classes=num_classes,
        max_len=seq_len,
    )

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-2)

    epochs = 10
    batch_size = 32

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        train_accs = []

        for xb, yb in batch_iter(x_train, y_train, batch_size=batch_size, shuffle=True, seed=epoch):
            tokens = Tensor(xb)
            targets = Tensor(yb)

            optimizer.zero_grad()

            logits = model(tokens)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            logits_np = logits.data if hasattr(logits, "data") else np.array(logits)
            loss_np = loss.data if hasattr(loss, "data") else np.array(loss)

            train_losses.append(float(np.array(loss_np)))
            train_accs.append(accuracy_from_logits(np.array(logits_np), yb))

        model.eval()
        val_losses = []
        val_accs = []

        for xb, yb in batch_iter(x_val, y_val, batch_size=batch_size, shuffle=False):
            tokens = Tensor(xb)
            targets = Tensor(yb)
            logits = model(tokens)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss = criterion(logits, targets)

            logits_np = logits.data if hasattr(logits, "data") else np.array(logits)
            loss_np = loss.data if hasattr(loss, "data") else np.array(loss)

            val_losses.append(float(np.array(loss_np)))
            val_accs.append(accuracy_from_logits(np.array(logits_np), yb))

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={np.mean(train_losses):.4f} | "
            f"train_acc={np.mean(train_accs):.4f} | "
            f"val_loss={np.mean(val_losses):.4f} | "
            f"val_acc={np.mean(val_accs):.4f}"
        )


if __name__ == "__main__":
    main()