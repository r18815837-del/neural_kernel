import os
import tempfile

import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn import CrossEntropyLoss
from kernel.nn.modules import TokenTransformerClassifier
from kernel.optim import Adam
from kernel.utils import load_checkpoint, save_checkpoint, set_seed


def make_synthetic_dataset(num_samples=256, seq_len=10, vocab_size=20, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, vocab_size, size=(num_samples, seq_len), dtype=np.int64)

    # simple binary target:
    # class 1 if average token id is high enough, else class 0
    y = (x.mean(axis=1) >= (vocab_size / 2)).astype(np.int64)
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


def train_one_epoch(model, optimizer, criterion, x, y, batch_size=32, seed=42):
    model.train()

    losses = []
    accs = []

    for xb, yb in batch_iter(x, y, batch_size=batch_size, shuffle=True, seed=seed):
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

        losses.append(float(np.array(loss_np)))
        accs.append(accuracy_from_logits(np.array(logits_np), yb))

    return float(np.mean(losses)), float(np.mean(accs))


def evaluate(model, criterion, x, y, batch_size=32):
    model.eval()

    losses = []
    accs = []

    for xb, yb in batch_iter(x, y, batch_size=batch_size, shuffle=False):
        tokens = Tensor(xb)
        targets = Tensor(yb)

        logits = model(tokens)
        if isinstance(logits, tuple):
            logits = logits[0]

        loss = criterion(logits, targets)

        logits_np = logits.data if hasattr(logits, "data") else np.array(logits)
        loss_np = loss.data if hasattr(loss, "data") else np.array(loss)

        losses.append(float(np.array(loss_np)))
        accs.append(accuracy_from_logits(np.array(logits_np), yb))

    return float(np.mean(losses)), float(np.mean(accs))


def make_model(seq_len, vocab_size):
    return TokenTransformerClassifier(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        num_classes=2,
        max_len=seq_len,
    )


def main():
    set_seed(42)

    vocab_size = 20
    seq_len = 10

    x_train, y_train = make_synthetic_dataset(
        num_samples=512,
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=42,
    )
    x_val, y_val = make_synthetic_dataset(
        num_samples=128,
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=123,
    )

    criterion = CrossEntropyLoss()

    print("=== Initial training ===")
    model = make_model(seq_len, vocab_size)
    optimizer = Adam(model.parameters(), lr=1e-2)

    for epoch in range(1, 4):
        train_loss, train_acc = train_one_epoch(
            model, optimizer, criterion, x_train, y_train, batch_size=32, seed=epoch
        )
        val_loss, val_acc = evaluate(model, criterion, x_val, y_val, batch_size=32)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "transformer_classifier.pkl")

        print("\nSaving checkpoint...")
        save_checkpoint(model, ckpt_path)

        print("Creating fresh model and loading checkpoint...")
        resumed_model = make_model(seq_len, vocab_size)
        load_checkpoint(resumed_model, ckpt_path)

        before_resume_loss, before_resume_acc = evaluate(
            resumed_model, criterion, x_val, y_val, batch_size=32
        )

        print(
            f"Loaded model eval | "
            f"val_loss={before_resume_loss:.4f} | val_acc={before_resume_acc:.4f}"
        )

        print("\n=== Continue training from loaded weights ===")
        resumed_optimizer = Adam(resumed_model.parameters(), lr=1e-2)

        for epoch in range(4, 7):
            train_loss, train_acc = train_one_epoch(
                resumed_model,
                resumed_optimizer,
                criterion,
                x_train,
                y_train,
                batch_size=32,
                seed=epoch,
            )
            val_loss, val_acc = evaluate(
                resumed_model, criterion, x_val, y_val, batch_size=32
            )

            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
            )

    print("\nCheckpoint resume demo finished.")


if __name__ == "__main__":
    main()