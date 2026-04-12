import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.losses import CrossEntropyLoss
from kernel.nn.modules.token_classifier import TokenTransformerClassifier
from kernel.optim.adam import Adam


def make_batch(batch_size=16, seq_len=6, vocab_size=20):
    token_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)

    # простая синтетическая задача:
    # класс зависит от суммы токенов по модулю 3
    targets = (token_ids.sum(axis=1) % 3).astype(np.int64)

    return token_ids, targets


def compute_accuracy(logits: Tensor, targets) -> float:
    logits_np = logits.detach().numpy()
    pred = logits_np.argmax(axis=1)
    targets_np = np.asarray(targets)
    return float((pred == targets_np).mean())


def main():
    np.random.seed(42)

    model = TokenTransformerClassifier(
        vocab_size=20,
        d_model=16,
        num_heads=4,
        d_ff=32,
        num_layers=2,
        num_classes=3,
        dropout_p=0.0,
        max_len=32,
        activation="gelu",
        pooling="mean",
    )

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-2)

    print("start token classifier training demo")
    first_loss = None
    last_loss = None

    for step in range(1, 81):
        token_ids, targets = make_batch(batch_size=16, seq_len=6, vocab_size=20)

        model.train()
        model.zero_grad()

        logits, _ = model(token_ids)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().numpy())
        acc_value = compute_accuracy(logits, targets)

        if first_loss is None:
            first_loss = loss_value
        last_loss = loss_value

        if step % 10 == 0 or step == 1:
            print(f"step={step:03d}  loss={loss_value:.6f}  acc={acc_value:.3f}")

    print()
    print(f"first_loss={first_loss:.6f}")
    print(f"last_loss={last_loss:.6f}")

    if last_loss > first_loss:
        raise AssertionError(
            f"Training did not improve: first_loss={first_loss:.6f}, last_loss={last_loss:.6f}"
        )

    print("[OK] token classifier training demo")


if __name__ == "__main__":
    main()