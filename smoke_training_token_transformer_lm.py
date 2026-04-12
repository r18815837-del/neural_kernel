import numpy as np

from kernel.nn.losses import CrossEntropyLoss
from kernel.nn.modules.token_lm import TokenTransformerLM
from kernel.optim.adam import Adam


def make_batch(batch_size=16, seq_len=8, vocab_size=20):
    # Генерируем последовательности с простой закономерностью:
    # стартовый токен + шаг 1 по модулю vocab_size
    starts = np.random.randint(0, vocab_size, size=(batch_size, 1), dtype=np.int64)
    offsets = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)
    token_ids = (starts + offsets) % vocab_size
    return token_ids.astype(np.int64)


def compute_token_accuracy(logits, targets) -> float:
    logits_np = logits.detach().numpy()
    pred = logits_np.argmax(axis=1)
    targets_np = np.asarray(targets)
    return float((pred == targets_np).mean())


def main():
    np.random.seed(42)

    model = TokenTransformerLM(
        vocab_size=20,
        d_model=16,
        num_heads=4,
        d_ff=32,
        num_layers=2,
        dropout_p=0.0,
        max_len=32,
        activation="gelu",
    )

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-2)

    print("start token LM training demo")
    first_loss = None
    last_loss = None

    for step in range(1, 101):
        token_ids = make_batch(batch_size=16, seq_len=8, vocab_size=20)

        model.train()
        model.zero_grad()

        logits, _ = model(token_ids, use_causal_mask=True)   # (B, T, V)

        input_logits = logits[:, :-1, :]                     # predict next token
        targets = token_ids[:, 1:]                           # next-token labels

        B, Tm1, V = input_logits.shape
        flat_logits = input_logits.reshape(B * Tm1, V)
        flat_targets = targets.reshape(B * Tm1)

        loss = loss_fn(flat_logits, flat_targets)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().numpy())
        acc_value = compute_token_accuracy(flat_logits, flat_targets)

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

    print("[OK] token LM training demo")


if __name__ == "__main__":
    main()