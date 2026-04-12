import numpy as np

from kernel.nn.losses import CrossEntropyLoss
from kernel.nn.modules.token_lm import TokenTransformerLM
from kernel.optim.adam import Adam


def make_batch(batch_size=16, seq_len=8, vocab_size=20):
    starts = np.random.randint(0, vocab_size, size=(batch_size, 1), dtype=np.int64)
    offsets = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)
    token_ids = (starts + offsets) % vocab_size
    return token_ids.astype(np.int64)


def train_model():
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

    first_loss = None
    last_loss = None

    for step in range(1, 101):
        token_ids = make_batch(batch_size=16, seq_len=8, vocab_size=20)

        model.train()
        model.zero_grad()

        logits, _ = model(token_ids, use_causal_mask=True)

        input_logits = logits[:, :-1, :]
        targets = token_ids[:, 1:]

        B, Tm1, V = input_logits.shape
        flat_logits = input_logits.reshape(B * Tm1, V)
        flat_targets = targets.reshape(B * Tm1)

        loss = loss_fn(flat_logits, flat_targets)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().numpy())
        if first_loss is None:
            first_loss = loss_value
        last_loss = loss_value

    return model, first_loss, last_loss


def expected_continuation(prompt, max_new_tokens, vocab_size):
    current = prompt.copy()
    for _ in range(max_new_tokens):
        next_token = (current[:, -1:] + 1) % vocab_size
        current = np.concatenate([current, next_token], axis=1)
    return current


def main():
    print("train model for generation demo")
    model, first_loss, last_loss = train_model()

    print(f"first_loss={first_loss:.6f}")
    print(f"last_loss={last_loss:.6f}")

    if last_loss > first_loss:
        raise AssertionError(
            f"Training did not improve: first_loss={first_loss:.6f}, last_loss={last_loss:.6f}"
        )

    prompt = np.array([[7, 8, 9], [18, 19, 0]], dtype=np.int64)
    generated = model.generate(prompt, max_new_tokens=4)
    expected = expected_continuation(prompt, max_new_tokens=4, vocab_size=20)

    print("prompt:")
    print(prompt)
    print("generated:")
    print(generated)
    print("expected:")
    print(expected)

    if not np.array_equal(generated, expected):
        raise AssertionError("Generated sequence does not match expected continuation")

    print("[OK] token LM trained generation demo")


if __name__ == "__main__":
    main()