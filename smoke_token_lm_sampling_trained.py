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
    print("train model for sampling demo")
    model, first_loss, last_loss = train_model()

    print(f"first_loss={first_loss:.6f}")
    print(f"last_loss={last_loss:.6f}")

    if last_loss > first_loss:
        raise AssertionError(
            f"Training did not improve: first_loss={first_loss:.6f}, last_loss={last_loss:.6f}"
        )

    prompt = np.array([[7, 8, 9], [18, 19, 0]], dtype=np.int64)
    expected = expected_continuation(prompt, max_new_tokens=4, vocab_size=20)

    print()
    print("prompt:")
    print(prompt)
    print("expected:")
    print(expected)

    greedy = model.generate(
        prompt,
        max_new_tokens=4,
        do_sample=False,
    )
    print()
    print("greedy:")
    print(greedy)

    if not np.array_equal(greedy, expected):
        raise AssertionError("Greedy generation does not match expected continuation")

    topk1 = model.generate(
        prompt,
        max_new_tokens=4,
        do_sample=True,
        top_k=1,
        temperature=1.0,
    )
    print()
    print("top_k=1 sampling:")
    print(topk1)

    if not np.array_equal(topk1, expected):
        raise AssertionError("top_k=1 sampling should match greedy continuation")

    np.random.seed(42)
    sampled_low_temp = model.generate(
        prompt,
        max_new_tokens=4,
        do_sample=True,
        top_k=5,
        temperature=0.7,
    )
    print()
    print("sampled temperature=0.7 top_k=5:")
    print(sampled_low_temp)

    np.random.seed(42)
    sampled_high_temp = model.generate(
        prompt,
        max_new_tokens=4,
        do_sample=True,
        top_k=5,
        temperature=1.5,
    )
    print()
    print("sampled temperature=1.5 top_k=5:")
    print(sampled_high_temp)

    assert sampled_low_temp.shape == expected.shape
    assert sampled_high_temp.shape == expected.shape

    print()
    print("[OK] token LM trained sampling demo")


if __name__ == "__main__":
    main()