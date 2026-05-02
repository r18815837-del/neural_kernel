import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.modules import TokenTransformerLM
from kernel.utils import set_seed


def make_toy_sequences(num_samples=256, seq_len=12, vocab_size=10, seed=42):
    rng = np.random.default_rng(seed)
    xs = []

    for _ in range(num_samples):
        start = rng.integers(0, vocab_size)
        seq = [(start + i) % vocab_size for i in range(seq_len)]
        xs.append(seq)

    xs = np.asarray(xs, dtype=np.int64)
    return xs


def batch_iter(x, batch_size=32, shuffle=True, seed=42):
    indices = np.arange(len(x))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, len(x), batch_size):
        batch_idx = indices[start:start + batch_size]
        yield x[batch_idx]


def decode_tokens(tokens):
    return " ".join(str(int(t)) for t in tokens)


def main():
    set_seed(42)

    vocab_size = 10
    seq_len = 12

    train_data = make_toy_sequences(
        num_samples=512,
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=42,
    )

    model = TokenTransformerLM(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        max_len=seq_len + 8,
        tie_embeddings=True,
    )

    # local imports to avoid relying on incomplete root exports
    from kernel.nn import CrossEntropyLoss
    from kernel.optim import Adam

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-2)

    epochs = 10
    batch_size = 32

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for xb in batch_iter(train_data, batch_size=batch_size, shuffle=True, seed=epoch):
            # language modeling setup:
            # input  = tokens[:-1]
            # target = tokens[1:]
            inp = xb[:, :-1]
            tgt = xb[:, 1:]

            tokens = Tensor(inp)
            targets = Tensor(tgt)

            optimizer.zero_grad()

            logits, _ = model(tokens)

            # flatten (B, T, V) -> (B*T, V)
            b, t, v = logits.shape
            logits_flat = logits.reshape(b * t, v)
            targets_flat = targets.reshape(b * t)

            loss = criterion(logits_flat, targets_flat)
            loss.backward()
            optimizer.step()

            loss_np = loss.data if hasattr(loss, "data") else np.array(loss)
            losses.append(float(np.array(loss_np)))

        print(f"Epoch {epoch:02d} | train_loss={np.mean(losses):.4f}")

    model.eval()

    prompt = np.array([[3, 4, 5]], dtype=np.int64)

    greedy = model.generate(
        prompt,
        max_new_tokens=6,
        do_sample=False,
    )

    sampled = model.generate(
        prompt,
        max_new_tokens=6,
        do_sample=True,
        temperature=1.0,
        top_k=3,
    )

    sampled_top_p = model.generate(
        prompt,
        max_new_tokens=6,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
    )

    print("\nPrompt:")
    print(decode_tokens(prompt[0]))

    print("\nGreedy generation:")
    print(decode_tokens(greedy[0]))

    print("\nTop-k sampling:")
    print(decode_tokens(sampled[0]))

    print("\nTop-p sampling:")
    print(decode_tokens(sampled_top_p[0]))


if __name__ == "__main__":
    main()