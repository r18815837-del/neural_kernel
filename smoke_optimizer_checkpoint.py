import os
import tempfile

import numpy as np

from kernel.nn.losses import CrossEntropyLoss
from kernel.nn.modules.token_lm import TokenTransformerLM
from kernel.optim.adam import Adam
from kernel.utils.checkpoint import save_checkpoint, load_checkpoint


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def make_batch(batch_size=8, seq_len=6, vocab_size=20):
    starts = np.random.randint(0, vocab_size, size=(batch_size, 1), dtype=np.int64)
    offsets = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)
    token_ids = (starts + offsets) % vocab_size
    return token_ids.astype(np.int64)


def train_one_step(model, optimizer, loss_fn, token_ids):
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

    return float(loss.detach().numpy())


def smoke_optimizer_checkpoint_roundtrip():
    np.random.seed(42)

    model_a = TokenTransformerLM(
        vocab_size=20,
        d_model=16,
        num_heads=4,
        d_ff=32,
        num_layers=2,
        dropout_p=0.0,
        max_len=32,
        activation="gelu",
        tie_embeddings=True,
    )
    optimizer_a = Adam(model_a.parameters(), lr=1e-2)
    loss_fn = CrossEntropyLoss()

    token_ids = make_batch(batch_size=8, seq_len=6, vocab_size=20)
    _ = train_one_step(model_a, optimizer_a, loss_fn, token_ids)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "resume_checkpoint.pkl")
        save_checkpoint(
            model_a,
            path,
            optimizer=optimizer_a,
            meta={"epoch": 3, "note": "resume-test"},
        )

        model_b = TokenTransformerLM(
            vocab_size=20,
            d_model=16,
            num_heads=4,
            d_ff=32,
            num_layers=2,
            dropout_p=0.0,
            max_len=32,
            activation="gelu",
            tie_embeddings=True,
        )
        optimizer_b = Adam(model_b.parameters(), lr=1e-2)

        meta = load_checkpoint(model_b, path, optimizer=optimizer_b)

    assert meta["epoch"] == 3
    assert meta["note"] == "resume-test"

    token_ids_eval = make_batch(batch_size=4, seq_len=6, vocab_size=20)

    logits_a, _ = model_a(token_ids_eval, use_causal_mask=True)
    logits_b, _ = model_b(token_ids_eval, use_causal_mask=True)

    assert np.allclose(logits_a.numpy(), logits_b.numpy(), atol=1e-8)
    assert optimizer_a.t == optimizer_b.t
    assert optimizer_a.lr == optimizer_b.lr


def smoke_missing_optimizer_state():
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
    optimizer = Adam(model.parameters(), lr=1e-2)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model_only.pkl")
        save_checkpoint(model, path, optimizer=None)

        failed = False
        try:
            load_checkpoint(model, path, optimizer=optimizer)
        except KeyError:
            failed = True

    assert failed, "Expected KeyError when optimizer_state is missing"


def smoke_resume_training_step():
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
        tie_embeddings=True,
    )
    optimizer = Adam(model.parameters(), lr=1e-2)
    loss_fn = CrossEntropyLoss()

    token_ids = make_batch(batch_size=8, seq_len=6, vocab_size=20)
    loss_before = train_one_step(model, optimizer, loss_fn, token_ids)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "resume.pkl")
        save_checkpoint(model, path, optimizer=optimizer, meta={"step": 1})

        model2 = TokenTransformerLM(
            vocab_size=20,
            d_model=16,
            num_heads=4,
            d_ff=32,
            num_layers=2,
            dropout_p=0.0,
            max_len=32,
            activation="gelu",
            tie_embeddings=True,
        )
        optimizer2 = Adam(model2.parameters(), lr=1e-2)
        _ = load_checkpoint(model2, path, optimizer=optimizer2)

        loss_after = train_one_step(model2, optimizer2, loss_fn, token_ids)

    assert np.isfinite(loss_before)
    assert np.isfinite(loss_after)


def main():
    check("optimizer checkpoint roundtrip", smoke_optimizer_checkpoint_roundtrip)
    check("missing optimizer state", smoke_missing_optimizer_state)
    check("resume training step", smoke_resume_training_step)


if __name__ == "__main__":
    main()