import numpy as np

from kernel import Tensor
from kernel.nn.modules import TokenTransformerLM


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "data"):
        data = x.data
        if isinstance(data, np.ndarray):
            return data
        try:
            return np.array(data)
        except Exception:
            pass
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.array(x)


def make_lm(vocab_size=20, d_model=8, num_heads=2, d_ff=16, num_layers=2, max_len=16):
    return TokenTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_len=max_len,
    )


def assert_token_range(tokens, vocab_size):
    arr = np.asarray(tokens)
    assert np.issubdtype(arr.dtype, np.integer)
    assert arr.min() >= 0
    assert arr.max() < vocab_size


def test_token_lm_forward_returns_logits_and_attention():
    vocab_size = 20
    model = make_lm(vocab_size=vocab_size)

    tokens = Tensor(np.array([[1, 2, 3, 4]], dtype=np.int64))
    logits, attn_all = model(tokens)

    logits_arr = to_numpy(logits)

    assert logits_arr.shape == (1, 4, vocab_size)
    assert np.isfinite(logits_arr).all()
    assert attn_all is not None


def test_generate_greedy_runs_and_extends_sequence():
    vocab_size = 20
    model = make_lm(vocab_size=vocab_size)

    start = np.array([[1, 2, 3]], dtype=np.int64)
    out = model.generate(start, max_new_tokens=4, do_sample=False)

    assert out.shape == (1, 7)
    assert_token_range(out, vocab_size)


def test_generate_with_tensor_input_runs_and_extends_sequence():
    vocab_size = 20
    model = make_lm(vocab_size=vocab_size)

    start = Tensor(np.array([[1, 2, 3]], dtype=np.int64))
    out = model.generate(start, max_new_tokens=2, do_sample=False)

    assert out.shape == (1, 5)
    assert_token_range(out, vocab_size)


def test_generate_sampling_runs_and_extends_sequence():
    vocab_size = 20
    model = make_lm(vocab_size=vocab_size)

    start = np.array([[1, 2, 3]], dtype=np.int64)
    out = model.generate(
        start,
        max_new_tokens=4,
        temperature=1.0,
        do_sample=True,
    )

    assert out.shape == (1, 7)
    assert_token_range(out, vocab_size)


def test_generate_top_k_sampling_runs():
    vocab_size = 20
    model = make_lm(vocab_size=vocab_size)

    start = np.array([[1, 2, 3]], dtype=np.int64)
    out = model.generate(
        start,
        max_new_tokens=4,
        temperature=1.0,
        top_k=5,
        do_sample=True,
    )

    assert out.shape == (1, 7)
    assert_token_range(out, vocab_size)


def test_generate_top_p_sampling_runs():
    vocab_size = 20
    model = make_lm(vocab_size=vocab_size)

    start = np.array([[1, 2, 3]], dtype=np.int64)
    out = model.generate(
        start,
        max_new_tokens=4,
        temperature=1.0,
        top_p=0.9,
        do_sample=True,
    )

    assert out.shape == (1, 7)
    assert_token_range(out, vocab_size)


def test_generate_restores_training_mode():
    model = make_lm()
    model.train()

    start = np.array([[1, 2, 3]], dtype=np.int64)
    _ = model.generate(start, max_new_tokens=2, do_sample=False)

    assert model.training is True