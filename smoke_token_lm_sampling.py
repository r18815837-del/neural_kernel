import numpy as np

from kernel.nn.modules.token_lm import TokenTransformerLM


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def make_model():
    return TokenTransformerLM(
        vocab_size=20,
        d_model=16,
        num_heads=4,
        d_ff=32,
        num_layers=2,
        dropout_p=0.0,
        max_len=32,
        activation="gelu",
    )


def smoke_generate_greedy():
    model = make_model()
    token_ids = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)

    out = model.generate(token_ids, max_new_tokens=4)
    assert out.shape == (2, 7)


def smoke_generate_temperature():
    model = make_model()
    token_ids = np.array([[1, 2, 3]], dtype=np.int64)

    out = model.generate(
        token_ids,
        max_new_tokens=4,
        temperature=0.8,
        do_sample=False,
    )
    assert out.shape == (1, 7)


def smoke_generate_sampling():
    np.random.seed(42)

    model = make_model()
    token_ids = np.array([[1, 2, 3]], dtype=np.int64)

    out = model.generate(
        token_ids,
        max_new_tokens=4,
        temperature=1.0,
        do_sample=True,
    )
    assert out.shape == (1, 7)


def smoke_generate_top_k_sampling():
    np.random.seed(42)

    model = make_model()
    token_ids = np.array([[1, 2, 3]], dtype=np.int64)

    out = model.generate(
        token_ids,
        max_new_tokens=4,
        temperature=0.9,
        top_k=5,
        do_sample=True,
    )
    assert out.shape == (1, 7)


def smoke_invalid_temperature():
    model = make_model()
    token_ids = np.array([[1, 2, 3]], dtype=np.int64)

    failed = False
    try:
        _ = model.generate(token_ids, max_new_tokens=4, temperature=0.0)
    except ValueError:
        failed = True

    assert failed, "Expected ValueError for invalid temperature"


def smoke_invalid_top_k():
    model = make_model()
    token_ids = np.array([[1, 2, 3]], dtype=np.int64)

    failed = False
    try:
        _ = model.generate(token_ids, max_new_tokens=4, top_k=0, do_sample=True)
    except ValueError:
        failed = True

    assert failed, "Expected ValueError for invalid top_k"


def main():
    check("generate greedy", smoke_generate_greedy)
    check("generate temperature", smoke_generate_temperature)
    check("generate sampling", smoke_generate_sampling)
    check("generate top_k sampling", smoke_generate_top_k_sampling)
    check("invalid temperature", smoke_invalid_temperature)
    check("invalid top_k", smoke_invalid_top_k)


if __name__ == "__main__":
    main()