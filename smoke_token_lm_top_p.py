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
        tie_embeddings=True,
    )


def smoke_generate_top_p_sampling():
    np.random.seed(42)

    model = make_model()
    token_ids = np.array([[1, 2, 3]], dtype=np.int64)

    out = model.generate(
        token_ids,
        max_new_tokens=4,
        do_sample=True,
        top_p=0.9,
        temperature=1.0,
    )
    assert out.shape == (1, 7)


def smoke_generate_top_k_top_p_sampling():
    np.random.seed(42)

    model = make_model()
    token_ids = np.array([[1, 2, 3]], dtype=np.int64)

    out = model.generate(
        token_ids,
        max_new_tokens=4,
        do_sample=True,
        top_k=5,
        top_p=0.8,
        temperature=1.0,
    )
    assert out.shape == (1, 7)


def smoke_generate_top_p_greedy_mode():
    model = make_model()
    token_ids = np.array([[1, 2, 3]], dtype=np.int64)

    out = model.generate(
        token_ids,
        max_new_tokens=4,
        do_sample=False,
        top_p=0.9,
    )
    assert out.shape == (1, 7)


def smoke_invalid_top_p_zero():
    model = make_model()
    token_ids = np.array([[1, 2, 3]], dtype=np.int64)

    failed = False
    try:
        _ = model.generate(
            token_ids,
            max_new_tokens=4,
            do_sample=True,
            top_p=0.0,
        )
    except ValueError:
        failed = True

    assert failed, "Expected ValueError for top_p=0.0"


def smoke_invalid_top_p_gt_one():
    model = make_model()
    token_ids = np.array([[1, 2, 3]], dtype=np.int64)

    failed = False
    try:
        _ = model.generate(
            token_ids,
            max_new_tokens=4,
            do_sample=True,
            top_p=1.5,
        )
    except ValueError:
        failed = True

    assert failed, "Expected ValueError for top_p>1"


def main():
    check("generate top_p sampling", smoke_generate_top_p_sampling)
    check("generate top_k + top_p sampling", smoke_generate_top_k_top_p_sampling)
    check("generate top_p in greedy mode", smoke_generate_top_p_greedy_mode)
    check("invalid top_p zero", smoke_invalid_top_p_zero)
    check("invalid top_p > 1", smoke_invalid_top_p_gt_one)


if __name__ == "__main__":
    main()