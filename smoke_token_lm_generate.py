import numpy as np

from kernel.nn.modules.token_lm import TokenTransformerLM


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_generate_shape():
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

    token_ids = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    out = model.generate(token_ids, max_new_tokens=4)

    assert out.shape == (2, 7)


def smoke_generate_dtype():
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

    token_ids = np.array([[1, 2, 3]], dtype=np.int64)
    out = model.generate(token_ids, max_new_tokens=2)

    assert out.dtype.kind in {"i", "u"}


def main():
    check("generate shape", smoke_generate_shape)
    check("generate dtype", smoke_generate_dtype)


if __name__ == "__main__":
    main()