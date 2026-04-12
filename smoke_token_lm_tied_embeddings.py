import numpy as np

from kernel.nn.losses import CrossEntropyLoss
from kernel.nn.modules.token_lm import TokenTransformerLM


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_tied_lm_forward():
    token_ids = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int64)

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

    logits, attn_all = model(token_ids, use_causal_mask=True)

    assert logits.shape == (2, 4, 20)
    assert len(attn_all) == 2
    assert model.lm_head is None


def smoke_tied_lm_backward():
    token_ids = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int64)

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

    loss_fn = CrossEntropyLoss()

    logits, _ = model(token_ids, use_causal_mask=True)
    input_logits = logits[:, :-1, :]
    targets = token_ids[:, 1:]

    B, Tm1, V = input_logits.shape
    flat_logits = input_logits.reshape(B * Tm1, V)
    flat_targets = targets.reshape(B * Tm1)

    loss = loss_fn(flat_logits, flat_targets)
    loss.backward()

    assert model.embedding.weight.grad is not None
    assert model.embedding.weight.grad.shape == model.embedding.weight.shape


def smoke_tied_lm_generate():
    token_ids = np.array([[1, 2, 3]], dtype=np.int64)

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

    out = model.generate(token_ids, max_new_tokens=3)
    assert out.shape == (1, 6)


def smoke_untied_lm_still_works():
    token_ids = np.array([[1, 2, 3]], dtype=np.int64)

    model = TokenTransformerLM(
        vocab_size=20,
        d_model=16,
        num_heads=4,
        d_ff=32,
        num_layers=2,
        dropout_p=0.0,
        max_len=32,
        activation="gelu",
        tie_embeddings=False,
    )

    logits, _ = model(token_ids, use_causal_mask=True)
    assert logits.shape == (1, 3, 20)
    assert model.lm_head is not None


def main():
    check("tied lm forward", smoke_tied_lm_forward)
    check("tied lm backward", smoke_tied_lm_backward)
    check("tied lm generate", smoke_tied_lm_generate)
    check("untied lm still works", smoke_untied_lm_still_works)


if __name__ == "__main__":
    main()