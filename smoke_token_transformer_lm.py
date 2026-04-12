import numpy as np

from kernel.nn.losses import CrossEntropyLoss
from kernel.nn.modules.token_lm import TokenTransformerLM
from kernel.nn.functional.masks import make_padding_mask


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_lm_forward():
    token_ids = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int64)

    model = TokenTransformerLM(
        vocab_size=10,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout_p=0.0,
        max_len=16,
        activation="gelu",
    )

    logits, attn_all = model(token_ids)

    assert logits.shape == (2, 4, 10)
    assert len(attn_all) == 2
    for attn in attn_all:
        assert attn.shape == (2, 2, 4, 4)


def smoke_lm_backward_next_token():
    token_ids = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int64)

    model = TokenTransformerLM(
        vocab_size=10,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout_p=0.0,
        max_len=16,
        activation="gelu",
    )

    loss_fn = CrossEntropyLoss()

    logits, _ = model(token_ids)                  # (B, T, V)

    input_logits = logits[:, :-1, :]              # predict next token
    targets = token_ids[:, 1:]                    # next-token targets

    B, Tm1, V = input_logits.shape
    flat_logits = input_logits.reshape(B * Tm1, V)
    flat_targets = targets.reshape(B * Tm1)

    loss = loss_fn(flat_logits, flat_targets)
    loss.backward()

    assert model.embedding.weight.grad is not None
    assert model.lm_head.weight.grad is not None


def smoke_lm_with_padding_mask():
    token_ids = np.array([[1, 2, 3, 4], [4, 3, 0, 0]], dtype=np.int64)
    padding_mask = make_padding_mask([4, 2], max_len=4)

    model = TokenTransformerLM(
        vocab_size=10,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout_p=0.0,
        max_len=16,
        activation="gelu",
    )

    logits, attn_all = model(token_ids, mask=padding_mask, use_causal_mask=True)

    assert logits.shape == (2, 4, 10)
    assert len(attn_all) == 2
    for attn in attn_all:
        assert attn.shape == (2, 2, 4, 4)


def smoke_lm_disable_causal_mask():
    token_ids = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int64)

    model = TokenTransformerLM(
        vocab_size=10,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout_p=0.0,
        max_len=16,
        activation="gelu",
    )

    logits, attn_all = model(token_ids, use_causal_mask=False)

    assert logits.shape == (2, 4, 10)
    assert len(attn_all) == 2


def main():
    check("lm forward", smoke_lm_forward)
    check("lm backward next-token", smoke_lm_backward_next_token)
    check("lm with padding mask", smoke_lm_with_padding_mask)
    check("lm disable causal mask", smoke_lm_disable_causal_mask)


if __name__ == "__main__":
    main()