from __future__ import annotations

import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.module import Module
from kernel.nn.layers.embedding import Embedding
from kernel.nn.layers.linear import Linear
from kernel.nn.modules.encoder import TransformerEncoder
from kernel.nn.functional.masks import make_causal_mask


class TokenTransformerLM(Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout_p: float = 0.0,
        max_len: int = 5000,
        activation: str = "relu",
        tie_embeddings: bool = False,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {vocab_size}")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tie_embeddings = tie_embeddings

        self.embedding = Embedding(vocab_size, d_model)

        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout_p=dropout_p,
            max_len=max_len,
            use_positional_encoding=True,
            activation=activation,
        )

        if tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = Linear(d_model, vocab_size)

    def _build_attention_mask(
        self,
        seq_len: int,
        device: str,
        mask: Tensor | None = None,
        use_causal_mask: bool = True,
    ) -> Tensor | None:
        final_mask = None

        if use_causal_mask:
            final_mask = make_causal_mask(seq_len, device=device)

        if mask is not None:
            if mask.device != device:
                mask = mask.to(device)
            final_mask = mask if final_mask is None else (final_mask + mask)

        return final_mask

    def _project_to_vocab(self, encoded: Tensor) -> Tensor:
        if self.tie_embeddings:
            # encoded: (B, T, D)
            # embedding.weight: (V, D)
            # embedding.weight.T: (D, V)
            return encoded @ self.embedding.weight.T
        return self.lm_head(encoded)

    def forward(
        self,
        token_ids,
        mask: Tensor | None = None,
        use_causal_mask: bool = True,
    ):
        x = self.embedding(token_ids)  # (B, T, D)
        _, seq_len, _ = x.shape

        attn_mask = self._build_attention_mask(
            seq_len=seq_len,
            device=x.device,
            mask=mask,
            use_causal_mask=use_causal_mask,
        )

        encoded, attn_all = self.encoder(x, mask=attn_mask)  # (B, T, D)
        logits = self._project_to_vocab(encoded)             # (B, T, V)

        return logits, attn_all

    def generate(
            self,
            token_ids,
            max_new_tokens: int,
            temperature: float = 1.0,
            top_k: int | None = None,
            top_p: float | None = None,
            do_sample: bool = False,
    ):
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")

        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        if top_k is not None:
            if top_k <= 0:
                raise ValueError(f"top_k must be > 0, got {top_k}")
            top_k = min(top_k, self.vocab_size)

        if top_p is not None:
            if not (0.0 < top_p <= 1.0):
                raise ValueError(f"top_p must be in (0, 1], got {top_p}")

        if isinstance(token_ids, Tensor):
            current = token_ids.detach().numpy().astype(np.int64)
        else:
            current = np.asarray(token_ids, dtype=np.int64)

        if current.ndim != 2:
            raise ValueError(
                f"generate expects token_ids of shape (B, T), got {current.shape}"
            )

        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            logits, _ = self.forward(current, use_causal_mask=True)
            next_logits = logits[:, -1, :].detach().numpy()

            if do_sample:
                scaled = next_logits / temperature

                # top-k filtering
                if top_k is not None:
                    topk_idx = np.argpartition(scaled, -top_k, axis=1)[:, -top_k:]
                    filtered = np.full_like(scaled, -1e9)
                    row_idx = np.arange(scaled.shape[0])[:, None]
                    filtered[row_idx, topk_idx] = scaled[row_idx, topk_idx]
                    scaled = filtered

                # stable softmax
                shifted = scaled - np.max(scaled, axis=1, keepdims=True)
                probs = np.exp(shifted)
                probs = probs / np.sum(probs, axis=1, keepdims=True)

                # top-p filtering
                if top_p is not None:
                    filtered_probs = np.zeros_like(probs)

                    for i in range(probs.shape[0]):
                        sorted_idx = np.argsort(probs[i])[::-1]
                        sorted_probs = probs[i][sorted_idx]
                        cumulative = np.cumsum(sorted_probs)

                        keep_mask = cumulative <= top_p
                        if not np.any(keep_mask):
                            keep_mask[0] = True
                        else:
                            first_exceed = np.argmax(cumulative > top_p)
                            if cumulative[first_exceed] > top_p:
                                keep_mask[first_exceed] = True

                        kept_idx = sorted_idx[keep_mask]
                        filtered_probs[i, kept_idx] = probs[i, kept_idx]

                    probs = filtered_probs
                    probs = probs / np.sum(probs, axis=1, keepdims=True)

                next_token = []
                for i in range(probs.shape[0]):
                    next_token.append(np.random.choice(probs.shape[1], p=probs[i]))
                next_token = np.asarray(next_token, dtype=np.int64)

            else:
                if temperature != 1.0:
                    next_logits = next_logits / temperature
                next_token = next_logits.argmax(axis=1).astype(np.int64)

            next_token = next_token.reshape(current.shape[0], 1)
            current = np.concatenate([current, next_token], axis=1)

        if was_training:
            self.train()

        return current