"""Neural Kernel LM backend — loads a trained TokenTransformerLM and generates text.

This bridges the trained language model into the cognition pipeline.
The orchestrator uses it in the REASON stage to generate answers instead
of relying on hardcoded templates.

Usage::

    backend = LMBackend.from_checkpoint("checkpoints/lm", device="cpu")
    text = backend.generate("The capital of France is", max_tokens=30)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class LMBackend:
    """Wraps a trained TokenTransformerLM + BPETokenizer for text generation."""

    def __init__(self, model, tokenizer, device: str = "cpu") -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str = "checkpoints/lm",
        device: str = "cpu",
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 320,
    ) -> Optional["LMBackend"]:
        """Load a trained model from checkpoint directory.

        Returns None if checkpoint or tokenizer not found.
        """
        from kernel.nn.modules.token_lm import TokenTransformerLM
        from kernel.tokenization.bpe_tokenizer import BPETokenizer

        ckpt_dir = Path(checkpoint_dir)
        tokenizer_path = ckpt_dir / "tokenizer.json"
        model_path = ckpt_dir / "nk_lm_best.npz"

        # Fall back to final if best doesn't exist.
        if not model_path.exists():
            model_path = ckpt_dir / "nk_lm_final.npz"

        if not tokenizer_path.exists():
            log.warning("lm_backend: tokenizer not found at %s", tokenizer_path)
            return None
        if not model_path.exists():
            log.warning("lm_backend: model checkpoint not found at %s", model_path)
            return None

        # Load tokenizer.
        tokenizer = BPETokenizer.load(str(tokenizer_path))
        vocab_size = tokenizer.info().vocab_size

        # Create model with same architecture.
        model = TokenTransformerLM(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout_p=0.0,
            max_len=max_len,
            activation="gelu",
        )

        # Load weights.
        data = np.load(str(model_path))
        for param_name, param in model.named_parameters():
            if param_name in data:
                param.data = data[param_name].astype(np.float32)

        model.eval()
        log.info(
            "lm_backend: loaded model (%s, d=%d, layers=%d) from %s",
            vocab_size, d_model, num_layers, model_path,
        )

        return cls(model, tokenizer, device)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> str:
        """Generate text continuation from a prompt."""
        ids = self._tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = np.array([ids], dtype=np.int64)

        generated = self._model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
        )

        output = self._tokenizer.decode(
            generated[0].tolist(),
            skip_special_tokens=True,
        )
        return output.strip()

    def complete(self, context: str, question: str, max_tokens: int = 80) -> str:
        """Generate an answer given context and question.

        Formats a simple prompt: "Context: ... Question: ... Answer:"
        and lets the model continue.
        """
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        raw = self.generate(prompt, max_tokens=max_tokens, temperature=0.7)

        # Extract just the answer part (after "Answer:" if model repeats it).
        if "Answer:" in raw:
            parts = raw.split("Answer:")
            return parts[-1].strip()
        return raw

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
