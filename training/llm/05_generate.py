"""
05_generate.py
==============

Загружает обученный чекпоинт + tokenizer и генерирует текст по промпту.

Поддерживает greedy / top-k / top-p / temperature и несколько
параллельных промптов в одном батче.

Usage:
    python training/llm/05_generate.py --prompt "Once upon a time"
    python training/llm/05_generate.py --prompt "The dragon" --top-k 50 --temperature 0.8
    python training/llm/05_generate.py --prompt "Hello" --max-new-tokens 200 --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kernel.backend import is_cuda_available  # noqa: E402
from kernel.nn.modules.token_lm import TokenTransformerLM  # noqa: E402
from kernel.tokenization.bpe_tokenizer import BPETokenizer  # noqa: E402
from kernel.utils.checkpoint import load_checkpoint  # noqa: E402

from training.llm.config import CHECKPOINT_DIR, TOKENIZER_PATH  # noqa: E402


def build_model_from_meta(meta: dict) -> TokenTransformerLM:
    cfg = meta.get("model_config")
    if not cfg:
        raise RuntimeError(
            "Checkpoint meta has no 'model_config'. Make sure you trained with the new script."
        )

    return TokenTransformerLM(
        vocab_size=int(cfg["vocab_size"]),
        d_model=int(cfg["d_model"]),
        num_heads=int(cfg["num_heads"]),
        d_ff=int(cfg["d_ff"]),
        num_layers=int(cfg["num_layers"]),
        dropout_p=float(cfg.get("dropout_p", 0.0)),
        max_len=int(cfg["max_len"]),
        activation=cfg.get("activation", "gelu"),
        tie_embeddings=bool(cfg.get("tie_embeddings", True)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--ckpt", default=str(CHECKPOINT_DIR / "nk_lm_best.pkl"))
    parser.add_argument("--tokenizer", default=str(TOKENIZER_PATH))
    parser.add_argument("--prompt", required=True, help="Initial text to continue.")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-samples", type=int, default=1, help="How many continuations to generate.")
    parser.add_argument("--greedy", action="store_true", help="Disable sampling (argmax).")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.device == "cuda" and not is_cuda_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    tok_path = Path(args.tokenizer)
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")

    print(f"[load] tokenizer: {tok_path}")
    tok = BPETokenizer.load(str(tok_path))

    print(f"[load] checkpoint: {ckpt_path}")
    # Сначала смотрим метаданные, чтобы знать архитектуру.
    import pickle
    with open(ckpt_path, "rb") as f:
        payload = pickle.load(f)
    meta = payload.get("meta", {})
    model = build_model_from_meta(meta)

    if args.device == "cuda":
        model.to("cuda")

    load_checkpoint(model, ckpt_path)
    model.eval()

    n_params = sum(p.data.size for p in model.parameters())
    print(f"  params: {n_params:,} ({n_params / 1e6:.2f}M)")
    print(f"  trained step: {meta.get('step', '?')}, best_val_loss: {meta.get('best_val_loss', '?')}")
    print()

    prompt_ids = tok.encode(args.prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("Empty prompt after tokenization.")

    # Дублируем промпт num_samples раз, чтобы получить разные продолжения за один проход.
    batch = np.tile(np.array(prompt_ids, dtype=np.int64), (args.num_samples, 1))

    out = model.generate(
        batch,
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.greedy,
        temperature=args.temperature,
        top_k=None if args.greedy else args.top_k,
        top_p=args.top_p,
    )

    print(f"=== prompt ===\n{args.prompt}\n")
    for i in range(args.num_samples):
        text = tok.decode(out[i].tolist(), skip_special_tokens=True)
        print(f"=== sample {i + 1}/{args.num_samples} ===")
        print(text)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
