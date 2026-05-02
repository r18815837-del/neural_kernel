"""
04_train_lm.py
==============

Главный тренировочный цикл для TokenTransformerLM.

Особенности:
  * Читает packed dataset через np.memmap → не грузит весь корпус в память.
  * Random-window batching как в nanoGPT.
  * Warmup + cosine LR schedule.
  * Gradient clipping по глобальной норме.
  * Валидация (loss + perplexity) каждые `eval_every` шагов.
  * Periodic sampling — печатает короткий sample прямо во время тренировки.
  * Чекпоинты + ротация (хранит только последние N).
  * GPU через CuPy backend (если device='cuda' и CuPy установлен).

Usage:
    # Базовый запуск (берёт всё из training.llm.config)
    python training/llm/04_train_lm.py

    # Быстрая проверка на CPU (mini smoke test)
    python training/llm/04_train_lm.py --device cpu --max-steps 50 --eval-every 25 \
        --batch-size 4 --block-size 64 --d-model 64 --num-layers 2 --num-heads 4

    # Резюме с последнего чекпоинта
    python training/llm/04_train_lm.py --resume
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kernel.backend import is_cuda_available  # noqa: E402
from kernel.core.tensor import Tensor  # noqa: E402
from kernel.nn.losses import CrossEntropyLoss  # noqa: E402
from kernel.nn.modules.token_lm import TokenTransformerLM  # noqa: E402
from kernel.optim.adam import Adam  # noqa: E402
from kernel.optim.grad_clip import clip_grad_norm_  # noqa: E402
from kernel.tokenization.bpe_tokenizer import BPETokenizer  # noqa: E402
from kernel.utils import set_seed  # noqa: E402
from kernel.utils.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402

from training.llm.config import (  # noqa: E402
    CHECKPOINT_DIR,
    CONFIG,
    LOG_DIR,
    META_PATH,
    TOKENIZER_PATH,
    TRAIN_BIN,
    VAL_BIN,
)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_packed(path: Path) -> np.ndarray:
    """np.memmap для не-прожорливого случайного доступа к корпусу."""
    if not path.exists():
        raise FileNotFoundError(f"Packed dataset missing: {path}. Run 03_pack_dataset.py first.")
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data: np.ndarray, batch_size: int, block_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample random contiguous windows. Returns (X, Y) — (B, T) int64.

    Y = X сдвинутый на один токен (next-token prediction).
    """
    if len(data) <= block_size + 1:
        raise ValueError(
            f"Packed dataset too small ({len(data)} tokens) for block_size={block_size}."
        )

    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([np.asarray(data[s : s + block_size], dtype=np.int64) for s in starts])
    y = np.stack([np.asarray(data[s + 1 : s + 1 + block_size], dtype=np.int64) for s in starts])
    return x, y


# ----------------------------------------------------------------------
# LR schedule (warmup → cosine decay)
# ----------------------------------------------------------------------
def lr_at(step: int, max_lr: float, min_lr: float, warmup: int, total_steps: int) -> float:
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ----------------------------------------------------------------------
# Eval
# ----------------------------------------------------------------------
def estimate_loss(
    model: TokenTransformerLM,
    data: np.ndarray,
    n_batches: int,
    batch_size: int,
    block_size: int,
    device: str,
    rng: np.random.Generator,
    loss_fn: CrossEntropyLoss,
) -> float:
    was_training = model.training
    model.eval()

    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, block_size, rng)
        x_t = Tensor(x, device=device)
        y_t = Tensor(y, device=device)

        logits, _ = model(x_t, use_causal_mask=True)
        b, t, v = logits.shape
        loss = loss_fn(logits.reshape(b * t, v), y_t.reshape(b * t))
        losses.append(float(np.array(loss.detach().numpy())))

    if was_training:
        model.train()

    return float(np.mean(losses))


# ----------------------------------------------------------------------
# Checkpoint rotation
# ----------------------------------------------------------------------
def rotate_checkpoints(directory: Path, prefix: str, keep: int) -> None:
    files = sorted(
        directory.glob(f"{prefix}_step_*.pkl"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if len(files) > keep:
        for f in files[:-keep]:
            try:
                f.unlink()
            except OSError:
                pass


# ----------------------------------------------------------------------
# Main training
# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Train TokenTransformerLM.")

    # Hardware
    parser.add_argument("--device", default=CONFIG.train.device, choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=CONFIG.train.seed)

    # Optim
    parser.add_argument("--batch-size", type=int, default=CONFIG.train.batch_size)
    parser.add_argument("--block-size", type=int, default=CONFIG.train.block_size)
    parser.add_argument("--learning-rate", type=float, default=CONFIG.train.learning_rate)
    parser.add_argument("--min-lr", type=float, default=CONFIG.train.min_lr)
    parser.add_argument("--grad-clip", type=float, default=CONFIG.train.grad_clip)

    # Schedule
    parser.add_argument("--max-steps", type=int, default=CONFIG.train.max_steps)
    parser.add_argument("--warmup-steps", type=int, default=CONFIG.train.warmup_steps)
    parser.add_argument("--log-every", type=int, default=CONFIG.train.log_every)
    parser.add_argument("--eval-every", type=int, default=CONFIG.train.eval_every)
    parser.add_argument("--eval-iters", type=int, default=CONFIG.train.eval_iters)
    parser.add_argument("--save-every", type=int, default=CONFIG.train.save_every)
    parser.add_argument("--keep-last", type=int, default=CONFIG.train.keep_last_n_checkpoints)

    # Sampling
    parser.add_argument("--sample-every", type=int, default=CONFIG.train.sample_every)
    parser.add_argument("--sample-prompt", default=CONFIG.train.sample_prompt)
    parser.add_argument("--sample-max-new-tokens", type=int, default=CONFIG.train.sample_max_new_tokens)
    parser.add_argument("--sample-temperature", type=float, default=CONFIG.train.sample_temperature)
    parser.add_argument("--sample-top-k", type=int, default=CONFIG.train.sample_top_k)

    # Model overrides
    parser.add_argument("--d-model", type=int, default=CONFIG.model.d_model)
    parser.add_argument("--num-heads", type=int, default=CONFIG.model.num_heads)
    parser.add_argument("--d-ff", type=int, default=CONFIG.model.d_ff)
    parser.add_argument("--num-layers", type=int, default=CONFIG.model.num_layers)
    parser.add_argument("--dropout", type=float, default=CONFIG.model.dropout_p)
    parser.add_argument("--max-len", type=int, default=CONFIG.model.max_len)
    parser.add_argument("--no-tie", action="store_true", help="Disable weight tying.")

    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    parser.add_argument("--run-name", default="nk_lm", help="Prefix for checkpoints.")

    args = parser.parse_args()

    # ---------- Setup ----------
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.device == "cuda" and not is_cuda_available():
        print("[warn] CUDA requested but CuPy is unavailable. Falling back to CPU.")
        args.device = "cpu"

    if not META_PATH.exists():
        raise FileNotFoundError(f"meta.json missing: {META_PATH}. Run 03_pack_dataset.py first.")

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    vocab_size = int(meta["vocab_size"])

    if args.block_size > args.max_len:
        raise ValueError(
            f"block_size ({args.block_size}) > max_len ({args.max_len}). "
            f"Either lower block_size or raise max_len."
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Data ----------
    print(f"[data] loading {TRAIN_BIN.name} and {VAL_BIN.name}")
    train_data = load_packed(TRAIN_BIN)
    val_data = load_packed(VAL_BIN)
    print(f"  train: {len(train_data):,} tokens")
    print(f"  val  : {len(val_data):,} tokens")

    # ---------- Model ----------
    print(f"\n[model] device={args.device}  vocab={vocab_size}  "
          f"d_model={args.d_model}  layers={args.num_layers}  heads={args.num_heads}")

    model = TokenTransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        dropout_p=args.dropout,
        max_len=args.max_len,
        activation=CONFIG.model.activation,
        tie_embeddings=not args.no_tie,
    )

    if args.device == "cuda":
        model.to("cuda")

    n_params = sum(p.data.size for p in model.parameters())
    print(f"  params: {n_params:,} ({n_params / 1e6:.2f}M)")

    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=CONFIG.train.betas,
    )
    loss_fn = CrossEntropyLoss()

    # ---------- Resume ----------
    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpts = sorted(
            CHECKPOINT_DIR.glob(f"{args.run_name}_step_*.pkl"),
            key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
        )
        if not ckpts:
            print("[resume] no checkpoint found, starting fresh.")
        else:
            latest = ckpts[-1]
            print(f"[resume] loading {latest}")
            payload_meta = load_checkpoint(model, latest, optimizer=optimizer)
            start_step = int(payload_meta.get("step", 0))
            best_val_loss = float(payload_meta.get("best_val_loss", best_val_loss))
            print(f"  resumed at step {start_step}, best_val_loss={best_val_loss:.4f}")

    # Optional tokenizer for periodic sampling.
    tokenizer = None
    if TOKENIZER_PATH.exists() and args.sample_every > 0:
        try:
            tokenizer = BPETokenizer.load(str(TOKENIZER_PATH))
        except Exception as e:
            print(f"[warn] failed to load tokenizer for sampling: {e}")
            tokenizer = None

    # ---------- Logging ----------
    log_path = LOG_DIR / f"{args.run_name}_train.log.jsonl"
    log_f = open(log_path, "a", encoding="utf-8")

    def log_event(event: dict) -> None:
        event = {"ts": time.time(), **event}
        log_f.write(json.dumps(event) + "\n")
        log_f.flush()

    log_event({"event": "start", "args": vars(args), "meta": meta})

    # ---------- Train loop ----------
    print(f"\n[train] starting at step {start_step}, target {args.max_steps}")
    print("=" * 80)

    t0 = time.time()
    last_log_t = t0

    try:
        for step in range(start_step, args.max_steps):
            # LR schedule.
            lr = lr_at(step, args.learning_rate, args.min_lr, args.warmup_steps, args.max_steps)
            optimizer.lr = lr

            x_np, y_np = get_batch(train_data, args.batch_size, args.block_size, rng)
            x_t = Tensor(x_np, device=args.device)
            y_t = Tensor(y_np, device=args.device)

            model.train()
            optimizer.zero_grad()
            logits, _ = model(x_t, use_causal_mask=True)
            b, t, v = logits.shape
            loss = loss_fn(logits.reshape(b * t, v), y_t.reshape(b * t))
            loss.backward()

            if args.grad_clip > 0:
                clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()

            # Logging.
            if (step + 1) % args.log_every == 0:
                loss_val = float(np.array(loss.detach().numpy()))
                now = time.time()
                tok_per_s = args.batch_size * args.block_size * args.log_every / max(now - last_log_t, 1e-6)
                last_log_t = now

                msg = (
                    f"step {step+1:6d}/{args.max_steps} | "
                    f"loss {loss_val:7.4f} | "
                    f"lr {lr:8.2e} | "
                    f"tok/s {tok_per_s:8.0f} | "
                    f"elapsed {(now - t0) / 60:5.1f}m"
                )
                print(msg)
                log_event({"event": "step", "step": step + 1, "loss": loss_val, "lr": lr, "tok_per_s": tok_per_s})

            # Validation.
            if (step + 1) % args.eval_every == 0 or (step + 1) == args.max_steps:
                val_loss = estimate_loss(
                    model, val_data, args.eval_iters,
                    args.batch_size, args.block_size, args.device, rng, loss_fn,
                )
                ppl = math.exp(min(val_loss, 20.0))
                marker = ""
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    marker = "  ← new best"
                    save_checkpoint(
                        model,
                        CHECKPOINT_DIR / f"{args.run_name}_best.pkl",
                        optimizer=optimizer,
                        meta={
                            "step": step + 1,
                            "val_loss": val_loss,
                            "best_val_loss": best_val_loss,
                            "model_config": asdict(CONFIG.model) | {
                                "vocab_size": vocab_size,
                                "d_model": args.d_model,
                                "num_heads": args.num_heads,
                                "d_ff": args.d_ff,
                                "num_layers": args.num_layers,
                                "dropout_p": args.dropout,
                                "max_len": args.max_len,
                                "tie_embeddings": not args.no_tie,
                            },
                        },
                    )
                print(f"  ↳ val_loss {val_loss:.4f} | ppl {ppl:.2f}{marker}")
                log_event({"event": "eval", "step": step + 1, "val_loss": val_loss, "ppl": ppl})

            # Periodic sample.
            if tokenizer is not None and args.sample_every > 0 and (step + 1) % args.sample_every == 0:
                try:
                    prompt_ids = tokenizer.encode(args.sample_prompt, add_special_tokens=False)
                    if not prompt_ids:
                        prompt_ids = [tokenizer._token2id["<bos>"]]  # type: ignore[attr-defined]
                    prompt = np.array([prompt_ids], dtype=np.int64)
                    out = model.generate(
                        prompt,
                        max_new_tokens=args.sample_max_new_tokens,
                        do_sample=True,
                        temperature=args.sample_temperature,
                        top_k=args.sample_top_k,
                    )
                    text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
                    print(f"  ↳ sample: {text!r}")
                    log_event({"event": "sample", "step": step + 1, "text": text})
                except Exception as e:
                    print(f"  ↳ sample failed: {e}")

            # Periodic checkpoint.
            if args.save_every > 0 and (step + 1) % args.save_every == 0:
                ckpt_path = CHECKPOINT_DIR / f"{args.run_name}_step_{step+1:07d}.pkl"
                save_checkpoint(
                    model, ckpt_path, optimizer=optimizer,
                    meta={
                        "step": step + 1,
                        "best_val_loss": best_val_loss,
                    },
                )
                rotate_checkpoints(CHECKPOINT_DIR, args.run_name, args.keep_last)

    except KeyboardInterrupt:
        print("\n[interrupt] user requested stop. Saving current state…")
        save_checkpoint(
            model,
            CHECKPOINT_DIR / f"{args.run_name}_interrupt.pkl",
            optimizer=optimizer,
            meta={"step": step + 1 if 'step' in locals() else 0,
                  "best_val_loss": best_val_loss},
        )

    finally:
        log_event({"event": "end", "best_val_loss": best_val_loss})
        log_f.close()

    print("\n" + "=" * 80)
    print(f"done | best_val_loss = {best_val_loss:.4f} | best ckpt: {CHECKPOINT_DIR / (args.run_name + '_best.pkl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
