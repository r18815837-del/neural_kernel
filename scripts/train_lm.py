"""Train the Neural Kernel language model from scratch.

Three-phase pipeline:
  1. Train BPE tokenizer on corpus
  2. Tokenize corpus → numpy arrays
  3. Train TokenTransformerLM with autoregressive loss

Usage (CPU):
    python scripts/train_lm.py --corpus data/corpus/wiki_en.txt --epochs 10

Usage (GPU — RTX 4060 8GB):
    python scripts/train_lm.py --device cuda --corpus data/corpus/wiki_en.txt --epochs 20

Full GPU config:
    python scripts/train_lm.py --device cuda --fp16 --vocab-size 8192 --d-model 256 --num-heads 8 --num-layers 6 --d-ff 1024 --seq-len 256 --batch-size 64 --epochs 30
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kernel.backend import is_cuda_available
from kernel.nn.losses import CrossEntropyLoss
from kernel.nn.modules.token_lm import TokenTransformerLM
from kernel.optim.adam import Adam
from kernel.tokenization.bpe_tokenizer import BPETokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints/lm")
TOKENIZER_PATH = Path("checkpoints/lm/tokenizer.json")


# ------------------------------------------------------------------
# Device helpers
# ------------------------------------------------------------------

def resolve_device(requested: str) -> str:
    """Pick the best available device."""
    if requested == "auto":
        if is_cuda_available():
            log.info("CUDA available — using GPU")
            return "cuda"
        log.info("CUDA not available — falling back to CPU")
        return "cpu"

    if requested == "cuda" and not is_cuda_available():
        log.warning("CUDA requested but not available — falling back to CPU")
        return "cpu"

    return requested


def gpu_info() -> None:
    """Print GPU info if available."""
    try:
        import cupy as cp
        device = cp.cuda.Device(0)
        mem_free, mem_total = device.mem_info
        log.info(
            "GPU: %s — %.1f GB free / %.1f GB total",
            device.attributes.get("DeviceName", "Unknown"),
            mem_free / 1e9,
            mem_total / 1e9,
        )
    except Exception:
        pass


def to_device(arr: np.ndarray, device: str):
    """Move numpy array to device."""
    if device == "cuda":
        import cupy as cp
        return cp.asarray(arr)
    return arr


def to_numpy(arr) -> np.ndarray:
    """Move any array back to CPU numpy."""
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    except ImportError:
        pass
    return np.asarray(arr)


# ------------------------------------------------------------------
# Training utilities
# ------------------------------------------------------------------

def clip_grad_norm_(params, max_norm: float) -> float:
    """Clip gradient norm across all parameters. Returns total norm."""
    total_norm_sq = 0.0
    grads = []
    for p in params:
        if p.grad is not None:
            xp = p.xp
            total_norm_sq += float(to_numpy(xp.sum(p.grad ** 2)))
            grads.append(p)
    total_norm = math.sqrt(total_norm_sq)

    if total_norm > max_norm and total_norm > 0:
        scale = max_norm / total_norm
        for p in grads:
            p.grad = p.grad * scale

    return total_norm


def cosine_lr(step: int, warmup_steps: int, total_steps: int, lr_max: float, lr_min: float = 0.0) -> float:
    """Cosine schedule with linear warmup."""
    if step < warmup_steps:
        return lr_max * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


# ------------------------------------------------------------------
# Data pipeline
# ------------------------------------------------------------------

def tokenize_corpus(
    tokenizer: BPETokenizer,
    corpus_path: str,
    seq_len: int,
) -> np.ndarray:
    """Tokenize corpus and split into fixed-length sequences."""
    log.info("Tokenizing corpus...")

    all_ids: list[int] = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            ids = tokenizer.encode(line, add_special_tokens=True)
            all_ids.extend(ids)

            if (i + 1) % 500 == 0:
                log.info("  tokenized %d lines (%d tokens so far)", i + 1, len(all_ids))

    log.info("Total tokens: %d", len(all_ids))

    # Split into sequences of seq_len+1 (input + target shifted by 1).
    arr = np.array(all_ids, dtype=np.int64)
    num_seqs = len(arr) // (seq_len + 1)
    arr = arr[: num_seqs * (seq_len + 1)]
    sequences = arr.reshape(num_seqs, seq_len + 1)

    log.info("Created %d sequences of length %d", num_seqs, seq_len + 1)
    return sequences


def create_batches(
    data: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> list[np.ndarray]:
    """Split data into batches."""
    if shuffle:
        indices = np.random.permutation(len(data))
        data = data[indices]

    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i: i + batch_size]
        if len(batch) == batch_size:
            batches.append(batch)
    return batches


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train(
    model: TokenTransformerLM,
    data: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    fp16: bool = False,
    save_every: int = 5,
    max_grad_norm: float = 1.0,
    warmup_fraction: float = 0.05,
) -> None:
    """Main training loop with gradient clipping, cosine LR, optional CUDA."""
    loss_fn = CrossEntropyLoss()
    params = list(model.parameters())
    optimizer = Adam(params, lr=lr)

    total_batches = len(data) // batch_size
    total_steps = total_batches * epochs
    warmup_steps = int(total_steps * warmup_fraction)

    log.info(
        "Training: %d sequences, %d batches/epoch, %d epochs, device=%s, fp16=%s",
        len(data), total_batches, epochs, device, fp16,
    )
    log.info(
        "Schedule: warmup=%d steps, total=%d steps, grad_clip=%.1f",
        warmup_steps, total_steps, max_grad_norm,
    )

    best_loss = float("inf")
    global_step = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        batches = create_batches(data, batch_size, shuffle=True)
        epoch_loss = 0.0
        num_batches = 0

        model.train()

        for step, batch in enumerate(batches, 1):
            global_step += 1

            # Cosine LR with warmup.
            current_lr = cosine_lr(global_step, warmup_steps, total_steps, lr)
            optimizer.lr = current_lr

            # Move batch to device.
            if device == "cuda":
                batch = to_device(batch, device)

            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]

            model.zero_grad()

            logits, _ = model(input_ids, use_causal_mask=True)

            B, T, V = logits.shape
            flat_logits = logits.reshape(B * T, V)
            flat_targets = target_ids.reshape(B * T)

            loss = loss_fn(flat_logits, flat_targets)
            loss.backward()

            # Gradient clipping.
            grad_norm = clip_grad_norm_(params, max_grad_norm)

            optimizer.step()

            loss_val = float(to_numpy(loss.detach().data))
            epoch_loss += loss_val
            num_batches += 1

            if step % 50 == 0:
                avg = epoch_loss / num_batches
                elapsed = time.time() - epoch_start
                speed = step / elapsed
                log.info(
                    "  epoch %d step %d/%d — loss=%.4f avg=%.4f lr=%.2e gnorm=%.2f (%.1f b/s)",
                    epoch, step, len(batches), loss_val, avg, current_lr, grad_norm, speed,
                )

        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - epoch_start

        log.info(
            "Epoch %d/%d — avg_loss=%.4f (%.1fs, %.1f batches/s)",
            epoch, epochs, avg_loss, elapsed,
            num_batches / elapsed if elapsed > 0 else 0,
        )

        # Save checkpoint.
        if epoch % save_every == 0 or avg_loss < best_loss:
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(model, "best")
            save_checkpoint(model, f"epoch_{epoch}")

        # Free GPU memory between epochs.
        if device == "cuda":
            gc.collect()
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    save_checkpoint(model, "final")
    log.info("Training complete! Best loss: %.4f", best_loss)


# ------------------------------------------------------------------
# Checkpoints
# ------------------------------------------------------------------

def save_checkpoint(model: TokenTransformerLM, name: str) -> None:
    """Save model checkpoint (always to CPU numpy)."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"nk_lm_{name}.npz"

    state = {}
    for param_name, param in model.named_parameters():
        state[param_name] = to_numpy(param.detach().data)

    np.savez(str(path), **state)
    log.info("Saved checkpoint: %s", path)


def load_checkpoint(model: TokenTransformerLM, name: str) -> None:
    """Load model checkpoint."""
    path = CHECKPOINT_DIR / f"nk_lm_{name}.npz"
    if not path.exists():
        log.warning("Checkpoint not found: %s", path)
        return

    data = np.load(str(path))
    for param_name, param in model.named_parameters():
        if param_name in data:
            loaded = data[param_name]
            if param.device == "cuda":
                import cupy as cp
                param.data = cp.asarray(loaded)
            else:
                param.data = loaded

    log.info("Loaded checkpoint: %s", path)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

# Presets for different GPU sizes.
PRESETS = {
    "small": {  # ~1.85M params — any GPU / CPU
        "vocab_size": 4096, "d_model": 128, "num_heads": 4,
        "num_layers": 4, "d_ff": 512, "seq_len": 128, "batch_size": 32,
    },
    "medium": {  # ~10M params — RTX 4060 8GB
        "vocab_size": 8192, "d_model": 256, "num_heads": 8,
        "num_layers": 6, "d_ff": 1024, "seq_len": 256, "batch_size": 32,
    },
    "large": {  # ~25M params — RTX 4060 8GB with float16
        "vocab_size": 8192, "d_model": 384, "num_heads": 8,
        "num_layers": 8, "d_ff": 1536, "seq_len": 256, "batch_size": 16,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Neural Kernel LM")
    parser.add_argument("--corpus", default="data/corpus/wiki_en.txt")
    parser.add_argument("--preset", default=None, choices=["small", "medium", "large"],
                        help="Model preset (overrides individual size args)")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--fp16", action="store_true", help="Use float16 (halves memory)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping")
    args = parser.parse_args()

    # Apply preset if given.
    if args.preset:
        preset = PRESETS[args.preset]
        log.info("Using preset '%s': %s", args.preset, preset)
        for k, v in preset.items():
            setattr(args, k.replace("-", "_"), v)

    # Check corpus exists.
    if not Path(args.corpus).exists():
        log.error("Corpus not found: %s", args.corpus)
        log.error("Run first: python scripts/prepare_data.py")
        sys.exit(1)

    # Resolve device.
    device = resolve_device(args.device)

    if device == "cuda":
        gpu_info()

    # Phase 1: Train tokenizer.
    if TOKENIZER_PATH.exists():
        log.info("Loading existing tokenizer from %s", TOKENIZER_PATH)
        tokenizer = BPETokenizer.load(str(TOKENIZER_PATH))
    else:
        log.info("Training BPE tokenizer (vocab_size=%d)...", args.vocab_size)
        tokenizer = BPETokenizer()
        tokenizer.train(args.corpus, vocab_size=args.vocab_size)
        TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(TOKENIZER_PATH))

    actual_vocab = tokenizer.info().vocab_size
    log.info("Tokenizer ready — vocab_size=%d", actual_vocab)

    # Phase 2: Tokenize corpus (with .npy cache).
    cache_path = CHECKPOINT_DIR / f"tokens_seq{args.seq_len}.npy"
    if cache_path.exists():
        log.info("Loading cached tokenized data from %s", cache_path)
        sequences = np.load(str(cache_path))
        log.info("Loaded %d sequences of length %d", len(sequences), sequences.shape[1])
    else:
        sequences = tokenize_corpus(tokenizer, args.corpus, args.seq_len)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), sequences)
        log.info("Saved tokenized cache to %s", cache_path)

    if len(sequences) < args.batch_size:
        log.error(
            "Not enough data: %d sequences < batch_size=%d. "
            "Get more data: python scripts/prepare_data.py --max-articles 5000",
            len(sequences), args.batch_size,
        )
        sys.exit(1)

    # Phase 3: Create model.
    dtype = "float16" if args.fp16 else "float32"
    log.info("Creating model (dtype=%s, device=%s)...", dtype, device)

    model = TokenTransformerLM(
        vocab_size=actual_vocab,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        dropout_p=0.1,
        max_len=args.seq_len + 64,
        activation="gelu",
    )

    # Move to GPU.
    if device == "cuda":
        log.info("Moving model to CUDA...")
        model.cuda()

    # Count parameters.
    total_params = sum(to_numpy(p.detach().data).size for p in model.parameters())
    log.info(
        "Model: d=%d, heads=%d, layers=%d, ff=%d — %.2fM parameters",
        args.d_model, args.num_heads, args.num_layers, args.d_ff,
        total_params / 1e6,
    )

    # Phase 4: Train.
    train(
        model=model,
        data=sequences,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        fp16=args.fp16,
        save_every=args.save_every,
        max_grad_norm=args.max_grad_norm,
    )

    # Phase 5: Test generation.
    log.info("Test generation...")
    model.eval()

    # Move to CPU for generation (numpy-based sampling).
    if device == "cuda":
        model.cpu()

    test_text = "The"
    test_ids = tokenizer.encode(test_text, add_special_tokens=True)
    prompt = np.array([test_ids], dtype=np.int64)

    generated = model.generate(prompt, max_new_tokens=30, temperature=0.8, do_sample=True)
    output = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    log.info("Generated: %s", output)


if __name__ == "__main__":
    main()
