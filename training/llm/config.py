"""
Centralized configuration for the LLM training pipeline.

Хранит все пути и гиперпараметры в одном месте — чтобы менять
настройки в одном файле, а не охотиться по всем скриптам.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ----------------------------------------------------------------------
# Project paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
TOKENIZER_DIR = PROJECT_ROOT / "data" / "tokenizer"
PACKED_DIR = PROJECT_ROOT / "data" / "packed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "lm"
LOG_DIR = PROJECT_ROOT / "checkpoints" / "lm" / "logs"

TOKENIZER_PATH = TOKENIZER_DIR / "tokenizer.json"
TRAIN_BIN = PACKED_DIR / "train.bin"
VAL_BIN = PACKED_DIR / "val.bin"
META_PATH = PACKED_DIR / "meta.json"


# ----------------------------------------------------------------------
# Tokenizer config
# ----------------------------------------------------------------------
@dataclass
class TokenizerConfig:
    vocab_size: int = 8192
    min_frequency: int = 2
    # Если None — читаем все строки. Поставить число для быстрых экспериментов.
    max_lines: int | None = None


# ----------------------------------------------------------------------
# Dataset config
# ----------------------------------------------------------------------
@dataclass
class DatasetConfig:
    # Доля корпуса, отдаваемая в валидационную выборку.
    val_fraction: float = 0.05
    # Какие файлы из data/corpus подхватывать. Можно расширения и glob.
    corpus_glob: str = "*.txt"


# ----------------------------------------------------------------------
# Model config
# ----------------------------------------------------------------------
@dataclass
class ModelConfig:
    # vocab_size заполняется автоматически из meta.json
    vocab_size: int = 8192
    d_model: int = 256
    num_heads: int = 8
    d_ff: int = 1024
    num_layers: int = 4
    dropout_p: float = 0.1
    max_len: int = 512  # context window
    activation: str = "gelu"
    tie_embeddings: bool = True


# ----------------------------------------------------------------------
# Training config
# ----------------------------------------------------------------------
@dataclass
class TrainConfig:
    # Hardware
    device: str = "cuda"  # "cuda" | "cpu"
    seed: int = 1337

    # Optimization
    batch_size: int = 32
    block_size: int = 256          # длина обучающего окна (<= ModelConfig.max_len)
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.0      # в Adam пока не реализован, оставляем для будущего
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)

    # Schedule
    max_steps: int = 5000
    warmup_steps: int = 200
    log_every: int = 25
    eval_every: int = 250
    eval_iters: int = 50
    save_every: int = 500          # сохранять чекпоинт каждые N шагов
    keep_last_n_checkpoints: int = 3

    # Generation samples during training
    sample_every: int = 500
    sample_prompt: str = "Once upon a time"
    sample_max_new_tokens: int = 60
    sample_temperature: float = 0.9
    sample_top_k: int = 50


# ----------------------------------------------------------------------
# Defaults — единая точка истины
# ----------------------------------------------------------------------
@dataclass
class FullConfig:
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


CONFIG = FullConfig()
