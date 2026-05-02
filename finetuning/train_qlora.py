"""
train_qlora.py
==============

QLoRA fine-tuning Qwen2.5-Coder 3B на твоём датасете через Unsloth.

Запускается ВНУТРИ WSL2 (см. WSL2_SETUP.md). На чистой Windows не работает.

Настройки подобраны под 8 ГБ VRAM:
  - модель: Qwen2.5-Coder 3B base (не Instruct — дотренируем сами)
  - 4-bit квантизация базовой модели (QLoRA)
  - LoRA rank 16, alpha 16
  - seq_len 2048 (можно опустить до 1024 если OOM)
  - batch=2, grad_accum=8 (эффективный батч 16)
  - 3 эпохи

Время: 4-12 часов в зависимости от размера датасета.

Usage:
    # ВНУТРИ WSL2:
    cd /mnt/c/Users/r1881/Downloads/neural_kernel
    source .venv-wsl/bin/activate
    python finetuning/train_qlora.py

    # Опционально — другая модель / меньше шагов для теста:
    python finetuning/train_qlora.py --max-steps 100 --base-model unsloth/Qwen2.5-Coder-1.5B-bnb-4bit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Заглушаем шум от tokenizers / datasets.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")


_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = _ROOT / "data" / "finetune" / "dataset.jsonl"
DEFAULT_OUTPUT = _ROOT / "checkpoints" / "lora" / "qwen-3b-coder-nk"


def main() -> int:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Qwen 2.5-Coder via Unsloth.")
    parser.add_argument(
        "--base-model",
        default="unsloth/Qwen2.5-Coder-3B-bnb-4bit",
        help="Pre-quantized base. Для 8 ГБ — 3B; для теста — 1.5B; если есть >12 ГБ — можно 7B.",
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))

    # LoRA params
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Training params
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1, help="-1 = до конца эпох")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    # Misc
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=10)

    args = parser.parse_args()

    # ----- Lazy imports (Unsloth тяжёлый, не грузим если просто --help) -----
    print("[boot] importing unsloth + torch (это может занять 30 сек)…")
    try:
        import torch
        from unsloth import FastLanguageModel
        from datasets import load_dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError as e:
        print(f"[error] {e}")
        print("Похоже, Unsloth не установлен. См. finetuning/WSL2_SETUP.md, шаг 4.")
        return 1

    # ----- Sanity checks -----
    if not torch.cuda.is_available():
        print("[error] CUDA недоступна. Unsloth требует GPU.")
        print("Проверь: nvidia-smi должна показать твою карту.")
        return 1

    gpu = torch.cuda.get_device_properties(0)
    print(f"[gpu] {gpu.name}, VRAM: {gpu.total_memory / 1e9:.1f} GB")

    if not Path(args.dataset).exists():
        print(f"[error] Датасет не найден: {args.dataset}")
        print("Сначала запусти: python finetuning/collect_dataset.py")
        return 1

    # Считаем кол-во примеров и средний размер.
    n_examples = 0
    total_chars = 0
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            n_examples += 1
            try:
                obj = json.loads(line)
                for m in obj["messages"]:
                    total_chars += len(m["content"])
            except Exception:
                pass

    avg_chars = total_chars / max(n_examples, 1)
    print(f"[data] {n_examples} examples, ~{avg_chars:.0f} chars/example avg")

    if n_examples < 50:
        print("[warn] Датасет очень маленький (<50 примеров). Модель переобучится.")
        print("       Рекомендация: собери больше данных, добавь --extra-projects")
        print("       в collect_dataset.py.")

    # ----- Загрузка модели -----
    print(f"\n[model] loading {args.base_model}…")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.seq_len,
        dtype=None,            # автодетект (bfloat16 на новых GPU, fp16 на старых)
        load_in_4bit=True,     # QLoRA — обязательно
    )

    # Применяем LoRA адаптеры.
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        # Какие модули адаптируем — стандартный набор для Qwen.
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    # Печатаем количество тренируемых параметров.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M  "
          f"({100 * trainable / total:.2f}%)")

    # ----- Подготовка датасета -----
    print(f"\n[data] loading {args.dataset}")
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    print(f"[data] loaded {len(dataset)} examples")

    # Формат: используем chat template токенизатора.
    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    print(f"[data] formatted, sample length: {len(dataset[0]['text'])} chars")

    # ----- Тренировка -----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=args.log_every,
        save_steps=args.save_every,
        save_total_limit=3,
        optim="adamw_8bit",          # экономит VRAM
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        seed=args.seed,
        report_to="none",            # не шлём в wandb
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.seq_len,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    print(f"\n[train] starting QLoRA fine-tune…")
    print(f"        effective batch size = {args.batch_size * args.grad_accum}")
    print(f"        seq_len = {args.seq_len}")
    print(f"        epochs = {args.epochs}")
    print()

    # Стартовая статистика по VRAM.
    used_before = torch.cuda.memory_reserved() / 1e9
    print(f"[vram] before training: {used_before:.2f} GB reserved")
    print()

    trainer_stats = trainer.train()

    used_after = torch.cuda.max_memory_reserved() / 1e9
    print()
    print(f"[vram] peak during training: {used_after:.2f} GB reserved")
    print(f"[done] training time: {trainer_stats.metrics['train_runtime'] / 60:.1f} min")
    print(f"       loss: {trainer_stats.metrics.get('train_loss', '?')}")

    # ----- Сохранение LoRA -----
    print(f"\n[save] writing LoRA adapter to {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Размер адаптера.
    total_size = sum(p.stat().st_size for p in output_dir.glob("*") if p.is_file())
    print(f"       adapter size: {total_size / 1e6:.1f} MB")

    print()
    print("=" * 70)
    print("Готово. Дальше:")
    print(f"  bash finetuning/deploy_to_ollama.sh \\")
    print(f"    --lora-dir {output_dir} \\")
    print(f"    --base-model {args.base_model} \\")
    print(f"    --name nk-coder")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
