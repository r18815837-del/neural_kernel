# Training the Neural Kernel LLM

Полный пайплайн обучения языковой модели (decoder-only Transformer)
на собственном `kernel/` фреймворке. Работает на CPU (NumPy) или GPU (CuPy).

## Pipeline в 4 шага

```
data/corpus/*.txt
       │
       ▼   01_download_data.py     (скачать TinyStories, опционально)
data/corpus/tinystories_valid.txt
       │
       ▼   02_train_tokenizer.py   (обучить BPE, vocab=8192)
data/tokenizer/tokenizer.json
       │
       ▼   03_pack_dataset.py      (токенизировать → uint16 .bin, 95/5 split)
data/packed/train.bin, val.bin, meta.json
       │
       ▼   04_train_lm.py          (тренировка с warmup+cosine, eval, sampling)
checkpoints/lm/nk_lm_best.pkl
       │
       ▼   05_generate.py          (генерация по промпту)
```

## Quick start (на CPU, smoke test, ~5 минут)

```bash
cd <project root>

# 1. Скачать корпус (~22 MB)
python training/llm/01_download_data.py

# 2. Тренировать tokenizer (быстро, ~1-2 минуты на 22 MB)
python training/llm/02_train_tokenizer.py --vocab-size 4096

# 3. Упаковать датасет
python training/llm/03_pack_dataset.py

# 4. Тренировать tiny-модель для проверки pipeline
python training/llm/04_train_lm.py \
    --device cpu \
    --max-steps 200 --eval-every 100 \
    --batch-size 8 --block-size 64 \
    --d-model 64 --num-layers 2 --num-heads 4 --d-ff 256
```

## Полная тренировка (GPU, реальная LLM, несколько часов)

```bash
# 1-3 как выше, но vocab побольше
python training/llm/02_train_tokenizer.py --vocab-size 8192
python training/llm/03_pack_dataset.py

# 4. Полная тренировка
python training/llm/04_train_lm.py --device cuda
# По умолчанию: d_model=256, layers=4, heads=8, block=256, batch=32
# 5000 шагов, warmup=200, eval каждые 250 шагов
```

После 5000 шагов на TinyStories модель должна писать осмысленные
короткие истории. Проверить:

```bash
python training/llm/05_generate.py \
    --prompt "Once upon a time, there was a little dragon" \
    --max-new-tokens 200 \
    --temperature 0.8 \
    --top-k 50
```

## Резюме после прерывания

```bash
python training/llm/04_train_lm.py --resume
```

Загрузит последний чекпоинт `nk_lm_step_*.pkl` и продолжит.

## Что под капотом

* **`TokenTransformerLM`** уже есть в `kernel/nn/modules/token_lm.py` —
  это GPT-style модель (encoder с causal mask = decoder).
* **BPE tokenizer** — `kernel/tokenization/bpe_tokenizer.py`,
  обучается с нуля (zero deps).
* **Optimizer** — Adam (`kernel/optim/adam.py`), бета поправлены
  под LM-настройки (0.9, 0.95).
* **LR schedule** — линейный warmup + cosine decay (как в LLaMA / GPT-3).
* **Loss** — CrossEntropy на (B*T, V) flatten.
* **Сэмплирование** — top-k / top-p / temperature через
  встроенный `model.generate()`.
* **Чекпоинты** — pickle с `state_dict`, optimizer state, и meta
  (включая `model_config`, чтобы 05_generate.py пересоздал
  правильную архитектуру).

## Свои данные

Положи `.txt` файлы в `data/corpus/` — пайплайн их подхватит:

```bash
cp my_books/*.txt data/corpus/
python training/llm/02_train_tokenizer.py
python training/llm/03_pack_dataset.py
python training/llm/04_train_lm.py
```

## Гиперпараметры

Все базовые значения — в `training/llm/config.py`. Любой можно
переопределить через CLI флаги (см. `--help` каждого скрипта).

Ориентир для разных размеров (на GPU):

| модель | params | d_model | layers | block | batch | контекст |
|--------|-------:|--------:|-------:|------:|------:|---------:|
| nano   |  1-3M  |  64-128 |  2-4   |   64  |  16   |    64    |
| micro  |  5-15M |   256   |  4-6   |  256  |  32   |   256    |
| small  | 30-60M |   384   |  6-8   |  512  |  32   |   512    |

Чем больше `d_model` × `layers`, тем лучше качество, но тем дольше шаг
и больше памяти.

## Логи

Каждый запуск пишет JSONL-лог в `checkpoints/lm/logs/<run-name>_train.log.jsonl`.
Каждая строка — это `{"event": ..., "step": ..., ...}`.
Удобно скармливать в pandas / matplotlib.

## Что улучшить дальше (research-направления)

1. **RoPE** вместо синусоидальной позиционной кодировки — лучшая экстраполяция.
2. **RMSNorm + SwiGLU** (LLaMA-style блок) — быстрее и точнее, чем LayerNorm + ReLU.
3. **KV-cache** для инференса — ускорит `generate()` в N раз.
4. **GQA/MQA** — экономит память при инференсе длинных контекстов.
5. **AdamW с weight decay** — нужно добавить в `kernel/optim/`.
6. **Mixed precision (FP16/BF16)** — пока всё в FP32, есть запас.

Можно делать инкрементально, не ломая текущий pipeline — модель и
training loop этому не мешают.
