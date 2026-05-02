# Fine-tuning pipeline для neural_kernel

Цель: получить **твою личную модель**, обученную на твоём коде и стиле.

## Архитектура

```
твой git history + код
        │
        ▼  collect_dataset.py     (Windows, прямо сейчас)
data/finetune/dataset.jsonl
        │
        ▼  WSL2 + Unsloth         (нужен Linux)
        ▼  train_qlora.py
checkpoints/lora/qwen-3b-coder-nk/
        │
        ▼  deploy_to_ollama.sh    (Linux + Windows Ollama)
ollama run nk-coder              ← твоя личная модель!
        │
        ▼  Continue.dev
твой VS Code знает твой стиль
```

## Этапы

### Этап 1 — Dataset (можно прямо сейчас на Windows)

```powershell
python finetuning\collect_dataset.py
```

Создаст `data/finetune/dataset.jsonl` с парами:
- (commit message, diff) — «реализуй X» → код
- (docstring, function body) — описание → реализация
- (test, implementation) — тест → реализация

После — посмотри на 10-20 случайных примеров глазами:

```powershell
Get-Content data\finetune\dataset.jsonl | Get-Random -Count 10
```

Если в датасете мусор — это **главная** проблема для fine-tuning, а не «модель плохая». Лучше 500 чистых примеров чем 5000 грязных.

### Этап 2 — Установить WSL2 (одноразово, ~30 минут)

См. [WSL2_SETUP.md](WSL2_SETUP.md).

После — внутри WSL2 у тебя будет работающий Python + CUDA + Unsloth.

### Этап 3 — Обучить (внутри WSL2)

```bash
# уже в Ubuntu/WSL2
cd /mnt/c/Users/r1881/Downloads/neural_kernel
source .venv-wsl/bin/activate
python finetuning/train_qlora.py
```

Время: **4–12 часов** на 8 ГБ VRAM, зависит от размера датасета.

### Этап 4 — Задеплоить в Ollama

```bash
bash finetuning/deploy_to_ollama.sh
```

Скрипт:
1. Мерджит LoRA в Qwen2.5-Coder 3B base.
2. Конвертирует в GGUF Q4_K_M.
3. Создаёт Modelfile.
4. Регистрирует в Windows Ollama как `nk-coder:latest`.

После этого в Continue меняешь модель на `nk-coder:latest` — всё.

## Важно про датасет

Размер ≠ качество. Реалистичные ожидания:

| размер датасета | результат |
|-----------------|-----------|
| <100 примеров   | Overfit, модель попугайничает. Не делай. |
| 200-500         | Модель учит твой стиль, но не обобщает. Ок для узких задач. |
| 500-2000        | Хороший баланс. Большинство кейсов попадают. |
| 2000+           | Заметное улучшение, но качество данных важнее. |

`neural_kernel` ~50 файлов кода + ~50 коммитов даст порядка 200-400 пар. Этого достаточно для прототипа. Если хочешь больше — добавляй другие свои проекты в `collect_dataset.py`.

## Что после деплоя

В Continue.dev обновляешь конфиг:

```json
{
  "models": [
    {
      "title": "NK-Coder (my fine-tune)",
      "provider": "ollama",
      "model": "nk-coder:latest",
      "apiBase": "http://localhost:11434"
    }
  ]
}
```

Дальше — пользуешься как обычно, но модель отвечает в твоём стиле.
