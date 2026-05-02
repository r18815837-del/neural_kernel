# WSL2 + CUDA setup для fine-tuning

Unsloth (нашa тренировочная библиотека) **работает только на Linux**. На чистой Windows — нет: triton / xformers / bitsandbytes конфликтуют. Решение — WSL2 (Windows Subsystem for Linux), это полноценный Linux внутри Windows с доступом к твоей видеокарте.

После настройки у тебя будет:

- Ubuntu 22.04 параллельно с Windows.
- Python + CUDA + Unsloth работают.
- Доступ к C:\ диску из Linux через `/mnt/c/`.
- Файлы открыты обеим системам — пишешь в VS Code на Windows, обучаешь в WSL.

Время: **30-60 минут одноразово**.

---

## Шаг 1 — Установить WSL2 (5 минут)

В **PowerShell, запущенном от администратора**:

```powershell
wsl --install -d Ubuntu-22.04
```

Это:
- Включит компонент Windows Subsystem for Linux.
- Скачает и установит Ubuntu 22.04.

После — **перезагрузи компьютер** (важно!).

После перезагрузки автоматически откроется окно терминала Ubuntu и попросит:

- **Username** — что хочешь, например `raian`.
- **Password** — придумай (не путать с Windows-паролем). Запиши.

После этого ты внутри Ubuntu. Проверь:

```bash
uname -a    # должно показать что-то с "Linux" и "WSL2"
nvidia-smi  # ВАЖНО: должно показать твою видеокарту
```

Если `nvidia-smi` говорит «command not found» — переходи к шагу 2.

Если `nvidia-smi` показывает информацию о GPU — отлично, **переходи к шагу 3** (NVIDIA уже подцепилась).

---

## Шаг 2 — Установить NVIDIA CUDA Toolkit для WSL (если нужно)

Если у тебя **свежий драйвер NVIDIA на Windows** (с поддержкой WSL — это драйверы 510+, у тебя почти наверняка такой), то `nvidia-smi` уже работает в WSL без дополнительной установки.

Если нет — обнови драйвер NVIDIA на Windows:

1. Скачай [последний драйвер NVIDIA для своей карты](https://www.nvidia.com/Download/index.aspx).
2. Установи, перезагрузи Windows.
3. В WSL: `nvidia-smi` — должно работать.

Дополнительно поставь CUDA Toolkit внутри WSL (нужно для Unsloth):

```bash
# Внутри Ubuntu/WSL2:
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4
```

После — добавь в `~/.bashrc`:

```bash
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Проверь:

```bash
nvcc --version    # должно показать 12.4
```

---

## Шаг 3 — Установить Python + venv (5 минут)

Ubuntu 22.04 идёт с Python 3.10. Поставь pip и venv:

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git curl build-essential
```

Перейди в свой проект (через `/mnt/c/`):

```bash
cd /mnt/c/Users/r1881/Downloads/neural_kernel
```

Создай **отдельный venv для WSL** (нельзя использовать `.venv` от Windows — там бинарники под Windows):

```bash
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
pip install --upgrade pip
```

После активации в начале строки появится `(.venv-wsl)`.

---

## Шаг 4 — Установить Unsloth (10-15 минут)

Это самый «жирный» шаг. Unsloth + PyTorch + xformers скачается ~5-7 ГБ.

```bash
# В активированном venv:

# 1. PyTorch с CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. Unsloth + зависимости (одной командой)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 3. Дополнительно для нашего пайплайна
pip install datasets accelerate bitsandbytes peft trl
```

Проверь, что Unsloth видит твой GPU:

```bash
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
from unsloth import FastLanguageModel
print('Unsloth import OK')
"
```

Должно показать твою видеокарту, ~8 ГБ VRAM, и `Unsloth import OK`.

Если падает с ошибкой про `bitsandbytes` или `triton` — попробуй переустановить:

```bash
pip uninstall -y bitsandbytes triton
pip install bitsandbytes triton
```

---

## Шаг 5 — Готово. Что дальше

Теперь у тебя WSL2 + Ubuntu + Python + Unsloth. Когда захочешь обучать:

```bash
# 1. Открой Ubuntu (из меню Пуск → Ubuntu, или wsl в PowerShell)
# 2. Перейди в проект
cd /mnt/c/Users/r1881/Downloads/neural_kernel
# 3. Активируй venv
source .venv-wsl/bin/activate
# 4. Запусти обучение (это будет следующим шагом)
python finetuning/train_qlora.py
```

Файлы между Windows и Linux **общие** — можешь редактировать в VS Code на Windows, запускать в WSL.

---

## Полезные команды

```bash
# Сколько занято VRAM прямо сейчас
nvidia-smi

# Перезапустить WSL целиком (если что-то застряло)
# (это в Windows PowerShell, не в WSL!)
wsl --shutdown

# Размер занятого WSL диска
du -sh ~/

# Открыть текущую Linux-папку в Проводнике Windows
explorer.exe .
```

## Troubleshooting

### `wsl --install` говорит «команда не найдена»

У тебя Windows старый. Нужен Windows 10 build 19041+ или Windows 11. Проверь:

```powershell
winver
```

Если build < 19041 — обнови Windows через Settings → Update.

### `nvidia-smi` в WSL возвращает «No devices found»

- Проверь, что в Windows установлен NVIDIA-драйвер версии 510+. В Windows: правый клик на десктопе → NVIDIA Control Panel → System Information.
- Перезапусти WSL: в PowerShell `wsl --shutdown`, потом снова открой Ubuntu.

### Unsloth падает на импорте

Самая частая причина — несовместимые версии torch/CUDA. Снеси venv и поставь заново:

```bash
deactivate
rm -rf .venv-wsl
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
# и заново шаг 4
```

### Очень медленный доступ к /mnt/c/

Это известная особенность WSL2 — файловая система Windows через `/mnt/c/` медленная. Если будет тормозить чтение датасета, можно скопировать данные внутрь WSL:

```bash
mkdir -p ~/nk-data
cp data/finetune/dataset.jsonl ~/nk-data/
```

И тренироваться оттуда. Веса всё равно сохраняй обратно в `/mnt/c/...` чтобы Windows их видела.
