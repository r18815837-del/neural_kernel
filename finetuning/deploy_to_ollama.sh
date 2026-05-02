#!/usr/bin/env bash
# deploy_to_ollama.sh
# ====================
#
# После того как train_qlora.py обучил LoRA-адаптер,
# этот скрипт превращает его в живую модель в Ollama:
#
#   1. Мерджит LoRA в base через PEFT (unsloth.save_pretrained_merged)
#   2. Конвертирует merged HF model → GGUF
#   3. Квантизирует в Q4_K_M (под 8 ГБ VRAM)
#   4. Создаёт Modelfile
#   5. Регистрирует в локальной Ollama как nk-coder:latest
#
# Запускается ВНУТРИ WSL2 после train_qlora.py.
#
# Usage:
#   bash finetuning/deploy_to_ollama.sh \
#     --lora-dir checkpoints/lora/qwen-3b-coder-nk \
#     --base-model unsloth/Qwen2.5-Coder-3B-bnb-4bit \
#     --name nk-coder

set -euo pipefail

# ----- Defaults -----
LORA_DIR="checkpoints/lora/qwen-3b-coder-nk"
BASE_MODEL="unsloth/Qwen2.5-Coder-3B-bnb-4bit"
OUT_NAME="nk-coder"
QUANT="Q4_K_M"

# ----- Parse args -----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lora-dir)   LORA_DIR="$2"; shift 2 ;;
        --base-model) BASE_MODEL="$2"; shift 2 ;;
        --name)       OUT_NAME="$2"; shift 2 ;;
        --quant)      QUANT="$2"; shift 2 ;;
        -h|--help)
            cat <<EOF
Usage: $0 [OPTIONS]
  --lora-dir DIR     Папка с LoRA-адаптером (вывод train_qlora.py)
  --base-model NAME  HF имя базовой модели (как в train_qlora.py)
  --name NAME        Имя модели в Ollama (default: nk-coder)
  --quant Q          GGUF квантизация (default: Q4_K_M)
EOF
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=================================================================="
echo " Deploy LoRA → Ollama"
echo "=================================================================="
echo "  LoRA dir   : $LORA_DIR"
echo "  Base model : $BASE_MODEL"
echo "  Out name   : $OUT_NAME"
echo "  Quant      : $QUANT"
echo

# ----- Sanity check: LoRA dir exists -----
if [[ ! -d "$LORA_DIR" ]]; then
    echo "[error] LoRA directory not found: $LORA_DIR"
    echo "        Сначала запусти: python finetuning/train_qlora.py"
    exit 1
fi

WORK_DIR="$(dirname "$LORA_DIR")/_deploy_${OUT_NAME}"
MERGED_DIR="${WORK_DIR}/merged"
GGUF_DIR="${WORK_DIR}/gguf"
mkdir -p "$MERGED_DIR" "$GGUF_DIR"

# ----- Step 1: Merge LoRA into base -----
echo "[1/5] Merging LoRA into base model…"
python <<PYEOF
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from unsloth import FastLanguageModel

print("  loading base + adapter…")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="$LORA_DIR",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,   # для merge нужны полные веса
)

print("  saving merged model to $MERGED_DIR …")
model.save_pretrained_merged(
    "$MERGED_DIR",
    tokenizer,
    save_method="merged_16bit",   # bf16/fp16 веса
)
print("  done.")
PYEOF

# ----- Step 2: llama.cpp + конвертация в GGUF -----
echo
echo "[2/5] Setting up llama.cpp converter (если ещё нет)…"

LLAMA_CPP_DIR="${HOME}/llama.cpp"
if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
    git clone https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR"
fi

cd "$LLAMA_CPP_DIR"
git pull --quiet || true

# Питон-зависимости конвертера.
python -m pip install --quiet -r requirements.txt 2>/dev/null || \
    python -m pip install --quiet -r requirements/requirements-convert_hf_to_gguf.txt

# Соберём quantize, если ещё нет.
if [[ ! -x "build/bin/llama-quantize" && ! -x "llama-quantize" ]]; then
    echo "  building llama-quantize…"
    cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF >/dev/null
    cmake --build build --target llama-quantize -j --config Release >/dev/null
fi

QUANTIZE_BIN="build/bin/llama-quantize"
[[ -x "$QUANTIZE_BIN" ]] || QUANTIZE_BIN="./llama-quantize"

cd - >/dev/null

# ----- Step 3: HF → GGUF (fp16 промежуточный) -----
echo
echo "[3/5] Converting merged HF model → GGUF (fp16)…"
GGUF_FP16="${GGUF_DIR}/${OUT_NAME}-fp16.gguf"

python "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" \
    "$MERGED_DIR" \
    --outfile "$GGUF_FP16" \
    --outtype f16

echo "  fp16 size: $(du -h "$GGUF_FP16" | cut -f1)"

# ----- Step 4: Quantize to Q4_K_M -----
echo
echo "[4/5] Quantizing GGUF to $QUANT…"
GGUF_QUANT="${GGUF_DIR}/${OUT_NAME}-${QUANT}.gguf"

"${LLAMA_CPP_DIR}/${QUANTIZE_BIN}" \
    "$GGUF_FP16" \
    "$GGUF_QUANT" \
    "$QUANT"

echo "  quant size: $(du -h "$GGUF_QUANT" | cut -f1)"

# Удалим fp16 (больше не нужен, ~6 ГБ).
rm -f "$GGUF_FP16"

# ----- Step 5: Modelfile + ollama create -----
echo
echo "[5/5] Registering in Ollama as '$OUT_NAME'…"

MODELFILE="${WORK_DIR}/Modelfile"
cat > "$MODELFILE" <<EOF
FROM $GGUF_QUANT

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

SYSTEM """You are a programming assistant fine-tuned on the user's personal codebase
(neural_kernel — a from-scratch deep learning framework). You match the user's
coding style, naming conventions, docstring format, and library choices.
Be direct and concise. Show code, explain only when asked."""
EOF

# Ollama в Windows слушает на localhost:11434 — из WSL она доступна по
# адресу хоста. Установим env, чтобы ollama-cli знал куда ходить.
WINDOWS_HOST="$(ip route show | grep -i default | awk '{print $3}' | head -n1)"
export OLLAMA_HOST="${OLLAMA_HOST:-http://${WINDOWS_HOST}:11434}"
echo "  using Ollama at: $OLLAMA_HOST"

# Если внутри WSL стоит свой ollama — используй его, иначе — Windows.
if command -v ollama >/dev/null 2>&1; then
    ollama create "$OUT_NAME" -f "$MODELFILE"
else
    echo "[note] ollama CLI не найден внутри WSL."
    echo "       Скопируй GGUF и Modelfile в Windows и зарегистрируй там:"
    echo
    WINDOWS_PATH=$(wslpath -w "$GGUF_QUANT")
    WINDOWS_MF=$(wslpath -w "$MODELFILE")
    echo "       В PowerShell:"
    echo "         ollama create $OUT_NAME -f \"$WINDOWS_MF\""
    echo
    echo "       (Modelfile уже ссылается на $WINDOWS_PATH)"
    exit 0
fi

echo
echo "=================================================================="
echo " Готово!"
echo "=================================================================="
echo
echo "Проверь, что модель появилась:"
echo "  ollama list  # должна быть строка с '$OUT_NAME'"
echo
echo "Попробуй:"
echo "  ollama run $OUT_NAME 'Write a docstring for this function'"
echo
echo "В Continue.dev обнови ~/.continue/config.json — поменяй model на:"
echo "  \"model\": \"$OUT_NAME:latest\""
echo
