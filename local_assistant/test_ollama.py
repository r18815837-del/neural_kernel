"""
test_ollama.py
==============

Sanity-check твоего локального Ollama + модели.

Проверяет:
  1. Ollama-сервер живой (http://localhost:11434 отвечает).
  2. Все три модели скачаны (qwen2.5-coder:7b, qwen2.5-coder:1.5b-base, nomic-embed-text).
  3. Chat-модель отвечает на тестовый запрос про код.
  4. Embedding-модель возвращает вектор.

Запуск:
    python local_assistant/test_ollama.py

Должен вывести "[OK] All checks passed" и короткий ответ Qwen про Python.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request


OLLAMA_URL = "http://localhost:11434"
REQUIRED_MODELS = [
    "qwen2.5-coder:7b",
    "qwen2.5-coder:1.5b-base",
    "nomic-embed-text",
]


# ----------------------------------------------------------------------
# Pretty printing
# ----------------------------------------------------------------------
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}[OK]{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}[FAIL]{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}[warn]{RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {DIM}{msg}{RESET}")


# ----------------------------------------------------------------------
# HTTP helpers
# ----------------------------------------------------------------------
def http_get(url: str, timeout: float = 5.0) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.URLError as e:
        raise ConnectionError(f"Failed to reach {url}: {e}")


def http_post_json(url: str, payload: dict, timeout: float = 60.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


# ----------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------
def check_server() -> bool:
    print("\n[1/4] Проверяем что Ollama-сервер запущен…")
    try:
        status, body = http_get(f"{OLLAMA_URL}/", timeout=3.0)
    except ConnectionError as e:
        fail(f"Ollama недоступна на {OLLAMA_URL}")
        info(str(e))
        info("Что сделать:")
        info("  1. Проверь что в трее есть иконка ламы.")
        info("  2. Если нет — запусти Ollama из меню Пуск.")
        info("  3. Или вручную в PowerShell:")
        info("     & \"$env:LOCALAPPDATA\\Programs\\Ollama\\ollama.exe\" serve")
        return False

    text = body.decode("utf-8", errors="replace").strip()
    if "ollama is running" in text.lower():
        ok(f"сервер отвечает на {OLLAMA_URL}")
        return True

    warn(f"сервер отвечает, но тело неожиданное: {text[:80]!r}")
    return True  # всё равно достаточно, что отвечает


def check_models() -> tuple[bool, list[str]]:
    print("\n[2/4] Проверяем что модели скачаны…")
    try:
        status, body = http_get(f"{OLLAMA_URL}/api/tags", timeout=10.0)
    except ConnectionError as e:
        fail(f"не смог получить список моделей: {e}")
        return False, []

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        fail("/api/tags вернул не JSON")
        return False, []

    installed = [m["name"] for m in data.get("models", [])]
    info(f"установлено моделей: {len(installed)}")

    all_ok = True
    missing: list[str] = []
    for required in REQUIRED_MODELS:
        # Ollama хранит имена в формате "name:tag". Если в required уже есть тег
        # (например "qwen2.5-coder:7b") — сравниваем точно. Если тега нет
        # (например "nomic-embed-text") — Ollama по умолчанию добавит ":latest",
        # так что ищем любой тег.
        if ":" in required:
            match = any(m == required for m in installed)
        else:
            match = any(
                m == required or m.startswith(required + ":") for m in installed
            )

        if match:
            ok(f"{required}")
        else:
            fail(f"{required} — НЕ УСТАНОВЛЕНА")
            missing.append(required)
            all_ok = False

    if missing:
        print()
        info("Чтобы скачать недостающие:")
        for m in missing:
            info(f"  ollama pull {m}")

    return all_ok, missing


def check_chat_model() -> bool:
    print("\n[3/4] Посылаем тестовый запрос chat-модели (qwen2.5-coder:7b)…")
    info("это может занять 10-30 секунд при первом запуске (модель грузится в VRAM)")

    payload = {
        "model": "qwen2.5-coder:7b",
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise programming assistant. Reply in one sentence.",
            },
            {
                "role": "user",
                "content": "In Python, how do I reverse a list in-place?",
            },
        ],
        "options": {
            "temperature": 0.1,
            "num_ctx": 2048,
        },
    }

    t0 = time.time()
    try:
        resp = http_post_json(
            f"{OLLAMA_URL}/api/chat", payload, timeout=120.0
        )
    except Exception as e:
        fail(f"запрос упал: {e}")
        return False

    elapsed = time.time() - t0
    content = resp.get("message", {}).get("content", "").strip()

    if not content:
        fail(f"пустой ответ: {resp!r}")
        return False

    ok(f"ответ получен за {elapsed:.1f}s")
    print()
    print(f"    Q: In Python, how do I reverse a list in-place?")
    print(f"    A: {content}")
    print()

    # sanity: в ответе должно встретиться что-то релевантное
    lower = content.lower()
    if any(hint in lower for hint in ("reverse", ".reverse", "[::-1]", "reversed")):
        ok("ответ выглядит осмысленным")
        return True
    else:
        warn("ответ пришёл, но выглядит неожиданно. Возможно, модель в странном состоянии.")
        return True


def check_embedding_model() -> bool:
    print("\n[4/4] Проверяем embedding-модель (nomic-embed-text)…")

    payload = {
        "model": "nomic-embed-text",
        "prompt": "def hello_world(): return 42",
    }

    try:
        resp = http_post_json(
            f"{OLLAMA_URL}/api/embeddings", payload, timeout=30.0
        )
    except Exception as e:
        fail(f"embedding запрос упал: {e}")
        return False

    embedding = resp.get("embedding", [])
    if not embedding or not isinstance(embedding, list):
        fail(f"ответ без embedding: {resp!r}")
        return False

    ok(f"получен вектор размерности {len(embedding)}")
    return True


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    print("=" * 60)
    print(" Ollama sanity check")
    print("=" * 60)

    all_ok = True

    server_ok = check_server()
    all_ok = all_ok and server_ok

    if not server_ok:
        # Без сервера остальные проверки бессмысленны.
        print()
        print(f"{RED}[FAIL]{RESET} Ollama не запущена. Остальные проверки пропущены.")
        return 1

    models_ok, missing = check_models()
    all_ok = all_ok and models_ok

    if missing and "qwen2.5-coder:7b" in missing:
        # Без chat модели следующую проверку пропускаем.
        warn("chat-модель не скачана — пропускаем проверку 3")
    else:
        chat_ok = check_chat_model()
        all_ok = all_ok and chat_ok

    if missing and "nomic-embed-text" in missing:
        warn("embedding-модель не скачана — пропускаем проверку 4")
    else:
        embed_ok = check_embedding_model()
        all_ok = all_ok and embed_ok

    print()
    print("=" * 60)
    if all_ok:
        print(f"{GREEN}[OK]{RESET} All checks passed. Можно идти к шагу 4 из SETUP.md.")
        return 0
    else:
        print(f"{RED}[FAIL]{RESET} Есть проблемы. Исправь и запусти снова.")
        print(f"     Подробности — в SETUP.md секция Troubleshooting.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
