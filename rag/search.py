"""
search.py
=========

CLI для семантического поиска по проиндексированному коду.

Usage:
    python rag/search.py "как работает causal mask?"
    python rag/search.py "BPE tokenizer training" --top 10
    python rag/search.py "Adam optimizer" --kind function

Что делает:
  1. Берёт твой запрос.
  2. Получает его embedding через Ollama (nomic-embed-text).
  3. Ищет top-K ближайших чанков в Qdrant.
  4. Печатает результаты с подсветкой пути и кода.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
COLLECTION = "neural_kernel_code"


# ----- ANSI colors -----
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def http_request(method: str, url: str, payload: dict | None = None, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def embed_query(text: str) -> list[float]:
    resp = http_request(
        "POST",
        f"{OLLAMA_URL}/api/embeddings",
        {"model": EMBED_MODEL, "prompt": text},
        timeout=30.0,
    )
    return resp["embedding"]


def search(query_vec: list[float], top: int, kind: str | None) -> list[dict]:
    payload = {
        "vector": query_vec,
        "limit": top,
        "with_payload": True,
        "with_vector": False,
    }
    if kind:
        payload["filter"] = {
            "must": [{"key": "kind", "match": {"value": kind}}]
        }

    resp = http_request(
        "POST",
        f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
        payload,
        timeout=15.0,
    )
    return resp.get("result", [])


def render_result(rank: int, hit: dict, max_lines: int) -> None:
    payload = hit.get("payload", {})
    score = hit.get("score", 0.0)

    file = payload.get("file", "?")
    start = payload.get("start_line", 0)
    end = payload.get("end_line", 0)
    kind = payload.get("kind", "?")
    name = payload.get("name", "?")
    text = payload.get("text", "")

    # Заголовок результата.
    print()
    print(f"{BOLD}{CYAN}[{rank}]{RESET}  "
          f"{BOLD}{file}{RESET}"
          f"{DIM}:{start}-{end}{RESET}  "
          f"{YELLOW}{kind} {name}{RESET}  "
          f"{DIM}(score {score:.3f}){RESET}")

    # Подрезаем длинные сниппеты.
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    else:
        truncated = False

    for line in lines:
        print(f"  {DIM}│{RESET} {line}")
    if truncated:
        print(f"  {DIM}│ … ({len(text.splitlines()) - max_lines} more lines){RESET}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Semantic code search.")
    parser.add_argument("query", nargs="+", help="Search query (in any language).")
    parser.add_argument("--top", type=int, default=5, help="Top-K results (default 5).")
    parser.add_argument(
        "--kind",
        choices=["function", "class", "module", "paragraph"],
        default=None,
        help="Фильтр по типу чанка.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=20,
        help="Сколько строк показывать на результат.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Вывести как JSON (для скриптов).",
    )
    args = parser.parse_args()

    query = " ".join(args.query)

    # Sanity: используем API-эндпоинты, возвращающие JSON.
    try:
        http_request("GET", f"{QDRANT_URL}/collections")
    except Exception as e:
        print(f"[error] Qdrant недоступен на {QDRANT_URL}: {e}", file=sys.stderr)
        print("        Подними его: cd rag && docker compose up -d", file=sys.stderr)
        return 1

    try:
        http_request("GET", f"{OLLAMA_URL}/api/tags")
    except Exception as e:
        print(f"[error] Ollama недоступна на {OLLAMA_URL}: {e}", file=sys.stderr)
        print("        Проверь что Ollama запущена (иконка ламы в трее).", file=sys.stderr)
        return 1

    # Embed + search.
    if not args.json:
        print(f"{DIM}query:{RESET} {query!r}")
        print(f"{DIM}embedding…{RESET}")

    try:
        qvec = embed_query(query)
    except Exception as e:
        print(f"[error] embedding failed: {e}", file=sys.stderr)
        return 1

    try:
        hits = search(qvec, top=args.top, kind=args.kind)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"[error] Qdrant: {e.code} — {body[:300]}", file=sys.stderr)
        if "doesn't exist" in body or e.code == 404:
            print("        Сначала запусти индексацию: python rag/index_codebase.py",
                  file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return 0

    if not hits:
        print(f"\n{YELLOW}[no results]{RESET}")
        return 0

    print(f"{GREEN}[found {len(hits)} results]{RESET}")
    for i, hit in enumerate(hits, start=1):
        render_result(i, hit, args.max_lines)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
