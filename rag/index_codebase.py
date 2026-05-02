"""
index_codebase.py
=================

Индексирует все .py / .md / .txt файлы проекта в локальный Qdrant.

Pipeline:
  1. Читает файлы по glob.
  2. Разбивает на чанки (по функциям для .py, по абзацам для .md/.txt).
  3. Получает embedding для каждого чанка через Ollama (nomic-embed-text).
  4. Загружает в Qdrant коллекцию.

Идемпотентно: повторный запуск обновляет существующие чанки (по hash).

Usage:
    # Сначала подними Qdrant:
    cd rag && docker compose up -d

    # Потом запусти индексацию:
    python rag/index_codebase.py

    # Или с другим путём:
    python rag/index_codebase.py --root C:\\Users\\r1881\\Downloads\\other_project
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768                         # размерность nomic-embed-text
COLLECTION = "neural_kernel_code"

# Сколько чанков отправлять параллельно в Qdrant.
UPSERT_BATCH = 32

# Лимиты на размер чанка.
# MAX_CHUNK_CHARS — максимум, с которым `nomic-embed-text` работает стабильно.
# У модели лимит ~8192 токена ≈ ~30_000 символов, но чтобы не упираться в
# граничные случаи, режем жёстче.
MIN_CHUNK_CHARS = 50
MAX_CHUNK_CHARS = 4000
# Hard-cap при отправке в Ollama (на случай если длинный .md прорвался).
EMBED_HARD_CAP_CHARS = 12000


# ----------------------------------------------------------------------
# HTTP helpers
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Chunk extraction
# ----------------------------------------------------------------------
@dataclass
class Chunk:
    file: str
    start_line: int
    end_line: int
    kind: str          # "function" | "class" | "module" | "paragraph"
    name: str
    text: str

    @property
    def chunk_id(self) -> str:
        # Стабильный ID на основе пути и имени, чтобы при повторной индексации
        # обновлялся тот же документ.
        h = hashlib.sha1(f"{self.file}::{self.name}".encode("utf-8")).hexdigest()
        # Qdrant требует UUID или uint64. Берём первые 16 hex → uint64.
        return str(int(h[:16], 16))


def chunk_python(file_path: Path, repo_root: Path) -> list[Chunk]:
    """Разбивает .py на функции / классы / module-level."""
    rel = file_path.relative_to(repo_root).as_posix()
    try:
        source = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    src_lines = source.splitlines()
    chunks: list[Chunk] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = node.end_lineno or start + 1
            text = "\n".join(src_lines[start:end])
            if not (MIN_CHUNK_CHARS <= len(text) <= MAX_CHUNK_CHARS):
                continue
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            chunks.append(
                Chunk(
                    file=rel, start_line=start + 1, end_line=end,
                    kind=kind, name=node.name, text=text,
                )
            )

    # Module docstring + top-level код.
    docstring = ast.get_docstring(tree)
    if docstring and len(docstring) >= MIN_CHUNK_CHARS:
        chunks.append(
            Chunk(
                file=rel, start_line=1, end_line=1,
                kind="module", name="__module_doc__", text=docstring,
            )
        )

    return chunks


def chunk_text(file_path: Path, repo_root: Path) -> list[Chunk]:
    """Разбивает .md/.txt на абзацы (разделитель — пустая строка)."""
    rel = file_path.relative_to(repo_root).as_posix()
    try:
        text = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    # Сшиваем абзацы в чанки до MAX_CHUNK_CHARS.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_size = 0
    para_idx = 0

    def flush(start_idx: int):
        nonlocal buf, buf_size
        if buf and buf_size >= MIN_CHUNK_CHARS:
            chunks.append(
                Chunk(
                    file=rel, start_line=start_idx, end_line=start_idx + len(buf),
                    kind="paragraph", name=f"para_{start_idx}",
                    text="\n\n".join(buf),
                )
            )
        buf = []
        buf_size = 0

    start_idx = 0
    for para in paragraphs:
        if buf_size + len(para) > MAX_CHUNK_CHARS and buf:
            flush(start_idx)
            start_idx = para_idx
        buf.append(para)
        buf_size += len(para)
        para_idx += 1
    flush(start_idx)

    return chunks


def extract_chunks(repo_root: Path, globs: list[str]) -> list[Chunk]:
    # Папки, которые никогда не индексируем: окружения, сборки, сырые корпусы
    # для обучения, чекпоинты, индекс самого Qdrant.
    SKIP_PARTS = {
        "__pycache__", ".venv", ".venv-wsl", "venv", "env",
        "build", "dist", "node_modules",
        "qdrant_storage", ".git", ".idea", ".vscode", ".pytest_cache",
        "checkpoints", "corpus", "packed", "tokenizer",
    }

    chunks: list[Chunk] = []
    for g in globs:
        for fp in sorted(repo_root.glob(g)):
            if any(part in fp.parts for part in SKIP_PARTS):
                continue
            if fp.suffix == ".py":
                chunks.extend(chunk_python(fp, repo_root))
            elif fp.suffix in (".md", ".txt"):
                chunks.extend(chunk_text(fp, repo_root))
    return chunks


# ----------------------------------------------------------------------
# Qdrant collection management
# ----------------------------------------------------------------------
def ensure_collection() -> None:
    print(f"[qdrant] checking collection '{COLLECTION}'…")
    try:
        info = http_request("GET", f"{QDRANT_URL}/collections/{COLLECTION}")
        existing_dim = info.get("result", {}).get("config", {}).get("params", {}) \
            .get("vectors", {}).get("size")
        if existing_dim != EMBED_DIM:
            print(f"  collection exists but dim={existing_dim}, expected {EMBED_DIM} — recreating")
            http_request("DELETE", f"{QDRANT_URL}/collections/{COLLECTION}")
        else:
            print(f"  exists with correct dim={EMBED_DIM}")
            return
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise

    print(f"  creating collection '{COLLECTION}'…")
    http_request(
        "PUT",
        f"{QDRANT_URL}/collections/{COLLECTION}",
        {
            "vectors": {
                "size": EMBED_DIM,
                "distance": "Cosine",
            },
            "optimizers_config": {
                "default_segment_number": 2,
            },
        },
    )
    print(f"  ok")


def embed_one(text: str) -> list[float] | None:
    """Получаем embedding через Ollama nomic-embed-text.

    Возвращает None, если Ollama вернула ошибку (например, 500 на слишком
    длинном тексте) — чтобы вызывающий код мог пропустить проблемный чанк,
    а не падать на всём датасете.
    """
    # Hard cap — обрезаем чанк, если он почему-то очень длинный (редкий .md).
    if len(text) > EMBED_HARD_CAP_CHARS:
        text = text[:EMBED_HARD_CAP_CHARS]

    try:
        resp = http_request(
            "POST",
            f"{OLLAMA_URL}/api/embeddings",
            {"model": EMBED_MODEL, "prompt": text},
            timeout=60.0,
        )
    except urllib.error.HTTPError as e:
        # 500 обычно = слишком длинный вход. Пробуем ещё раз с половиной.
        if e.code == 500 and len(text) > 2000:
            try:
                resp = http_request(
                    "POST",
                    f"{OLLAMA_URL}/api/embeddings",
                    {"model": EMBED_MODEL, "prompt": text[: len(text) // 2]},
                    timeout=60.0,
                )
            except Exception:
                return None
        else:
            return None
    except Exception:
        return None

    emb = resp.get("embedding", [])
    if not emb or len(emb) != EMBED_DIM:
        return None
    return emb


def upsert_batch(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    points = [
        {
            "id": int(c.chunk_id),
            "vector": v,
            "payload": {
                "file": c.file,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "kind": c.kind,
                "name": c.name,
                "text": c.text,
            },
        }
        for c, v in zip(chunks, embeddings)
    ]
    http_request(
        "PUT",
        f"{QDRANT_URL}/collections/{COLLECTION}/points",
        {"points": points},
        timeout=60.0,
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Index codebase into Qdrant.")
    parser.add_argument("--root", default=str(_ROOT))
    parser.add_argument(
        "--glob",
        action="append",
        default=None,
        help="Glob pattern (можно несколько). По умолчанию: kernel/**/*.py, training/**/*.py, **/*.md.",
    )
    args = parser.parse_args()

    if args.glob is None:
        args.glob = [
            "kernel/**/*.py",
            "training/**/*.py",
            "finetuning/**/*.py",
            "rag/**/*.py",
            "examples/**/*.py",
            "tests_parity/**/*.py",
            "tests_cuda/**/*.py",
            "**/*.md",
            "**/*.txt",
        ]

    repo_root = Path(args.root).resolve()
    print(f"[index] root: {repo_root}")
    print(f"[index] globs: {args.glob}")

    # Sanity: Qdrant и Ollama живы.
    # Используем API-эндпоинты, которые точно возвращают JSON.
    print()
    try:
        http_request("GET", f"{QDRANT_URL}/collections")
        print("[ok] Qdrant отвечает")
    except Exception as e:
        print(f"[error] Qdrant недоступен на {QDRANT_URL}: {e}")
        print("        Подними его: cd rag && docker compose up -d")
        return 1

    try:
        http_request("GET", f"{OLLAMA_URL}/api/tags")
        print("[ok] Ollama отвечает")
    except Exception as e:
        print(f"[error] Ollama недоступна на {OLLAMA_URL}: {e}")
        print("        Проверь что Ollama запущена (иконка ламы в трее).")
        return 1

    # Соберём чанки.
    print()
    print("[chunk] extracting…")
    chunks = extract_chunks(repo_root, args.glob)
    print(f"[chunk] total: {len(chunks)} chunks")

    if not chunks:
        print("[warn] нет чанков — проверь --glob")
        return 1

    by_kind: dict[str, int] = {}
    for c in chunks:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
    for k, n in sorted(by_kind.items()):
        print(f"        {k:12s}  {n}")

    # Готовим коллекцию.
    print()
    ensure_collection()

    # Embedding + upsert батчами.
    # Устойчиво к ошибкам: если конкретный чанк не эмбеддится — пропускаем,
    # продолжаем с остальными. В конце показываем сводку.
    print()
    print(f"[embed] computing embeddings (batch={UPSERT_BATCH})…")
    started = time.time()

    n_indexed = 0
    failed: list[Chunk] = []

    for i in range(0, len(chunks), UPSERT_BATCH):
        batch = chunks[i:i + UPSERT_BATCH]
        good_chunks: list[Chunk] = []
        good_vectors: list[list[float]] = []

        for c in batch:
            vec = embed_one(c.text)
            if vec is None:
                failed.append(c)
            else:
                good_chunks.append(c)
                good_vectors.append(vec)

        if good_chunks:
            try:
                upsert_batch(good_chunks, good_vectors)
                n_indexed += len(good_chunks)
            except Exception as e:
                print(f"  [warn] upsert batch failed: {e}")
                failed.extend(good_chunks)

        elapsed = time.time() - started
        rate = (i + len(batch)) / max(elapsed, 1e-6)
        print(f"  {i + len(batch):5d}/{len(chunks)}  "
              f"({rate:.1f} chunks/s, ok={n_indexed}, fail={len(failed)})")

    print()
    print(f"[done] indexed {n_indexed}/{len(chunks)} chunks in {time.time() - started:.1f}s")
    if failed:
        print(f"       failed: {len(failed)} chunks")
        for c in failed[:10]:
            print(f"         - {c.file}:{c.start_line} ({c.kind} {c.name}) — {len(c.text)} chars")
        if len(failed) > 10:
            print(f"         … and {len(failed) - 10} more")
    print(f"       collection: {COLLECTION}")
    print(f"       веб-UI: http://localhost:6333/dashboard#/collections/{COLLECTION}")
    print()
    print("Попробуй поиск:")
    print("  python rag/search.py 'как работает causal mask?'")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
