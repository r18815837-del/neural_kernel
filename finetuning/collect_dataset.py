"""
collect_dataset.py
==================

Собирает обучающий датасет из твоего проекта в трёх форматах:

  1. (commit message → diff)  — «реализуй такое-то изменение» → код
  2. (docstring → function body)  — описание функции → её реализация
  3. (test file → impl file)  — pytest → код, который оно тестит

Сохраняет в JSONL формате chat messages, готовом для Unsloth/SFTTrainer:

    {"messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]}

Usage:
    python finetuning/collect_dataset.py
    python finetuning/collect_dataset.py --max-commits 100 --min-diff-size 50
    python finetuning/collect_dataset.py --extra-projects ../other_proj1 ../other_proj2
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _ROOT / "data" / "finetune"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
@dataclass
class Example:
    user: str
    assistant: str
    source: str  # "commit" | "docstring" | "test"

    def to_chat(self) -> dict:
        return {
            "messages": [
                {"role": "user", "content": self.user},
                {"role": "assistant", "content": self.assistant},
            ],
            "meta": {"source": self.source},
        }


def run_git(args: list[str], cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        msg = e.stderr if hasattr(e, "stderr") and e.stderr else str(e)
        raise RuntimeError(f"git {' '.join(args)} failed: {msg}") from e


# ----------------------------------------------------------------------
# Source 1: commit messages → diffs
# ----------------------------------------------------------------------
def collect_from_git(
    repo_root: Path,
    max_commits: int,
    min_diff_size: int,
    max_diff_size: int,
) -> list[Example]:
    print(f"\n[git] mining commits from {repo_root}")

    if not (repo_root / ".git").exists():
        print(f"  [skip] not a git repo")
        return []

    log_text = run_git(
        ["log", "--pretty=format:%H%x09%s", f"-n{max_commits}"],
        cwd=repo_root,
    )

    commits = []
    for line in log_text.strip().splitlines():
        if "\t" not in line:
            continue
        sha, subject = line.split("\t", 1)
        commits.append((sha.strip(), subject.strip()))

    print(f"  found {len(commits)} commits")

    examples: list[Example] = []
    skipped_too_small = 0
    skipped_too_big = 0
    skipped_merge = 0

    for sha, subject in commits:
        # Пропускаем merge-коммиты (Subject обычно "Merge ...").
        if subject.lower().startswith("merge"):
            skipped_merge += 1
            continue

        # Пропускаем коммиты, которые сами по себе мусорные.
        if len(subject) < 5 or subject.lower() in {"wip", "fix", "update", "test"}:
            continue

        diff = run_git(
            [
                "show",
                "--no-color",
                "--format=",
                "--unified=3",
                "--no-renames",
                sha,
            ],
            cwd=repo_root,
        ).strip()

        # Фильтруем по размеру.
        if len(diff) < min_diff_size:
            skipped_too_small += 1
            continue
        if len(diff) > max_diff_size:
            skipped_too_big += 1
            continue

        # Игнорируем диффы, которые состоят почти только из удалений
        # (не интересно для обучения «как добавить фичу»).
        added = sum(1 for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
        removed = sum(1 for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---"))
        if added < removed:
            continue

        # Промпт — как будто пользователь просит сделать это.
        user_prompt = (
            f"Make the following change in this codebase:\n\n"
            f"{subject}\n\n"
            f"Show me the full diff."
        )

        examples.append(
            Example(
                user=user_prompt,
                assistant=f"```diff\n{diff}\n```",
                source="commit",
            )
        )

    print(
        f"  collected {len(examples)} commit examples "
        f"(skipped: {skipped_merge} merges, {skipped_too_small} too small, "
        f"{skipped_too_big} too big)"
    )
    return examples


# ----------------------------------------------------------------------
# Source 2: docstrings → function bodies
# ----------------------------------------------------------------------
def _extract_function_pairs(file_path: Path) -> list[Example]:
    try:
        source = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    examples: list[Example] = []
    src_lines = source.splitlines()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Игнорируем приватные / dunder функции — обычно неинтересно.
        if node.name.startswith("_"):
            continue

        # Берём docstring.
        docstring = ast.get_docstring(node)
        if not docstring or len(docstring) < 30:
            continue

        # Берём текст функции по line numbers.
        start = node.lineno - 1
        end = (node.end_lineno or start + 1)
        func_text = "\n".join(src_lines[start:end])

        # Слишком короткие/длинные функции — пропускаем.
        if len(func_text) < 80 or len(func_text) > 4000:
            continue

        # Сигнатура для промпта.
        sig = src_lines[start].strip()

        user_prompt = (
            f"Implement this function for the `neural_kernel` framework:\n\n"
            f"```python\n{sig}\n    \"\"\"{docstring}\"\"\"\n```\n\n"
            f"Return only the function body (the part after the docstring)."
        )

        examples.append(
            Example(
                user=user_prompt,
                assistant=f"```python\n{func_text}\n```",
                source="docstring",
            )
        )

    return examples


def collect_docstring_pairs(repo_root: Path, code_glob: str) -> list[Example]:
    print(f"\n[docstrings] scanning {repo_root}/{code_glob}")

    examples: list[Example] = []
    py_files = sorted(repo_root.glob(code_glob))

    skipped_files = 0
    for fp in py_files:
        # Пропускаем тесты и бенчмарки — для них есть отдельный сборщик.
        rel = fp.relative_to(repo_root).as_posix()
        if any(part in rel for part in ("/tests", "tests_", "benchmarks/", "/__pycache__/")):
            skipped_files += 1
            continue

        examples.extend(_extract_function_pairs(fp))

    print(f"  found {len(examples)} docstring→body pairs (skipped {skipped_files} test/bench files)")
    return examples


# ----------------------------------------------------------------------
# Source 3: test files → implementation files
# ----------------------------------------------------------------------
_SKIP_DIRS = {
    "__pycache__", ".git", ".venv", ".venv-wsl", "venv", "env",
    "node_modules", "build", "dist", ".pytest_cache",
    ".idea", ".vscode", ".github", "qdrant_storage",
}


def _walk_python_files(root: Path):
    """Быстрый walk по проекту, минующий тяжёлые папки (.venv и т.п.).

    Гораздо быстрее Path.rglob, потому что не уходит в site-packages.
    """
    import os
    for dirpath, dirnames, filenames in os.walk(root):
        # Срезаем тяжёлые папки на месте — os.walk их не посетит.
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


def _guess_impl_path(test_path: Path, repo_root: Path, all_py: list[Path] | None = None) -> Path | None:
    """Эвристика: test_xxx_yyy.py обычно тестит kernel/.../xxx.py или xxx/yyy.py."""
    name = test_path.stem
    if not name.startswith("test_"):
        return None

    target_name = name[len("test_"):]
    parts = target_name.split("_")

    # Берём первые 1-2 значимых слова.
    candidates_names = [f"{parts[0]}.py"]
    if len(parts) >= 2:
        candidates_names.append(f"{'_'.join(parts[:2])}.py")

    # Используем заранее собранный список файлов, если передан, иначе walk.
    files = all_py if all_py is not None else list(_walk_python_files(repo_root))

    for cand in candidates_names:
        for impl in files:
            if impl.name != cand:
                continue
            rel = impl.relative_to(repo_root).as_posix()
            if "test" in rel:
                continue
            return impl

    return None


def collect_test_impl_pairs(repo_root: Path, test_globs: list[str]) -> list[Example]:
    print(f"\n[tests] scanning test files in {repo_root}")

    test_files: list[Path] = []
    for g in test_globs:
        test_files.extend(repo_root.glob(g))

    # Один раз собираем список всех Python-файлов в проекте, минуя
    # .venv / __pycache__ / .git и т.п. — иначе rglob уходит надолго.
    all_py = list(_walk_python_files(repo_root))

    examples: list[Example] = []
    skipped = 0

    for tp in sorted(set(test_files)):
        impl = _guess_impl_path(tp, repo_root, all_py=all_py)
        if impl is None or not impl.exists():
            skipped += 1
            continue

        try:
            test_text = tp.read_text(encoding="utf-8")
            impl_text = impl.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        # Слишком большие/маленькие — пропускаем.
        if len(test_text) < 200 or len(test_text) > 6000:
            continue
        if len(impl_text) < 100 or len(impl_text) > 6000:
            continue

        rel_impl = impl.relative_to(repo_root).as_posix()

        user_prompt = (
            f"Here are pytest tests. Write the implementation in `{rel_impl}` "
            f"that makes them pass:\n\n"
            f"```python\n{test_text}\n```"
        )

        examples.append(
            Example(
                user=user_prompt,
                assistant=f"```python\n# {rel_impl}\n{impl_text}\n```",
                source="test",
            )
        )

    print(f"  collected {len(examples)} test→impl pairs (skipped {skipped} unmatched)")
    return examples


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Collect fine-tuning dataset.")
    parser.add_argument("--max-commits", type=int, default=300)
    parser.add_argument("--min-diff-size", type=int, default=80)
    parser.add_argument("--max-diff-size", type=int, default=8000)
    parser.add_argument(
        "--code-glob",
        default="kernel/**/*.py",
        help="Glob внутри проекта для Python-файлов с docstrings.",
    )
    parser.add_argument(
        "--test-glob",
        action="append",
        default=None,
        help="Glob для тестов (можно несколько раз).",
    )
    parser.add_argument(
        "--extra-projects",
        nargs="*",
        default=[],
        help="Доп. пути к другим проектам (увеличить датасет).",
    )
    parser.add_argument(
        "--out",
        default=str(_DATA_DIR / "dataset.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.test_glob is None:
        args.test_glob = ["tests/**/test_*.py", "tests_parity/**/test_*.py"]

    # Все проекты, которые мы майним.
    projects = [_ROOT, *(Path(p).resolve() for p in args.extra_projects)]
    print(f"Projects to mine: {[str(p) for p in projects]}")

    all_examples: list[Example] = []
    for proj in projects:
        if not proj.exists():
            print(f"[warn] project not found: {proj}")
            continue

        all_examples.extend(
            collect_from_git(
                proj,
                max_commits=args.max_commits,
                min_diff_size=args.min_diff_size,
                max_diff_size=args.max_diff_size,
            )
        )
        all_examples.extend(collect_docstring_pairs(proj, args.code_glob))
        all_examples.extend(collect_test_impl_pairs(proj, args.test_glob))

    # Дедупликация по user-промпту (на случай дублирующих коммитов).
    seen = set()
    unique: list[Example] = []
    for ex in all_examples:
        key = ex.user[:200]
        if key in seen:
            continue
        seen.add(key)
        unique.append(ex)

    print(f"\n[total] {len(all_examples)} raw → {len(unique)} after dedup")

    # Перемешиваем.
    import random
    random.Random(args.seed).shuffle(unique)

    # Запись.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in unique:
            f.write(json.dumps(ex.to_chat(), ensure_ascii=False) + "\n")

    print(f"\n[done] wrote {len(unique)} examples → {out_path}")
    print(f"       size: {out_path.stat().st_size / 1024:.1f} KB")

    # Статистика по источникам.
    from collections import Counter
    by_source = Counter(ex.source for ex in unique)
    print(f"\n  by source:")
    for src, n in by_source.most_common():
        print(f"    {src:12s}  {n:5d}")

    # Подсказка для просмотра.
    print(f"\nПосмотри 5 случайных примеров:")
    print(f"  python -c \"import json,random; "
          f"lines=open(r'{out_path}',encoding='utf-8').read().splitlines(); "
          f"[print(json.dumps(json.loads(l),indent=2,ensure_ascii=False),'\\n---') "
          f"for l in random.sample(lines,5)]\"")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
