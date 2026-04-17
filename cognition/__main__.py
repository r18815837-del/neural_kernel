"""Interactive CLI for the cognition engine.

Usage:
    python -m cognition
"""

from __future__ import annotations

import json
import sys

from .orchestrator import Orchestrator
from .specialists.coding_specialist import CodingSpecialist, analyze_python, detect_language
from .specialists.code_executor import CodeExecutor


def _read_code_block() -> str:
    """Read multi-line code input until user types 'END'."""
    print("  Paste your code below (type END on a new line to finish):")
    lines = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def _handle_code_command(args: str) -> None:
    """Handle /code commands."""
    parts = args.strip().split(maxsplit=1)
    subcmd = parts[0].lower() if parts else "help"

    if subcmd == "help":
        print("  /code analyze  — paste code for analysis")
        print("  /code run      — paste code to execute")
        print("  /code ask      — ask a programming question")
        print()
        return

    if subcmd == "analyze":
        code = _read_code_block()
        if not code.strip():
            print("  No code provided.\n")
            return

        lang = detect_language(code)
        print(f"\n  Language: {lang or 'unknown'}")

        if lang == "python":
            analysis = analyze_python(code)
            if not analysis["syntax_valid"]:
                print(f"  Syntax Error: {analysis['syntax_error']}")
            else:
                if analysis["functions"]:
                    print(f"  Functions: {', '.join(f['name'] for f in analysis['functions'])}")
                if analysis["classes"]:
                    print(f"  Classes: {', '.join(c['name'] for c in analysis['classes'])}")
                if analysis["imports"]:
                    print(f"  Imports: {', '.join(analysis['imports'])}")
                if analysis["issues"]:
                    print("  Issues:")
                    for issue in analysis["issues"]:
                        print(f"    - {issue}")
                if analysis["suggestions"]:
                    print("  Suggestions:")
                    for sug in analysis["suggestions"]:
                        print(f"    - {sug}")
                if not analysis["issues"] and not analysis["suggestions"]:
                    print("  No issues found — code looks clean!")
                print(f"  Complexity: {analysis['complexity']}")
        else:
            print("  (Full analysis available for Python. Dart: basic detection only.)")

        print()
        return

    if subcmd == "run":
        code = _read_code_block()
        if not code.strip():
            print("  No code provided.\n")
            return

        executor = CodeExecutor(timeout=5)
        result = executor.run(code)

        if result.success:
            print(f"\n  Output:\n{result.stdout}")
        else:
            print(f"\n  Error: {result.error_summary}")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")
            if result.timed_out:
                print("  (execution timed out)")
        print()
        return

    if subcmd == "ask":
        question = parts[1] if len(parts) > 1 else ""
        if not question:
            try:
                question = input("  Your question: ").strip()
            except (EOFError, KeyboardInterrupt):
                return

        specialist = CodingSpecialist()
        result = specialist.handle(question)
        print(f"\n  NK Code > {result.answer}")
        if result.tips:
            print(f"  Tips: {', '.join(result.tips)}")
        print()
        return

    print(f"  Unknown subcommand: {subcmd}. Try /code help\n")


def main() -> None:
    orch = Orchestrator()

    lm_status = "LM loaded" if orch._lm else "no LM (train first)"

    print()
    print("  Neural Kernel — Cognition Engine")
    print(f"  Backend: {lm_status}")
    print("  Type a question and press Enter.")
    print("  Commands:  /memory  /mistakes  /code  /quit")
    print("  " + "-" * 40)
    print()

    while True:
        try:
            text = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not text:
            continue

        # --- Commands ---
        if text.lower() in ("/quit", "/exit", "/q"):
            print("Bye!")
            break

        if text.lower() == "/memory":
            print(json.dumps(orch.memory.dump(), indent=2, ensure_ascii=False))
            print()
            continue

        if text.lower() == "/mistakes":
            print(json.dumps(orch.learner.summary(), indent=2, ensure_ascii=False))
            print()
            continue

        if text.lower().startswith("/code"):
            args = text[5:].strip()
            _handle_code_command(args if args else "help")
            continue

        # --- Ask ---
        result = orch.ask_sync(text)

        print()
        print(f"NK > {result.answer}")
        print()
        print(
            f"     sources: {result.sources}  "
            f"confidence: {result.confidence.value}"
        )

        # Show steps in compact form.
        steps = " → ".join(
            f"{s.kind.value}({'ok' if 'ok' in s.output_summary else s.output_summary[:20]})"
            for s in result.steps
        )
        print(f"     pipeline: {steps}")
        print()


if __name__ == "__main__":
    main()
