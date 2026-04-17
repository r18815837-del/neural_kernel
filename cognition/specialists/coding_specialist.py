"""Coding specialist — analyzes, explains, and helps fix Python and Dart code."""
from __future__ import annotations

import ast
import re
import textwrap
from typing import Optional

from .base_specialist import BaseSpecialist, SpecialistResult

_PYTHON_SIGNALS = [
    r'\bdef\s+\w+\s*\(', r'\bclass\s+\w+', r'\bimport\s+\w+',
    r'\bfrom\s+\w+\s+import', r'\bif\s+__name__\s*==', r'\bprint\s*\(',
    r'\breturn\s+', r'\bfor\s+\w+\s+in\s+', r'\bwhile\s+',
    r'\btry\s*:', r'\bexcept\s+', r'\bwith\s+', r'\basync\s+def\s+',
    r'\bawait\s+', r'\byield\s+', r'\blambda\s+', r'\braise\s+', r'\bassert\s+',
]

_DART_SIGNALS = [
    r'\bvoid\s+\w+\s*\(', r'\bWidget\s+build\b', r'\bStatefulWidget\b',
    r'\bStatelessWidget\b', r'\bFuture<', r'\bStream<',
    r'\bfinal\s+\w+\s*=', r'\bvar\s+\w+\s*=', r'\bconst\s+\w+\s*\(',
    r'\boverride\b', r'\bextends\s+\w+', r'\bimplements\s+\w+',
    r'\basync\s*\{', r'\bawait\s+', r'=>\s*',
    r'\bmain\s*\(\s*\)\s*\{', r'\bString\b', r'\bint\b\s+\w+',
    r'\bdouble\b\s+\w+', r'\bbool\b\s+\w+', r'\bprint\s*\(.+\)\s*;',
    r'\bList<', r'\bMap<', r'\bProvider\b', r'\bStateNotifier\b',
    r'\bConsumerWidget\b', r'\bref\.\w+', r';\s*$',
]

_DART_STRONG = [
    r'\bWidget\b', r'\bStatelessWidget\b', r'\bStatefulWidget\b',
    r'\b@override\b', r'\bBuildContext\b', r'\bScaffold\b',
    r'\bContainer\b', r'\bMaterialApp\b',
]


def detect_language(text: str) -> Optional[str]:
    py_score = sum(1 for p in _PYTHON_SIGNALS if re.search(p, text))
    dart_score = sum(1 for p in _DART_SIGNALS if re.search(p, text))

    dart_strong = sum(1 for p in _DART_STRONG if re.search(p, text))
    if dart_strong >= 1:
        dart_score += dart_strong * 2

    if dart_score > py_score and dart_score >= 2:
        return "dart"
    if py_score >= 2:
        return "python"
    if dart_score >= 2:
        return "dart"
    if py_score == 1 and dart_score == 0:
        return "python"
    if dart_score == 1 and py_score == 0:
        return "dart"
    return None


def extract_code_block(text: str) -> Optional[str]:
    m = re.search(r'```(?:\w+)?\s*\n(.+?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()

    _CODE_STARTERS = (
        'def ', 'class ', 'import ', 'from ', 'for ', 'if ',
        'while ', 'try:', 'return ', 'print(', '#', '@',
        'raise ', 'with ', 'async ', 'yield ', 'assert ',
        'except', 'finally:', 'elif ', 'else:',
    )
    lines = text.split('\n')
    code_lines = []
    for line in lines:
        stripped = line.strip()
        if (stripped.startswith(_CODE_STARTERS)
                or (line.startswith('    ') and stripped)
                or re.match(r'^\s*\w+\s*[=\(]', stripped)):
            code_lines.append(line)

    if code_lines:
        return '\n'.join(code_lines)
    return None


def analyze_python(code: str) -> dict:
    result = {
        "syntax_valid": False, "syntax_error": None,
        "functions": [], "classes": [], "imports": [],
        "issues": [], "suggestions": [], "complexity": "simple",
    }

    try:
        tree = ast.parse(code)
        result["syntax_valid"] = True
    except SyntaxError as e:
        result["syntax_error"] = f"Line {e.lineno}: {e.msg}"
        result["issues"].append(f"Syntax error at line {e.lineno}: {e.msg}")
        return result

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_info = {
                "name": node.name,
                "args": [a.arg for a in node.args.args],
                "line": node.lineno,
                "has_docstring": (
                    isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, (ast.Constant, ast.Str))
                ) if node.body else False,
                "has_return": any(isinstance(n, ast.Return) for n in ast.walk(node)),
            }
            result["functions"].append(func_info)

            if not func_info["has_docstring"] and not node.name.startswith('_'):
                result["suggestions"].append(
                    f"Function '{node.name}' has no docstring — consider adding one."
                )
            if len(func_info["args"]) > 5:
                result["issues"].append(
                    f"Function '{node.name}' has {len(func_info['args'])} arguments — consider using a dataclass or dict."
                )

        elif isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            result["classes"].append({
                "name": node.name, "line": node.lineno,
                "methods": methods, "num_methods": len(methods),
            })

        elif isinstance(node, ast.Import):
            for alias in node.names:
                result["imports"].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                result["imports"].append(node.module)

    if 'except:' in code or 'except Exception:' in code:
        result["issues"].append("Bare 'except' catches all exceptions — catch specific ones instead.")
    if re.search(r'==\s*True\b|==\s*False\b', code):
        result["suggestions"].append("Use 'if x:' instead of 'if x == True:' — more Pythonic.")
    if re.search(r'==\s*None\b', code):
        result["suggestions"].append("Use 'is None' instead of '== None' for identity comparison.")
    if 'global ' in code:
        result["issues"].append("Using 'global' is generally discouraged — consider passing values as arguments.")
    if re.search(r'\beval\s*\(', code):
        result["issues"].append("Using eval() is a security risk — avoid if possible.")
    if re.search(r'\braise\s+NotImplemented\b(?!Error)', code):
        result["issues"].append(
            "raise NotImplemented is wrong — use raise NotImplementedError. "
            "NotImplemented is a special singleton for __eq__/__add__, not an exception."
        )
    if re.search(r'\bexcept\s+\w+\s*,\s*\w+\s*:', code):
        result["issues"].append("Old-style 'except Error, e:' — use 'except Error as e:' instead (Python 3 syntax).")
    if re.search(r'\bmutable\s+default\b|def\s+\w+\([^)]*=\s*(\[\]|\{\})', code):
        result["issues"].append("Mutable default argument (list/dict) — use None and create inside function.")
    if re.search(r'except\s+BaseException\b', code):
        result["issues"].append("Catching BaseException is too broad — it catches KeyboardInterrupt and SystemExit.")
    if re.search(r'\btype\s*\(\s*\w+\s*\)\s*==', code):
        result["suggestions"].append("Use isinstance() instead of type() == for type checking.")

    num_funcs = len(result["functions"])
    num_classes = len(result["classes"])
    if num_funcs > 5 or num_classes > 2:
        result["complexity"] = "complex"
    elif num_funcs > 2 or num_classes > 0:
        result["complexity"] = "moderate"

    return result


_DECORATOR_EXPLANATIONS = {
    "@property": "Python decorator that turns a method into a read-only attribute. Use @name.setter to add a setter.",
    "@staticmethod": "Method that doesn't receive self/cls. Called on class directly: MyClass.method().",
    "@classmethod": "Method that receives cls instead of self. Used for alternative constructors.",
    "@abstractmethod": "Forces subclasses to implement this method. Class must inherit from ABC.",
    "@dataclass": "Auto-generates __init__, __repr__, __eq__. Fields defined as class-level type annotations.",
    "@override": "Dart/Python 3.12+ marker that a method overrides a parent method.",
    "@cached_property": "Like @property but caches the result after first call.",
    "@wraps": "Preserves the original function's metadata when writing decorators.",
}

_PROGRAMMING_QUESTIONS = {
    "sort": "Use sorted() for a new list, or list.sort() for in-place. For custom order: sorted(items, key=lambda x: x.name)",
    "reverse": "Use reversed() or [::-1]. Example: 'hello'[::-1] -> 'olleh'",
    "list comprehension": "[expression for item in iterable if condition]. Example: [x**2 for x in range(10) if x % 2 == 0]",
    "dict comprehension": "{key: value for item in iterable}. Example: {k: v for k, v in zip(keys, values)}",
    "async": "Use async def for coroutines, await to call. Run with asyncio.run(main()).",
    "decorator": "Wraps a function. Use @decorator_name above def.",
    "class": "Define with class Name:. Use __init__ for constructor, self for instance.",
    "exception": "Use try/except/finally. Custom: class MyError(Exception): pass",
    "file": "Use with open() as context manager. Read: f.read(). Write: f.write(data).",
    "regex": "import re. re.search(pattern, text) for finding, re.sub() for replacing.",
    "type hint": "function(x: int) -> str. Import from typing: List, Dict, Optional. Python 3.10+: use |.",
    "pytest": "Name files test_*.py, functions test_*. Assert with assert. Fixtures with @pytest.fixture.",
    "virtual environment": "python -m venv .venv. Activate: source .venv/bin/activate. Install: pip install -r requirements.txt",
    "widget": "Flutter: StatelessWidget (immutable) or StatefulWidget (with State). Override build().",
    "riverpod": "Provider for values, StateProvider for mutable, FutureProvider for async. Access with ref.watch().",
    "go_router": "GoRoute(path, builder). Navigate with context.go('/path'). ShellRoute for persistent nav.",
}


def _answer_question(question: str) -> Optional[str]:
    q_lower = question.lower()
    for keyword, answer in _PROGRAMMING_QUESTIONS.items():
        if keyword in q_lower:
            return answer
    return None


def _format_analysis(analysis: dict, code: str) -> str:
    parts = []
    if not analysis["syntax_valid"]:
        parts.append(f"Syntax error found: {analysis['syntax_error']}")
        return '\n'.join(parts)

    funcs = analysis["functions"]
    classes = analysis["classes"]
    summary = []
    if funcs:
        summary.append(f"{len(funcs)} function(s): {', '.join(f['name'] for f in funcs)}")
    if classes:
        summary.append(f"{len(classes)} class(es): {', '.join(c['name'] for c in classes)}")
    if analysis["imports"]:
        summary.append(f"Imports: {', '.join(analysis['imports'])}")
    if summary:
        parts.append("Code analysis: " + '; '.join(summary) + '.')

    if funcs:
        for f in funcs[:3]:
            args_str = ', '.join(f['args']) if f['args'] else 'no args'
            parts.append(f"  - {f['name']}({args_str}) at line {f['line']}")

    if analysis["issues"]:
        parts.append("\nIssues found:")
        for issue in analysis["issues"]:
            parts.append(f"  - {issue}")

    if analysis["suggestions"]:
        parts.append("\nSuggestions:")
        for sug in analysis["suggestions"]:
            parts.append(f"  - {sug}")

    if not analysis["issues"] and not analysis["suggestions"]:
        parts.append("\nNo issues found — code looks clean!")

    parts.append(f"\nComplexity: {analysis['complexity']}")
    return '\n'.join(parts)


class CodingSpecialist(BaseSpecialist):
    topic = "coding"
    keywords = [
        "python", "dart", "flutter", "code", "function", "class",
        "bug", "error", "fix", "programming", "algorithm", "def ",
        "import ", "widget", "async", "await", "list", "dict",
        "sort", "how to", "write", "create", "implement",
        "explain", "what does", "review", "check",
    ]

    def can_handle(self, question: str) -> bool:
        if super().can_handle(question):
            return True
        if detect_language(question) is not None:
            return True
        code_patterns = [
            r'^\s*@\w+', r'^\s*raise\s+\w+', r'^\s*return\s+',
            r'^\s*yield\s+', r'\w+\s*\([^)]*\)\s*[:{]',
            r'^\s*#\s*\w+', r'\.\w+\s*\(',
        ]
        return any(re.search(p, question, re.MULTILINE) for p in code_patterns)

    def handle(self, question: str) -> SpecialistResult:
        dedented = textwrap.dedent(question).strip()
        code = extract_code_block(dedented)
        lang = detect_language(dedented)

        if not code and lang:
            code = dedented
        if not lang and code:
            if re.match(r'^@\w+', code) or re.match(r'^raise\s+', code):
                lang = "python"
        if not code and not lang:
            if re.match(r'^@\w+', dedented) or re.match(r'^raise\s+', dedented):
                code = dedented
                lang = "python"

        if code and re.match(r'^@\w+$', code.strip()):
            decorator = code.strip()
            explanation = _DECORATOR_EXPLANATIONS.get(decorator)
            if explanation:
                return SpecialistResult(
                    topic=self.topic, answer=explanation, confidence=0.9,
                    tips=[f"Use '{decorator}' above a method/class definition."],
                )
            return SpecialistResult(
                topic=self.topic,
                answer=f"'{decorator}' is a Python decorator. Place it above a def or class statement.",
                confidence=0.7,
                tips=["Common decorators: @property, @staticmethod, @classmethod, @dataclass"],
            )

        if code and (lang == "python" or detect_language(code) == "python"):
            analysis = analyze_python(code)
            explanation = _format_analysis(analysis, code)
            confidence = 0.9 if analysis["syntax_valid"] else 0.8
            return SpecialistResult(
                topic=self.topic, answer=explanation, confidence=confidence,
                tips=analysis.get("suggestions", [])[:3],
            )

        if code and (lang == "dart" or detect_language(code) == "dart"):
            suggestions = []
            if re.search(r'\bprint\s*\(', code) and 'debugPrint' not in code:
                suggestions.append("Consider using debugPrint() instead of print() in Flutter apps.")
            if re.search(r'\bsetState\b', code) and 'mounted' not in code:
                suggestions.append("Check 'mounted' before calling setState() in async callbacks.")
            if re.search(r'\bvar\s+\w+', code) and 'final' not in code:
                suggestions.append("Prefer 'final' over 'var' when the value won't change.")

            desc = f"Dart code detected ({len(code.splitlines())} line(s))."
            if suggestions:
                desc += "\n\nSuggestions: " + "; ".join(suggestions)
            else:
                desc += " No issues found — code looks clean!"

            return SpecialistResult(
                topic=self.topic, answer=desc, confidence=0.75, tips=suggestions[:3],
            )

        answer = _answer_question(question)
        if answer:
            return SpecialistResult(topic=self.topic, answer=answer, confidence=0.85, tips=[])

        if lang:
            return SpecialistResult(
                topic=self.topic,
                answer=f"I can help with {lang.capitalize()} code! Paste your code for analysis, or ask a specific question.",
                confidence=0.6, tips=["Paste code to get analysis", "Ask 'how to...' for examples"],
            )

        return SpecialistResult(
            topic=self.topic,
            answer=f"Programming question detected: '{question[:80]}'. I can analyze Python and Dart code, find bugs, and suggest improvements.",
            confidence=0.5,
            tips=["Paste code between ``` markers for best results"],
        )
