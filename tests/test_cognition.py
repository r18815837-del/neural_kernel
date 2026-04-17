"""Tests for the cognition engine — orchestrator, memory, specialists."""

import os
import tempfile

import pytest

from cognition.memory import Memory, MemoryEntry
from cognition.models import Query, Confidence
from cognition.orchestrator import Orchestrator
from cognition.specialists.coding_specialist import (
    CodingSpecialist,
    analyze_python,
    detect_language,
    extract_code_block,
)
from cognition.specialists.code_executor import CodeExecutor


# ------------------------------------------------------------------
# Language detection
# ------------------------------------------------------------------


class TestLanguageDetection:
    def test_detect_python_def(self):
        assert detect_language("def foo():\n    pass") == "python"

    def test_detect_python_import(self):
        assert detect_language("import os\nimport sys") == "python"

    def test_detect_dart_void_main(self):
        assert detect_language("void main() {\n  print('hi');\n}") == "dart"

    def test_detect_dart_flutter_widget(self):
        code = "class MyApp extends StatelessWidget {\n  Widget build(BuildContext c) => Container();\n}"
        assert detect_language(code) == "dart"

    def test_detect_dart_typed_vars(self):
        assert detect_language("final String name = 'nk';\nint count = 42;") == "dart"

    def test_detect_unknown_plain_text(self):
        assert detect_language("hello world how are you") is None

    def test_detect_dart_riverpod(self):
        code = "final p = StateNotifierProvider<A, B>((ref) => A());"
        assert detect_language(code) == "dart"


# ------------------------------------------------------------------
# Python AST analysis
# ------------------------------------------------------------------


class TestPythonAnalysis:
    def test_valid_syntax(self):
        r = analyze_python("x = 1")
        assert r["syntax_valid"] is True

    def test_syntax_error(self):
        r = analyze_python("def foo(\n")
        assert r["syntax_valid"] is False
        assert r["syntax_error"] is not None

    def test_find_functions(self):
        r = analyze_python("def add(a, b):\n    return a + b\ndef sub(a, b):\n    return a - b")
        names = [f["name"] for f in r["functions"]]
        assert "add" in names
        assert "sub" in names

    def test_find_classes(self):
        r = analyze_python("class Foo:\n    pass\nclass Bar:\n    pass")
        names = [c["name"] for c in r["classes"]]
        assert "Foo" in names
        assert "Bar" in names

    def test_find_imports(self):
        r = analyze_python("import os\nfrom sys import path")
        assert "os" in r["imports"]
        assert "sys" in r["imports"]

    def test_detect_bare_except(self):
        r = analyze_python("try:\n    x = 1\nexcept:\n    pass")
        assert any("except" in i.lower() for i in r["issues"])

    def test_detect_eval(self):
        r = analyze_python("x = eval('1+1')")
        assert any("eval" in i.lower() for i in r["issues"])

    def test_detect_raise_not_implemented(self):
        r = analyze_python("raise NotImplemented")
        assert any("NotImplementedError" in i for i in r["issues"])

    def test_detect_mutable_default(self):
        r = analyze_python("def f(x, lst=[]):\n    pass")
        assert any("Mutable" in i for i in r["issues"])

    def test_missing_docstring_suggestion(self):
        r = analyze_python("def foo():\n    pass")
        assert any("docstring" in s.lower() for s in r["suggestions"])

    def test_clean_code(self):
        r = analyze_python('def _helper():\n    """Do stuff."""\n    return 42')
        assert r["issues"] == []

    def test_complexity_simple(self):
        r = analyze_python("x = 1")
        assert r["complexity"] == "simple"

    def test_complexity_moderate(self):
        r = analyze_python("class A:\n    pass\ndef f():\n    pass\ndef g():\n    pass\ndef h():\n    pass")
        assert r["complexity"] in ("moderate", "complex")


# ------------------------------------------------------------------
# Code extraction
# ------------------------------------------------------------------


class TestCodeExtraction:
    def test_extract_backtick_block(self):
        text = "Look at this:\n```python\ndef foo():\n    pass\n```\nDone."
        assert "def foo" in extract_code_block(text)

    def test_extract_single_line_code(self):
        assert extract_code_block("def hello(): pass") is not None

    def test_extract_decorator(self):
        assert extract_code_block("@property") is not None

    def test_no_code_in_plain_text(self):
        assert extract_code_block("hello how are you today") is None


# ------------------------------------------------------------------
# Code executor
# ------------------------------------------------------------------


class TestCodeExecutor:
    @pytest.fixture
    def executor(self):
        return CodeExecutor(timeout=5)

    def test_simple_print(self, executor):
        r = executor.run("print(42)")
        assert r.success is True
        assert "42" in r.stdout

    def test_math(self, executor):
        r = executor.run("print(2 ** 10)")
        assert r.success
        assert "1024" in r.stdout

    def test_error_handling(self, executor):
        r = executor.run("1/0")
        assert r.success is False
        assert "ZeroDivision" in (r.error_summary or "")

    def test_block_os_system(self, executor):
        r = executor.run("import os; os.system('ls')")
        assert r.success is False
        assert "Blocked" in (r.error_summary or "")

    def test_block_subprocess(self, executor):
        r = executor.run("import subprocess")
        assert r.success is False

    def test_block_eval(self, executor):
        r = executor.run("eval('1+1')")
        assert r.success is False

    def test_block_infinite_loop(self, executor):
        r = executor.run("while True:\n    pass")
        assert r.success is False

    def test_to_dict(self, executor):
        r = executor.run("print('ok')")
        d = r.to_dict()
        assert "stdout" in d
        assert "success" in d
        assert d["success"] is True


# ------------------------------------------------------------------
# Coding specialist
# ------------------------------------------------------------------


class TestCodingSpecialist:
    @pytest.fixture
    def specialist(self):
        return CodingSpecialist()

    def test_can_handle_python_code(self, specialist):
        assert specialist.can_handle("def foo(): pass")

    def test_can_handle_decorator(self, specialist):
        assert specialist.can_handle("@property")

    def test_can_handle_raise(self, specialist):
        assert specialist.can_handle("raise NotImplemented")

    def test_cannot_handle_plain_text(self, specialist):
        # Plain greeting — no code signals.
        assert not specialist.can_handle("good morning")

    def test_handle_python_code(self, specialist):
        r = specialist.handle("def add(a, b):\n    return a + b")
        assert "add" in r.answer
        assert r.confidence >= 0.8

    def test_handle_decorator_explanation(self, specialist):
        r = specialist.handle("@property")
        assert "decorator" in r.answer.lower() or "property" in r.answer.lower()

    def test_handle_raise_notimplemented(self, specialist):
        r = specialist.handle("raise NotImplemented")
        assert "NotImplementedError" in r.answer

    def test_handle_sort_question(self, specialist):
        r = specialist.handle("how to sort a list in python?")
        assert "sort" in r.answer.lower()
        assert r.confidence >= 0.8

    def test_handle_dart_code(self, specialist):
        r = specialist.handle("void main() {\n  print('hello');\n}")
        assert "dart" in r.answer.lower() or "Dart" in r.answer


# ------------------------------------------------------------------
# Memory
# ------------------------------------------------------------------


class TestMemoryInMemory:
    """Test Memory with in-memory backend (no store)."""

    def test_remember_and_recall(self):
        mem = Memory()
        q = Query(text="What is X?")
        q.answer = "X is cool."
        q.confidence = Confidence.HIGH
        mem.remember(q)
        assert mem.recall("what is x?") == "X is cool."

    def test_recall_miss(self):
        mem = Memory()
        assert mem.recall("unknown") is None

    def test_promote(self):
        mem = Memory()
        q = Query(text="Test")
        q.answer = "Answer"
        q.confidence = Confidence.HIGH
        mem.remember(q)
        # Hit it enough times to promote.
        for _ in range(Memory.PROMOTE_THRESHOLD):
            mem.promote("test")
        dump = mem.dump()
        assert len(dump["long_term"]) == 1

    def test_forget(self):
        mem = Memory()
        q = Query(text="Forget me")
        q.answer = "Gone"
        q.confidence = Confidence.HIGH
        mem.remember(q)
        assert mem.forget("forget me") is True
        assert mem.recall("forget me") is None

    def test_failed_query_not_remembered(self):
        mem = Memory()
        q = Query(text="Bad query")
        q.error = "Something broke"
        mem.remember(q)
        assert mem.recall("bad query") is None


class TestMemorySQLite:
    """Test Memory with SQLite backend."""

    @pytest.fixture
    def mem(self):
        from persistence.sqlite_store import SQLiteStore
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        store = SQLiteStore(path)
        m = Memory(store=store)
        yield m
        os.unlink(path)

    def test_remember_and_recall(self, mem):
        q = Query(text="What is Python?")
        q.answer = "A language."
        q.confidence = Confidence.HIGH
        mem.remember(q)
        assert mem.recall("what is python?") == "A language."

    def test_auto_promote_on_recall(self, mem):
        q = Query(text="Frequent Q")
        q.answer = "Frequent A"
        q.confidence = Confidence.HIGH
        mem.remember(q)
        for _ in range(5):
            mem.recall("frequent q")
        dump = mem.dump()
        assert len(dump["long_term"]) == 1

    def test_forget_sqlite(self, mem):
        q = Query(text="Delete me")
        q.answer = "Bye"
        q.confidence = Confidence.HIGH
        mem.remember(q)
        mem.forget("delete me")
        assert mem.recall("delete me") is None


# ------------------------------------------------------------------
# Orchestrator (integration)
# ------------------------------------------------------------------


class TestOrchestrator:
    def test_ask_builtin_knowledge(self):
        orch = Orchestrator(load_lm=False)
        result = orch.ask_sync("what is neural kernel")
        assert "neural kernel" in result.answer.lower() or "AI" in result.answer
        assert result.succeeded

    def test_ask_code_routes_to_specialist(self):
        orch = Orchestrator(load_lm=False)
        result = orch.ask_sync("def foo(): pass")
        assert "specialist:coding" in result.sources

    def test_ask_decorator_routes_to_specialist(self):
        orch = Orchestrator(load_lm=False)
        result = orch.ask_sync("@property")
        assert "specialist:coding" in result.sources

    def test_pipeline_steps_recorded(self):
        orch = Orchestrator(load_lm=False)
        result = orch.ask_sync("hello world")
        assert len(result.steps) >= 3  # classify, recall, reason at minimum

    def test_orchestrator_with_store(self):
        from persistence.sqlite_store import SQLiteStore
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        store = SQLiteStore(path)
        orch = Orchestrator(load_lm=False, store=store)
        result = orch.ask_sync("what is neural kernel")
        assert result.succeeded
        # Memory should be persisted.
        answer = orch.memory.recall("what is neural kernel")
        assert answer is not None
        os.unlink(path)
