"""Tests for execution_validation.executor — safe subprocess wrapper."""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from artifacts.execution_validation.executor import CommandResult, run_command


def test_run_command_success():
    with tempfile.TemporaryDirectory() as d:
        r = run_command(["echo", "hello"], cwd=d)
        assert r.exit_code == 0
        assert "hello" in r.stdout
        assert r.timed_out is False
        assert r.duration_ms >= 0


def test_run_command_failure():
    with tempfile.TemporaryDirectory() as d:
        r = run_command(["false"], cwd=d)
        assert r.exit_code != 0
        assert r.timed_out is False


def test_run_command_timeout():
    with tempfile.TemporaryDirectory() as d:
        r = run_command(["sleep", "60"], cwd=d, timeout_seconds=1)
        assert r.timed_out is True
        assert r.exit_code == -1


def test_run_command_not_found():
    with tempfile.TemporaryDirectory() as d:
        r = run_command(["nonexistent_binary_xyz_123"], cwd=d)
        assert r.exit_code == -1
        assert "not found" in r.stderr.lower() or "Command not found" in r.stderr


def test_run_command_captures_stderr():
    with tempfile.TemporaryDirectory() as d:
        r = run_command([sys.executable, "-c", "import sys; sys.stderr.write('err\\n'); sys.exit(1)"], cwd=d)
        assert r.exit_code == 1
        assert "err" in r.stderr


def test_run_command_env():
    with tempfile.TemporaryDirectory() as d:
        r = run_command(
            [sys.executable, "-c", "import os; print(os.environ.get('TEST_VAR', ''))"],
            cwd=d,
            env={**os.environ, "TEST_VAR": "hello_val"},
        )
        assert "hello_val" in r.stdout


def test_command_result_to_dict():
    r = CommandResult(command=["echo"], exit_code=0, stdout="hi", stderr="", timed_out=False, duration_ms=10)
    d = r.to_dict()
    assert d["exit_code"] == 0
    assert d["command"] == ["echo"]
    assert d["timed_out"] is False


def test_run_command_truncates_large_output():
    """Output > 32KB should be truncated."""
    with tempfile.TemporaryDirectory() as d:
        r = run_command(
            [sys.executable, "-c", "print('x' * 100000)"],
            cwd=d,
        )
        # Should have been truncated
        assert len(r.stdout) <= 40000  # some margin


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
