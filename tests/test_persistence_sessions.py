"""Tests for persistence layer — sessions, memory, messages."""

import json
import os
import tempfile

import pytest

from persistence.sqlite_store import SQLiteStore
from persistence.models import StoredSession


@pytest.fixture
def store():
    """Create a temporary SQLite store for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = SQLiteStore(path)
    yield s
    os.unlink(path)


@pytest.fixture
def populated_store(store):
    """Store with 5 sessions pre-populated."""
    for i in range(5):
        s = StoredSession(
            session_id=f"sess-{i}",
            user_id="user-A" if i < 3 else "user-B",
        )
        store.save_session(s)
        store.append_message(f"sess-{i}", "user", f"Hello {i}")
        store.append_message(f"sess-{i}", "assistant", f"Reply {i}")
    return store


class TestSessionCRUD:
    def test_create_session(self, store):
        s = StoredSession(session_id="s1", user_id="u1")
        sid = store.save_session(s)
        assert sid == "s1"
        assert store.get_session("s1") is not None

    def test_get_nonexistent_session(self, store):
        assert store.get_session("nonexistent") is None

    def test_update_session(self, store):
        s = StoredSession(session_id="s1", user_id="u1", metadata={"v": 1})
        store.save_session(s)
        s.metadata = {"v": 2}
        store.save_session(s)
        retrieved = store.get_session("s1")
        assert retrieved.metadata == {"v": 2}

    def test_delete_session(self, store):
        store.save_session(StoredSession(session_id="s1", user_id="u1"))
        assert store.delete_session("s1") is True
        assert store.get_session("s1") is None

    def test_delete_nonexistent_session(self, store):
        assert store.delete_session("nope") is False


class TestSessionMessages:
    def test_append_message_with_timestamp(self, store):
        store.save_session(StoredSession(session_id="s1", user_id="u1"))
        store.append_message("s1", "user", "Hello")
        msgs = store.get_session_messages("s1")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"
        assert "timestamp" in msgs[0]

    def test_append_multiple_messages(self, store):
        store.save_session(StoredSession(session_id="s1", user_id="u1"))
        for i in range(10):
            store.append_message("s1", "user", f"msg-{i}")
        msgs = store.get_session_messages("s1")
        assert len(msgs) == 10

    def test_message_pagination(self, store):
        store.save_session(StoredSession(session_id="s1", user_id="u1"))
        for i in range(20):
            store.append_message("s1", "user", f"msg-{i}")
        page = store.get_session_messages("s1", limit=5, offset=10)
        assert len(page) == 5
        assert page[0]["content"] == "msg-10"

    def test_messages_for_nonexistent_session(self, store):
        assert store.get_session_messages("nope") == []


class TestListSessions:
    def test_list_all(self, populated_store):
        sessions = populated_store.list_sessions()
        assert len(sessions) == 5

    def test_list_by_user(self, populated_store):
        a = populated_store.list_sessions(user_id="user-A")
        b = populated_store.list_sessions(user_id="user-B")
        assert len(a) == 3
        assert len(b) == 2

    def test_list_pagination(self, populated_store):
        page1 = populated_store.list_sessions(limit=2, offset=0)
        page2 = populated_store.list_sessions(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].session_id != page2[0].session_id

    def test_list_empty(self, store):
        assert store.list_sessions() == []


class TestMemoryPersistence:
    def test_save_and_get_memory(self, store):
        store.save_memory_entry(
            key="what is python",
            answer="A programming language",
            hits=0,
            tier="short",
            query_text="What is Python?",
        )
        entry = store.get_memory_entry("what is python")
        assert entry is not None
        assert entry["answer"] == "A programming language"
        assert entry["tier"] == "short"
        assert entry["hits"] == 0

    def test_get_nonexistent_memory(self, store):
        assert store.get_memory_entry("nope") is None

    def test_update_memory_entry(self, store):
        store.save_memory_entry("k1", "answer1", hits=0, tier="short")
        store.save_memory_entry("k1", "answer1", hits=3, tier="long")
        entry = store.get_memory_entry("k1")
        assert entry["tier"] == "long"
        assert entry["hits"] == 3

    def test_list_memory_by_tier(self, store):
        store.save_memory_entry("k1", "a1", tier="short")
        store.save_memory_entry("k2", "a2", tier="short")
        store.save_memory_entry("k3", "a3", tier="long")
        assert len(store.list_memory(tier="short")) == 2
        assert len(store.list_memory(tier="long")) == 1
        assert len(store.list_memory()) == 3

    def test_delete_memory(self, store):
        store.save_memory_entry("k1", "a1")
        assert store.delete_memory_entry("k1") is True
        assert store.get_memory_entry("k1") is None
        assert store.delete_memory_entry("k1") is False
