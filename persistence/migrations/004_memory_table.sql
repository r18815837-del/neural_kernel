-- Migration 004: Add memory table for cognition layer persistence
-- Replaces in-memory dicts in cognition/memory.py

CREATE TABLE IF NOT EXISTS memory (
    key TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    answer TEXT NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0,
    tier TEXT NOT NULL DEFAULT 'short',  -- 'short' or 'long'
    source_query_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_tier ON memory(tier);
CREATE INDEX IF NOT EXISTS idx_memory_hits ON memory(hits);
CREATE INDEX IF NOT EXISTS idx_memory_updated_at ON memory(updated_at);
