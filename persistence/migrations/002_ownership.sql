-- Migration 002: Add ownership columns to projects and artifacts.
-- These columns are nullable so pre-existing rows remain valid.
-- SQLite doesn't support IF NOT EXISTS on ALTER TABLE ADD COLUMN,
-- so the Python migration runner catches "duplicate column" errors.

ALTER TABLE projects ADD COLUMN owner_client_id TEXT;
ALTER TABLE projects ADD COLUMN owner_user_id TEXT;

ALTER TABLE artifacts ADD COLUMN owner_client_id TEXT;
ALTER TABLE artifacts ADD COLUMN owner_user_id TEXT;

-- Ownership indexes
CREATE INDEX IF NOT EXISTS idx_projects_owner_client_id ON projects(owner_client_id);
CREATE INDEX IF NOT EXISTS idx_projects_owner_user_id ON projects(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_owner_client_id ON artifacts(owner_client_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_owner_user_id ON artifacts(owner_user_id);
