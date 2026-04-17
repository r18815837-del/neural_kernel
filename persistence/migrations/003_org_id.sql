-- Migration 003: Add org_id column for multi-tenant ownership.

ALTER TABLE projects ADD COLUMN org_id TEXT;
ALTER TABLE artifacts ADD COLUMN org_id TEXT;

CREATE INDEX IF NOT EXISTS idx_projects_org_id ON projects(org_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_org_id ON artifacts(org_id);
