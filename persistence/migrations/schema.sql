-- SQLite schema for neural_kernel persistence layer

-- Projects table
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    user_id TEXT,
    session_id TEXT,
    raw_text TEXT NOT NULL,
    project_name TEXT NOT NULL,
    summary TEXT NOT NULL,
    project_type TEXT NOT NULL,
    features_json TEXT NOT NULL,
    tech_stack_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    artifact_path TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Ownership columns (added in ownership/access-control migration)
-- SQLite ALTER TABLE ADD COLUMN is idempotent-safe with IF NOT EXISTS
-- workaround: we use a separate migration block.

-- Indexes for projects table
CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id);
CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status);
CREATE INDEX IF NOT EXISTS idx_projects_created_at ON projects(created_at);
CREATE INDEX IF NOT EXISTS idx_projects_session_id ON projects(session_id);

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT,
    messages TEXT NOT NULL,
    metadata TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Indexes for sessions table
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);

-- Artifacts table
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    format TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(project_id)
);

-- Indexes for artifacts table
CREATE INDEX IF NOT EXISTS idx_artifacts_project_id ON artifacts(project_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_created_at ON artifacts(created_at);

-- Request logs table
CREATE TABLE IF NOT EXISTS request_logs (
    log_id TEXT PRIMARY KEY,
    project_id TEXT,
    user_id TEXT,
    raw_text TEXT NOT NULL,
    parsed_features TEXT NOT NULL,
    parsed_tech_stack TEXT NOT NULL,
    processing_time_ms INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(project_id)
);

-- Indexes for request_logs table
CREATE INDEX IF NOT EXISTS idx_request_logs_project_id ON request_logs(project_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_user_id ON request_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_created_at ON request_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_request_logs_status ON request_logs(status);
