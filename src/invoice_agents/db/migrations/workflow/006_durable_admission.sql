CREATE TABLE submission_requests (
    request_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('single', 'batch')),
    fingerprint TEXT NOT NULL CHECK(
        length(fingerprint) = 64
        AND fingerprint NOT GLOB '*[^0-9a-f]*'
    ),
    redirect_target TEXT NOT NULL
);

CREATE TABLE source_run_claims (
    source_id TEXT PRIMARY KEY REFERENCES source_artifacts(source_id),
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    state TEXT NOT NULL CHECK(state IN ('queued', 'running', 'done', 'failed')),
    claimed_at TEXT NOT NULL,
    released_at TEXT
);

CREATE INDEX idx_source_run_claims_case_id
ON source_run_claims(case_id);

CREATE TABLE batches (
    batch_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    concurrency INTEGER NOT NULL CHECK(concurrency BETWEEN 1 AND 8),
    state TEXT NOT NULL CHECK(state IN ('queued', 'running', 'done', 'failed'))
);

CREATE INDEX idx_batches_state ON batches(state);

CREATE TABLE batch_entries (
    batch_id TEXT NOT NULL REFERENCES batches(batch_id),
    position INTEGER NOT NULL CHECK(position >= 0),
    source_id TEXT NOT NULL REFERENCES source_artifacts(source_id),
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    source_path TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('queued', 'running', 'done', 'failed')),
    PRIMARY KEY(batch_id, position),
    UNIQUE(batch_id, source_id)
);

CREATE INDEX idx_batch_entries_case_id
ON batch_entries(case_id);
