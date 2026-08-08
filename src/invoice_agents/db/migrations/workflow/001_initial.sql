CREATE TABLE source_artifacts (
    source_id TEXT PRIMARY KEY,
    canonical_path TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    source_format TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    modified_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_source_artifacts_hash ON source_artifacts(source_hash);

CREATE TABLE cases (
    case_id TEXT PRIMARY KEY,
    source_id TEXT REFERENCES source_artifacts(source_id),
    invoice_number TEXT,
    vendor TEXT,
    revision TEXT,
    status TEXT NOT NULL,
    stop_reason TEXT,
    result_json TEXT,
    team_state_json TEXT,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE INDEX idx_cases_invoice_vendor ON cases(invoice_number, vendor);
CREATE INDEX idx_cases_source_id ON cases(source_id);

CREATE TABLE extractions (
    extraction_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    version INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(case_id, version)
);

CREATE TABLE identity_results (
    identity_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE comparison_results (
    comparison_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    comparison_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE critique_results (
    critique_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE review_requests (
    review_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL UNIQUE REFERENCES cases(case_id),
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE TABLE human_decisions (
    decision_id TEXT PRIMARY KEY,
    review_id TEXT NOT NULL UNIQUE REFERENCES review_requests(review_id),
    reviewer TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    decided_at TEXT NOT NULL
);

CREATE TABLE final_decisions (
    decision_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL UNIQUE REFERENCES cases(case_id),
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE payments (
    payment_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    idempotency_key TEXT NOT NULL UNIQUE,
    vendor TEXT NOT NULL,
    amount TEXT NOT NULL,
    currency TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_payments_case_id ON payments(case_id);

CREATE TABLE events (
    event_id TEXT PRIMARY KEY,
    case_id TEXT,
    source_id TEXT,
    event_type TEXT NOT NULL,
    agent_name TEXT,
    tool_call_id TEXT,
    db_evidence_id TEXT,
    review_id TEXT,
    payment_id TEXT,
    provider_request_id TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_events_case_created ON events(case_id, created_at);

