CREATE TABLE IF NOT EXISTS legacy_authorization_reconciliations (
    reconciliation_id TEXT PRIMARY KEY,
    reviewer TEXT NOT NULL,
    reason TEXT NOT NULL,
    disposition TEXT NOT NULL CHECK(disposition = 'PERMANENTLY_NON_AUTHORIZING'),
    confirmed_at TEXT NOT NULL,
    source_schema_version INTEGER NOT NULL CHECK(source_schema_version IN (1, 2)),
    record_count INTEGER NOT NULL CHECK(record_count > 0),
    schema_manifest_hash TEXT NOT NULL,
    record_manifest_hash TEXT NOT NULL,
    table_counts_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state = 'COMPLETED')
);

CREATE TABLE IF NOT EXISTS legacy_authorization_table_manifests (
    manifest_id TEXT PRIMARY KEY,
    reconciliation_id TEXT NOT NULL REFERENCES legacy_authorization_reconciliations(
        reconciliation_id
    ) DEFERRABLE INITIALLY DEFERRED,
    source_table_order INTEGER NOT NULL CHECK(source_table_order >= 0),
    source_table TEXT NOT NULL CHECK(source_table IN (
        'review_requests', 'human_decisions', 'final_decisions', 'payments'
    )),
    source_table_sql BLOB NOT NULL,
    column_manifest BLOB NOT NULL,
    schema_hash TEXT NOT NULL,
    original_row_count INTEGER NOT NULL CHECK(original_row_count >= 0),
    UNIQUE(reconciliation_id, source_table_order),
    UNIQUE(reconciliation_id, source_table)
);

CREATE TABLE IF NOT EXISTS legacy_authorization_quarantine (
    archive_id TEXT PRIMARY KEY,
    reconciliation_id TEXT NOT NULL REFERENCES legacy_authorization_reconciliations(
        reconciliation_id
    ) DEFERRABLE INITIALLY DEFERRED,
    source_table TEXT NOT NULL CHECK(source_table IN (
        'review_requests', 'human_decisions', 'final_decisions', 'payments'
    )),
    source_record_key TEXT NOT NULL,
    source_row_ordinal INTEGER NOT NULL CHECK(source_row_ordinal >= 0),
    source_rowid INTEGER,
    original_row_json TEXT,
    typed_row BLOB NOT NULL,
    schema_hash TEXT NOT NULL,
    record_hash TEXT NOT NULL,
    authorization_state TEXT NOT NULL CHECK(
        authorization_state = 'PERMANENTLY_NON_AUTHORIZING'
    ),
    archived_at TEXT NOT NULL,
    UNIQUE(reconciliation_id, source_table, source_row_ordinal)
);

CREATE INDEX IF NOT EXISTS idx_legacy_authorization_quarantine_reconciliation
ON legacy_authorization_quarantine(reconciliation_id);

CREATE TRIGGER IF NOT EXISTS trg_legacy_authorization_reconciliations_immutable_update
BEFORE UPDATE ON legacy_authorization_reconciliations
BEGIN
    SELECT RAISE(ABORT, 'LEGACY_AUTHORIZATION_ARCHIVE_IMMUTABLE');
END;

CREATE TRIGGER IF NOT EXISTS trg_legacy_authorization_reconciliations_immutable_delete
BEFORE DELETE ON legacy_authorization_reconciliations
BEGIN
    SELECT RAISE(ABORT, 'LEGACY_AUTHORIZATION_ARCHIVE_IMMUTABLE');
END;

CREATE TRIGGER IF NOT EXISTS trg_legacy_authorization_table_manifests_immutable_update
BEFORE UPDATE ON legacy_authorization_table_manifests
BEGIN
    SELECT RAISE(ABORT, 'LEGACY_AUTHORIZATION_ARCHIVE_IMMUTABLE');
END;

CREATE TRIGGER IF NOT EXISTS trg_legacy_authorization_table_manifests_immutable_delete
BEFORE DELETE ON legacy_authorization_table_manifests
BEGIN
    SELECT RAISE(ABORT, 'LEGACY_AUTHORIZATION_ARCHIVE_IMMUTABLE');
END;

CREATE TRIGGER IF NOT EXISTS trg_legacy_authorization_quarantine_immutable_update
BEFORE UPDATE ON legacy_authorization_quarantine
BEGIN
    SELECT RAISE(ABORT, 'LEGACY_AUTHORIZATION_ARCHIVE_IMMUTABLE');
END;

CREATE TRIGGER IF NOT EXISTS trg_legacy_authorization_quarantine_immutable_delete
BEFORE DELETE ON legacy_authorization_quarantine
BEGIN
    SELECT RAISE(ABORT, 'LEGACY_AUTHORIZATION_ARCHIVE_IMMUTABLE');
END;
