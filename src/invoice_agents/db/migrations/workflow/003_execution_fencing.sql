ALTER TABLE cases ADD COLUMN execution_token TEXT;
ALTER TABLE cases ADD COLUMN execution_generation INTEGER NOT NULL DEFAULT 0;
ALTER TABLE cases ADD COLUMN execution_state TEXT NOT NULL DEFAULT 'IDLE';
ALTER TABLE cases ADD COLUMN lease_expires_at TEXT;

ALTER TABLE final_decisions ADD COLUMN decision_generation INTEGER NOT NULL DEFAULT 0;
ALTER TABLE payments ADD COLUMN decision_generation INTEGER NOT NULL DEFAULT 0;
ALTER TABLE extractions ADD COLUMN execution_generation INTEGER NOT NULL DEFAULT 0;
ALTER TABLE identity_results ADD COLUMN execution_generation INTEGER NOT NULL DEFAULT 0;
ALTER TABLE comparison_results ADD COLUMN execution_generation INTEGER NOT NULL DEFAULT 0;
ALTER TABLE critique_results ADD COLUMN execution_generation INTEGER NOT NULL DEFAULT 0;
ALTER TABLE review_requests ADD COLUMN execution_generation INTEGER NOT NULL DEFAULT 0;

CREATE INDEX idx_cases_execution_lease
ON cases(execution_state, lease_expires_at);

CREATE TRIGGER trg_final_decisions_no_insert_after_paid
BEFORE INSERT ON final_decisions
WHEN EXISTS (
    SELECT 1 FROM payments
    WHERE payments.case_id = NEW.case_id AND payments.status = 'PAID'
)
BEGIN
    SELECT RAISE(ABORT, 'PAID_FINAL_DECISION_IMMUTABLE');
END;

CREATE TRIGGER trg_final_decisions_no_update_after_paid
BEFORE UPDATE ON final_decisions
WHEN EXISTS (
    SELECT 1 FROM payments
    WHERE payments.case_id = OLD.case_id AND payments.status = 'PAID'
)
BEGIN
    SELECT RAISE(ABORT, 'PAID_FINAL_DECISION_IMMUTABLE');
END;

CREATE TRIGGER trg_final_decisions_no_delete_after_paid
BEFORE DELETE ON final_decisions
WHEN EXISTS (
    SELECT 1 FROM payments
    WHERE payments.case_id = OLD.case_id AND payments.status = 'PAID'
)
BEGIN
    SELECT RAISE(ABORT, 'PAID_FINAL_DECISION_IMMUTABLE');
END;

CREATE TRIGGER trg_cases_execution_authority_insert
BEFORE INSERT ON cases
WHEN NOT (
    (NEW.execution_state = 'IDLE' AND NEW.execution_token IS NULL
        AND NEW.lease_expires_at IS NULL
        AND typeof(NEW.execution_generation) = 'integer'
        AND NEW.execution_generation >= 0)
    OR (NEW.execution_state = 'RUNNING' AND NEW.execution_token IS NOT NULL
        AND NEW.execution_token <> '' AND NEW.lease_expires_at IS NOT NULL
        AND (
            (length(NEW.lease_expires_at) = 25 AND NEW.lease_expires_at GLOB
                '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]+00:00')
            OR (length(NEW.lease_expires_at) = 32 AND NEW.lease_expires_at GLOB
                '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]+00:00')
        )
        AND datetime(NEW.lease_expires_at) IS NOT NULL
        AND strftime('%Y-%m-%dT%H:%M:%S', NEW.lease_expires_at) =
            substr(NEW.lease_expires_at, 1, 19)
        AND CAST(substr(NEW.lease_expires_at, 12, 2) AS INTEGER) BETWEEN 0 AND 23
        AND CAST(substr(NEW.lease_expires_at, 15, 2) AS INTEGER) BETWEEN 0 AND 59
        AND CAST(substr(NEW.lease_expires_at, 18, 2) AS INTEGER) BETWEEN 0 AND 59
        AND typeof(NEW.execution_generation) = 'integer'
        AND NEW.execution_generation >= 1)
    OR (NEW.execution_state = 'FINISHED' AND NEW.execution_token IS NOT NULL
        AND NEW.execution_token <> '' AND NEW.lease_expires_at IS NULL
        AND typeof(NEW.execution_generation) = 'integer'
        AND NEW.execution_generation >= 1)
)
BEGIN
    SELECT RAISE(ABORT, 'INVALID_EXECUTION_AUTHORITY');
END;

CREATE TRIGGER trg_cases_execution_authority_update
BEFORE UPDATE OF execution_token, execution_generation, execution_state, lease_expires_at
ON cases
WHEN NOT (
    (NEW.execution_state = 'IDLE' AND NEW.execution_token IS NULL
        AND NEW.lease_expires_at IS NULL
        AND typeof(NEW.execution_generation) = 'integer'
        AND NEW.execution_generation >= 0)
    OR (NEW.execution_state = 'RUNNING' AND NEW.execution_token IS NOT NULL
        AND NEW.execution_token <> '' AND NEW.lease_expires_at IS NOT NULL
        AND (
            (length(NEW.lease_expires_at) = 25 AND NEW.lease_expires_at GLOB
                '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]+00:00')
            OR (length(NEW.lease_expires_at) = 32 AND NEW.lease_expires_at GLOB
                '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]+00:00')
        )
        AND datetime(NEW.lease_expires_at) IS NOT NULL
        AND strftime('%Y-%m-%dT%H:%M:%S', NEW.lease_expires_at) =
            substr(NEW.lease_expires_at, 1, 19)
        AND CAST(substr(NEW.lease_expires_at, 12, 2) AS INTEGER) BETWEEN 0 AND 23
        AND CAST(substr(NEW.lease_expires_at, 15, 2) AS INTEGER) BETWEEN 0 AND 59
        AND CAST(substr(NEW.lease_expires_at, 18, 2) AS INTEGER) BETWEEN 0 AND 59
        AND typeof(NEW.execution_generation) = 'integer'
        AND NEW.execution_generation >= 1)
    OR (NEW.execution_state = 'FINISHED' AND NEW.execution_token IS NOT NULL
        AND NEW.execution_token <> '' AND NEW.lease_expires_at IS NULL
        AND typeof(NEW.execution_generation) = 'integer'
        AND NEW.execution_generation >= 1)
)
BEGIN
    SELECT RAISE(ABORT, 'INVALID_EXECUTION_AUTHORITY');
END;
