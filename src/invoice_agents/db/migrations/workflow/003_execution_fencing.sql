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
ALTER TABLE review_requests ADD COLUMN evidence_snapshot_digest TEXT;
ALTER TABLE final_decisions ADD COLUMN evidence_snapshot_digest TEXT;
ALTER TABLE payments ADD COLUMN evidence_snapshot_digest TEXT;

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
        AND substr(NEW.lease_expires_at, 1, 4) <> '0000'
        AND (
            (length(NEW.lease_expires_at) = 25 AND NEW.lease_expires_at GLOB
                '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]+00:00')
            OR (length(NEW.lease_expires_at) = 32 AND NEW.lease_expires_at GLOB
                '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]+00:00'
                AND substr(NEW.lease_expires_at, 21, 6) <> '000000')
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
        AND substr(NEW.lease_expires_at, 1, 4) <> '0000'
        AND (
            (length(NEW.lease_expires_at) = 25 AND NEW.lease_expires_at GLOB
                '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]+00:00')
            OR (length(NEW.lease_expires_at) = 32 AND NEW.lease_expires_at GLOB
                '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]+00:00'
                AND substr(NEW.lease_expires_at, 21, 6) <> '000000')
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

CREATE TRIGGER trg_review_requests_snapshot_digest_insert
BEFORE INSERT ON review_requests
WHEN NEW.evidence_snapshot_digest IS NULL
    OR length(NEW.evidence_snapshot_digest) <> 64
    OR NEW.evidence_snapshot_digest GLOB '*[^0-9a-f]*'
BEGIN
    SELECT RAISE(ABORT, 'EVIDENCE_SNAPSHOT_DIGEST_INVALID');
END;

CREATE TRIGGER trg_review_requests_snapshot_digest_update
BEFORE UPDATE OF evidence_snapshot_digest ON review_requests
WHEN NEW.evidence_snapshot_digest IS NULL
    OR length(NEW.evidence_snapshot_digest) <> 64
    OR NEW.evidence_snapshot_digest GLOB '*[^0-9a-f]*'
    OR NEW.evidence_snapshot_digest <> OLD.evidence_snapshot_digest
BEGIN
    SELECT RAISE(ABORT, 'EVIDENCE_SNAPSHOT_DIGEST_INVALID');
END;

CREATE TRIGGER trg_final_decisions_snapshot_digest_insert
BEFORE INSERT ON final_decisions
WHEN NEW.evidence_snapshot_digest IS NULL
    OR length(NEW.evidence_snapshot_digest) <> 64
    OR NEW.evidence_snapshot_digest GLOB '*[^0-9a-f]*'
    OR COALESCE((
        SELECT evidence_snapshot_digest FROM review_requests
        WHERE case_id = NEW.case_id AND execution_generation = NEW.decision_generation
        ORDER BY sequence DESC LIMIT 1
    ), NEW.evidence_snapshot_digest) <> NEW.evidence_snapshot_digest
BEGIN
    SELECT RAISE(ABORT, 'EVIDENCE_SNAPSHOT_DIGEST_INVALID');
END;

CREATE TRIGGER trg_final_decisions_snapshot_digest_update
BEFORE UPDATE OF evidence_snapshot_digest, decision_generation ON final_decisions
WHEN NEW.evidence_snapshot_digest IS NULL
    OR length(NEW.evidence_snapshot_digest) <> 64
    OR NEW.evidence_snapshot_digest GLOB '*[^0-9a-f]*'
    OR COALESCE((
        SELECT evidence_snapshot_digest FROM review_requests
        WHERE case_id = NEW.case_id AND execution_generation = NEW.decision_generation
        ORDER BY sequence DESC LIMIT 1
    ), NEW.evidence_snapshot_digest) <> NEW.evidence_snapshot_digest
BEGIN
    SELECT RAISE(ABORT, 'EVIDENCE_SNAPSHOT_DIGEST_INVALID');
END;

CREATE TRIGGER trg_payments_snapshot_digest_insert
BEFORE INSERT ON payments
WHEN NEW.evidence_snapshot_digest IS NULL
    OR length(NEW.evidence_snapshot_digest) <> 64
    OR NEW.evidence_snapshot_digest GLOB '*[^0-9a-f]*'
    OR NOT EXISTS (
        SELECT 1 FROM final_decisions
        WHERE case_id = NEW.case_id
            AND decision_generation = NEW.decision_generation
            AND evidence_snapshot_digest = NEW.evidence_snapshot_digest
    )
BEGIN
    SELECT RAISE(ABORT, 'EVIDENCE_SNAPSHOT_DIGEST_INVALID');
END;

CREATE TRIGGER trg_payments_snapshot_digest_update
BEFORE UPDATE OF evidence_snapshot_digest, decision_generation, case_id ON payments
WHEN NEW.evidence_snapshot_digest IS NULL
    OR length(NEW.evidence_snapshot_digest) <> 64
    OR NEW.evidence_snapshot_digest GLOB '*[^0-9a-f]*'
    OR NOT EXISTS (
        SELECT 1 FROM final_decisions
        WHERE case_id = NEW.case_id
            AND decision_generation = NEW.decision_generation
            AND evidence_snapshot_digest = NEW.evidence_snapshot_digest
    )
BEGIN
    SELECT RAISE(ABORT, 'EVIDENCE_SNAPSHOT_DIGEST_INVALID');
END;

CREATE TRIGGER trg_source_artifacts_immutable_after_reference_update
BEFORE UPDATE ON source_artifacts
WHEN EXISTS (SELECT 1 FROM cases WHERE source_id = OLD.source_id)
BEGIN
    SELECT RAISE(ABORT, 'SOURCE_ARTIFACT_IMMUTABLE');
END;

CREATE TRIGGER trg_source_artifacts_immutable_after_reference_delete
BEFORE DELETE ON source_artifacts
WHEN EXISTS (SELECT 1 FROM cases WHERE source_id = OLD.source_id)
BEGIN
    SELECT RAISE(ABORT, 'SOURCE_ARTIFACT_IMMUTABLE');
END;

CREATE TRIGGER trg_extractions_immutable_after_final_insert
BEFORE INSERT ON extractions
WHEN EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = NEW.case_id AND decision_generation = NEW.execution_generation
) OR EXISTS (SELECT 1 FROM payments WHERE case_id = NEW.case_id AND status = 'PAID')
BEGIN
    SELECT RAISE(ABORT, 'AUTHORIZATION_EVIDENCE_IMMUTABLE');
END;

CREATE TRIGGER trg_extractions_immutable_after_final_update
BEFORE UPDATE ON extractions
WHEN EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = OLD.case_id AND decision_generation = OLD.execution_generation
) OR EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = NEW.case_id AND decision_generation = NEW.execution_generation
) OR EXISTS (SELECT 1 FROM payments WHERE case_id IN (OLD.case_id, NEW.case_id) AND status = 'PAID')
BEGIN
    SELECT RAISE(ABORT, 'AUTHORIZATION_EVIDENCE_IMMUTABLE');
END;

CREATE TRIGGER trg_extractions_immutable_after_final_delete
BEFORE DELETE ON extractions
WHEN EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = OLD.case_id AND decision_generation = OLD.execution_generation
) OR EXISTS (SELECT 1 FROM payments WHERE case_id = OLD.case_id AND status = 'PAID')
BEGIN
    SELECT RAISE(ABORT, 'AUTHORIZATION_EVIDENCE_IMMUTABLE');
END;

CREATE TRIGGER trg_identity_results_immutable_after_final_insert
BEFORE INSERT ON identity_results
WHEN EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = NEW.case_id AND decision_generation = NEW.execution_generation
) OR EXISTS (SELECT 1 FROM payments WHERE case_id = NEW.case_id AND status = 'PAID')
BEGIN
    SELECT RAISE(ABORT, 'AUTHORIZATION_EVIDENCE_IMMUTABLE');
END;

CREATE TRIGGER trg_identity_results_immutable_after_final_update
BEFORE UPDATE ON identity_results
WHEN EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = OLD.case_id AND decision_generation = OLD.execution_generation
) OR EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = NEW.case_id AND decision_generation = NEW.execution_generation
) OR EXISTS (SELECT 1 FROM payments WHERE case_id IN (OLD.case_id, NEW.case_id) AND status = 'PAID')
BEGIN
    SELECT RAISE(ABORT, 'AUTHORIZATION_EVIDENCE_IMMUTABLE');
END;

CREATE TRIGGER trg_identity_results_immutable_after_final_delete
BEFORE DELETE ON identity_results
WHEN EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = OLD.case_id AND decision_generation = OLD.execution_generation
) OR EXISTS (SELECT 1 FROM payments WHERE case_id = OLD.case_id AND status = 'PAID')
BEGIN
    SELECT RAISE(ABORT, 'AUTHORIZATION_EVIDENCE_IMMUTABLE');
END;

CREATE TRIGGER trg_comparison_results_immutable_after_final_insert
BEFORE INSERT ON comparison_results
WHEN EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = NEW.case_id AND decision_generation = NEW.execution_generation
) OR EXISTS (SELECT 1 FROM payments WHERE case_id = NEW.case_id AND status = 'PAID')
BEGIN
    SELECT RAISE(ABORT, 'AUTHORIZATION_EVIDENCE_IMMUTABLE');
END;

CREATE TRIGGER trg_comparison_results_immutable_after_final_update
BEFORE UPDATE ON comparison_results
WHEN EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = OLD.case_id AND decision_generation = OLD.execution_generation
) OR EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = NEW.case_id AND decision_generation = NEW.execution_generation
) OR EXISTS (SELECT 1 FROM payments WHERE case_id IN (OLD.case_id, NEW.case_id) AND status = 'PAID')
BEGIN
    SELECT RAISE(ABORT, 'AUTHORIZATION_EVIDENCE_IMMUTABLE');
END;

CREATE TRIGGER trg_comparison_results_immutable_after_final_delete
BEFORE DELETE ON comparison_results
WHEN EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = OLD.case_id AND decision_generation = OLD.execution_generation
) OR EXISTS (SELECT 1 FROM payments WHERE case_id = OLD.case_id AND status = 'PAID')
BEGIN
    SELECT RAISE(ABORT, 'AUTHORIZATION_EVIDENCE_IMMUTABLE');
END;

CREATE TRIGGER trg_critique_results_immutable_after_final_insert
BEFORE INSERT ON critique_results
WHEN EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = NEW.case_id AND decision_generation = NEW.execution_generation
) OR EXISTS (SELECT 1 FROM payments WHERE case_id = NEW.case_id AND status = 'PAID')
BEGIN
    SELECT RAISE(ABORT, 'AUTHORIZATION_EVIDENCE_IMMUTABLE');
END;

CREATE TRIGGER trg_critique_results_immutable_after_final_update
BEFORE UPDATE ON critique_results
WHEN EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = OLD.case_id AND decision_generation = OLD.execution_generation
) OR EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = NEW.case_id AND decision_generation = NEW.execution_generation
) OR EXISTS (SELECT 1 FROM payments WHERE case_id IN (OLD.case_id, NEW.case_id) AND status = 'PAID')
BEGIN
    SELECT RAISE(ABORT, 'AUTHORIZATION_EVIDENCE_IMMUTABLE');
END;

CREATE TRIGGER trg_critique_results_immutable_after_final_delete
BEFORE DELETE ON critique_results
WHEN EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = OLD.case_id AND decision_generation = OLD.execution_generation
) OR EXISTS (SELECT 1 FROM payments WHERE case_id = OLD.case_id AND status = 'PAID')
BEGIN
    SELECT RAISE(ABORT, 'AUTHORIZATION_EVIDENCE_IMMUTABLE');
END;

CREATE TRIGGER trg_review_requests_immutable_after_final_insert
BEFORE INSERT ON review_requests
WHEN EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = NEW.case_id AND decision_generation = NEW.execution_generation
) OR EXISTS (SELECT 1 FROM payments WHERE case_id = NEW.case_id AND status = 'PAID')
BEGIN
    SELECT RAISE(ABORT, 'AUTHORIZATION_EVIDENCE_IMMUTABLE');
END;

CREATE TRIGGER trg_review_requests_immutable_after_final_update
BEFORE UPDATE ON review_requests
WHEN EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = OLD.case_id AND decision_generation = OLD.execution_generation
) OR EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = NEW.case_id AND decision_generation = NEW.execution_generation
) OR EXISTS (SELECT 1 FROM payments WHERE case_id IN (OLD.case_id, NEW.case_id) AND status = 'PAID')
BEGIN
    SELECT RAISE(ABORT, 'AUTHORIZATION_EVIDENCE_IMMUTABLE');
END;

CREATE TRIGGER trg_review_requests_immutable_after_final_delete
BEFORE DELETE ON review_requests
WHEN EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = OLD.case_id AND decision_generation = OLD.execution_generation
) OR EXISTS (SELECT 1 FROM payments WHERE case_id = OLD.case_id AND status = 'PAID')
BEGIN
    SELECT RAISE(ABORT, 'AUTHORIZATION_EVIDENCE_IMMUTABLE');
END;

CREATE TRIGGER trg_paid_payments_immutable_update
BEFORE UPDATE ON payments
WHEN OLD.status = 'PAID'
BEGIN
    SELECT RAISE(ABORT, 'PAID_PAYMENT_IMMUTABLE');
END;

CREATE TRIGGER trg_paid_payments_immutable_delete
BEFORE DELETE ON payments
WHEN OLD.status = 'PAID'
BEGIN
    SELECT RAISE(ABORT, 'PAID_PAYMENT_IMMUTABLE');
END;

CREATE TRIGGER trg_resolved_review_immutable_update
BEFORE UPDATE ON review_requests
WHEN OLD.status = 'RESOLVED' AND NOT (
    NEW.review_id = OLD.review_id
    AND NEW.case_id = OLD.case_id
    AND NEW.sequence = OLD.sequence
    AND NEW.status = OLD.status
    AND NEW.payload_json = OLD.payload_json
    AND NEW.created_at = OLD.created_at
    AND NEW.resolved_at IS OLD.resolved_at
    AND NEW.execution_generation = OLD.execution_generation + 1
    AND NEW.evidence_snapshot_digest = OLD.evidence_snapshot_digest
)
BEGIN
    SELECT RAISE(ABORT, 'RESOLVED_REVIEW_IMMUTABLE');
END;

CREATE TRIGGER trg_resolved_review_immutable_delete
BEFORE DELETE ON review_requests
WHEN OLD.status = 'RESOLVED'
BEGIN
    SELECT RAISE(ABORT, 'RESOLVED_REVIEW_IMMUTABLE');
END;

CREATE TRIGGER trg_human_decisions_immutable_insert
BEFORE INSERT ON human_decisions
WHEN EXISTS (
    SELECT 1 FROM review_requests r JOIN final_decisions f ON f.case_id = r.case_id
    WHERE r.review_id = NEW.review_id AND f.decision_generation = r.execution_generation
) OR EXISTS (
    SELECT 1 FROM review_requests r JOIN payments p ON p.case_id = r.case_id
    WHERE r.review_id = NEW.review_id AND p.status = 'PAID'
)
BEGIN
    SELECT RAISE(ABORT, 'HUMAN_DECISION_IMMUTABLE');
END;

CREATE TRIGGER trg_human_decisions_immutable_update
BEFORE UPDATE ON human_decisions
BEGIN
    SELECT RAISE(ABORT, 'HUMAN_DECISION_IMMUTABLE');
END;

CREATE TRIGGER trg_human_decisions_immutable_delete
BEFORE DELETE ON human_decisions
BEGIN
    SELECT RAISE(ABORT, 'HUMAN_DECISION_IMMUTABLE');
END;
