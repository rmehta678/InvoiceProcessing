ALTER TABLE cases ADD COLUMN execution_token TEXT;
ALTER TABLE cases ADD COLUMN execution_generation INTEGER NOT NULL DEFAULT 0;
ALTER TABLE cases ADD COLUMN execution_state TEXT NOT NULL DEFAULT 'IDLE';
ALTER TABLE cases ADD COLUMN lease_expires_at TEXT;

ALTER TABLE final_decisions ADD COLUMN decision_generation INTEGER NOT NULL DEFAULT 0;
ALTER TABLE payments ADD COLUMN decision_generation INTEGER NOT NULL DEFAULT 0;
ALTER TABLE extractions ADD COLUMN execution_generation INTEGER NOT NULL DEFAULT 0;
ALTER TABLE identity_results ADD COLUMN execution_generation INTEGER NOT NULL DEFAULT 0;
ALTER TABLE identity_results ADD COLUMN evaluated_at TEXT;
ALTER TABLE comparison_results ADD COLUMN execution_generation INTEGER NOT NULL DEFAULT 0;
ALTER TABLE critique_results ADD COLUMN execution_generation INTEGER NOT NULL DEFAULT 0;
ALTER TABLE review_requests ADD COLUMN execution_generation INTEGER NOT NULL DEFAULT 0;
ALTER TABLE review_requests ADD COLUMN evidence_snapshot_digest TEXT;
ALTER TABLE final_decisions ADD COLUMN evidence_snapshot_digest TEXT;
ALTER TABLE payments ADD COLUMN evidence_snapshot_digest TEXT;
ALTER TABLE final_decisions ADD COLUMN source_id TEXT;
ALTER TABLE final_decisions ADD COLUMN invoice_number TEXT;
ALTER TABLE final_decisions ADD COLUMN vendor TEXT;
ALTER TABLE final_decisions ADD COLUMN authorized_amount TEXT;
ALTER TABLE final_decisions ADD COLUMN authorized_currency TEXT;
ALTER TABLE final_decisions ADD COLUMN payment_idempotency_key TEXT;
ALTER TABLE final_decisions ADD COLUMN review_id TEXT;
ALTER TABLE payments ADD COLUMN source_id TEXT;
ALTER TABLE payments ADD COLUMN invoice_number TEXT;
ALTER TABLE payments ADD COLUMN review_id TEXT;

UPDATE identity_results SET evaluated_at = created_at WHERE evaluated_at IS NULL;

CREATE INDEX idx_cases_execution_lease
ON cases(execution_state, lease_expires_at);

CREATE TABLE validated_evidence_snapshots (
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    execution_generation INTEGER NOT NULL,
    evidence_snapshot_digest TEXT NOT NULL,
    policy_review_required INTEGER NOT NULL,
    unresolved_blocker_count INTEGER NOT NULL,
    critique_disposition TEXT NOT NULL,
    review_id TEXT REFERENCES review_requests(review_id),
    review_snapshot_digest TEXT,
    validated_at TEXT NOT NULL,
    PRIMARY KEY(case_id, execution_generation),
    UNIQUE(case_id, execution_generation, evidence_snapshot_digest)
);

CREATE TRIGGER trg_validated_evidence_snapshots_insert
BEFORE INSERT ON validated_evidence_snapshots
WHEN NEW.execution_generation < 1
    OR NEW.evidence_snapshot_digest IS NULL
    OR length(NEW.evidence_snapshot_digest) <> 64
    OR NEW.evidence_snapshot_digest GLOB '*[^0-9a-f]*'
    OR typeof(NEW.policy_review_required) <> 'integer'
    OR NEW.policy_review_required NOT IN (0, 1)
    OR typeof(NEW.unresolved_blocker_count) <> 'integer'
    OR NEW.unresolved_blocker_count < 0
    OR NEW.critique_disposition NOT IN ('APPROVE', 'REJECT', 'HOLD', 'FAILED')
    OR NOT EXISTS (
        SELECT 1
        FROM cases c
        JOIN extractions e ON e.case_id = c.case_id
        JOIN identity_results i ON i.case_id = c.case_id
        JOIN comparison_results inventory ON inventory.case_id = c.case_id
            AND inventory.comparison_type = 'inventory'
        JOIN comparison_results risk ON risk.case_id = c.case_id
            AND risk.comparison_type = 'risk'
        JOIN critique_results critique ON critique.case_id = c.case_id
        WHERE c.case_id = NEW.case_id
            AND c.execution_generation = NEW.execution_generation
            AND e.execution_generation = NEW.execution_generation
            AND i.execution_generation = NEW.execution_generation
            AND inventory.execution_generation = NEW.execution_generation
            AND risk.execution_generation = NEW.execution_generation
            AND critique.execution_generation = NEW.execution_generation
            AND e.version = (
                SELECT MAX(latest.version) FROM extractions latest
                WHERE latest.case_id = NEW.case_id
                    AND latest.execution_generation = NEW.execution_generation
            )
            AND i.rowid = (
                SELECT MAX(latest.rowid) FROM identity_results latest
                WHERE latest.case_id = NEW.case_id
                    AND latest.execution_generation = NEW.execution_generation
            )
            AND inventory.rowid = (
                SELECT MAX(latest.rowid) FROM comparison_results latest
                WHERE latest.case_id = NEW.case_id
                    AND latest.execution_generation = NEW.execution_generation
                    AND latest.comparison_type = 'inventory'
            )
            AND risk.rowid = (
                SELECT MAX(latest.rowid) FROM comparison_results latest
                WHERE latest.case_id = NEW.case_id
                    AND latest.execution_generation = NEW.execution_generation
                    AND latest.comparison_type = 'risk'
            )
            AND critique.rowid = (
                SELECT MAX(latest.rowid) FROM critique_results latest
                WHERE latest.case_id = NEW.case_id
                    AND latest.execution_generation = NEW.execution_generation
            )
            AND NEW.evidence_snapshot_digest = stored_evidence_snapshot_digest(
                NEW.case_id,
                e.payload_json,
                i.payload_json,
                i.evaluated_at,
                inventory.payload_json,
                risk.payload_json,
                critique.payload_json
            )
            AND NEW.policy_review_required = CASE
                WHEN json_array_length(json_extract(
                    risk.payload_json, '$.policy_review_reasons'
                )) > 0 THEN 1 ELSE 0 END
            AND NEW.unresolved_blocker_count = stored_unresolved_blocker_count(
                risk.payload_json,
                (
                    SELECT h.payload_json FROM human_decisions h
                    WHERE h.review_id = NEW.review_id
                )
            )
            AND NEW.critique_disposition = json_extract(
                critique.payload_json, '$.recommended_disposition'
            )
    )
    OR NOT (
        (
            NEW.review_id IS NULL
            AND NEW.review_snapshot_digest IS NULL
            AND NEW.policy_review_required = 0
            AND NOT EXISTS (
                SELECT 1 FROM review_requests
                WHERE case_id = NEW.case_id
                    AND execution_generation = NEW.execution_generation
            )
        )
        OR EXISTS (
            SELECT 1
            FROM review_requests r
            JOIN human_decisions h ON h.review_id = r.review_id
            WHERE r.review_id = NEW.review_id
                AND r.case_id = NEW.case_id
                AND r.execution_generation = NEW.execution_generation
                AND r.sequence = (
                    SELECT MAX(latest.sequence) FROM review_requests latest
                    WHERE latest.case_id = NEW.case_id
                )
                AND r.status = 'RESOLVED'
                AND json_valid(r.payload_json) = 1
                AND json_extract(r.payload_json, '$.review_id') = r.review_id
                AND json_extract(r.payload_json, '$.case_id') = r.case_id
                AND json_extract(r.payload_json, '$.sequence') = r.sequence
                AND json_extract(r.payload_json, '$.status') = r.status
                AND json_extract(r.payload_json, '$.human_decision.review_id') = r.review_id
                AND r.resolved_at = h.decided_at
                AND r.evidence_snapshot_digest = NEW.review_snapshot_digest
                AND (
                    (
                        h.decision <> 'ESTABLISH_MAPPING'
                        AND r.evidence_snapshot_digest = NEW.evidence_snapshot_digest
                    )
                    OR (
                        h.decision = 'ESTABLISH_MAPPING'
                        AND r.execution_generation > 1
                        AND r.evidence_snapshot_digest = (
                            SELECT stored_evidence_snapshot_digest(
                                NEW.case_id,
                                predecessor_extraction.payload_json,
                                predecessor_identity.payload_json,
                                predecessor_identity.evaluated_at,
                                predecessor_inventory.payload_json,
                                predecessor_risk.payload_json,
                                predecessor_critique.payload_json
                            )
                            FROM extractions predecessor_extraction
                            JOIN identity_results predecessor_identity
                                ON predecessor_identity.case_id = predecessor_extraction.case_id
                            JOIN comparison_results predecessor_inventory
                                ON predecessor_inventory.case_id = predecessor_extraction.case_id
                                AND predecessor_inventory.comparison_type = 'inventory'
                            JOIN comparison_results predecessor_risk
                                ON predecessor_risk.case_id = predecessor_extraction.case_id
                                AND predecessor_risk.comparison_type = 'risk'
                            JOIN critique_results predecessor_critique
                                ON predecessor_critique.case_id = predecessor_extraction.case_id
                            WHERE predecessor_extraction.case_id = NEW.case_id
                                AND predecessor_extraction.execution_generation =
                                    NEW.execution_generation - 1
                                AND predecessor_identity.execution_generation =
                                    NEW.execution_generation - 1
                                AND predecessor_inventory.execution_generation =
                                    NEW.execution_generation - 1
                                AND predecessor_risk.execution_generation =
                                    NEW.execution_generation - 1
                                AND predecessor_critique.execution_generation =
                                    NEW.execution_generation - 1
                                AND predecessor_extraction.version = (
                                    SELECT MAX(latest.version) FROM extractions latest
                                    WHERE latest.case_id = NEW.case_id
                                        AND latest.execution_generation =
                                            NEW.execution_generation - 1
                                )
                                AND predecessor_identity.rowid = (
                                    SELECT MAX(latest.rowid) FROM identity_results latest
                                    WHERE latest.case_id = NEW.case_id
                                        AND latest.execution_generation =
                                            NEW.execution_generation - 1
                                )
                                AND predecessor_inventory.rowid = (
                                    SELECT MAX(latest.rowid) FROM comparison_results latest
                                    WHERE latest.case_id = NEW.case_id
                                        AND latest.execution_generation =
                                            NEW.execution_generation - 1
                                        AND latest.comparison_type = 'inventory'
                                )
                                AND predecessor_risk.rowid = (
                                    SELECT MAX(latest.rowid) FROM comparison_results latest
                                    WHERE latest.case_id = NEW.case_id
                                        AND latest.execution_generation =
                                            NEW.execution_generation - 1
                                        AND latest.comparison_type = 'risk'
                                )
                                AND predecessor_critique.rowid = (
                                    SELECT MAX(latest.rowid) FROM critique_results latest
                                    WHERE latest.case_id = NEW.case_id
                                        AND latest.execution_generation =
                                            NEW.execution_generation - 1
                                )
                        )
                    )
                )
                AND h.reviewer = json_extract(
                    r.payload_json, '$.human_decision.reviewer'
                )
                AND h.decision = json_extract(
                    r.payload_json, '$.human_decision.decision'
                )
                AND h.reason = json_extract(r.payload_json, '$.human_decision.reason')
                AND julianday(h.decided_at) = julianday(json_extract(
                    r.payload_json, '$.human_decision.decided_at'
                ))
                AND json_valid(h.payload_json) = 1
                AND json_extract(h.payload_json, '$.review_id') = h.review_id
                AND json_extract(h.payload_json, '$.reviewer') = h.reviewer
                AND json_extract(h.payload_json, '$.decision') = h.decision
                AND json_extract(h.payload_json, '$.reason') = h.reason
                AND julianday(json_extract(h.payload_json, '$.decided_at')) =
                    julianday(h.decided_at)
                AND json_extract(h.payload_json, '$.mappings') = json_extract(
                    r.payload_json, '$.human_decision.mappings'
                )
                AND json_extract(h.payload_json, '$.superseded_case_id') IS json_extract(
                    r.payload_json, '$.human_decision.superseded_case_id'
                )
                AND json_extract(h.payload_json, '$.addressed_blocker_ids') = json_extract(
                    r.payload_json, '$.human_decision.addressed_blocker_ids'
                )
                AND (
                    SELECT COUNT(*) FROM human_decisions exact
                    WHERE exact.review_id = r.review_id
                ) = 1
        )
    )
BEGIN
    SELECT RAISE(ABORT, 'VALIDATED_EVIDENCE_SNAPSHOT_INVALID');
END;

CREATE TRIGGER trg_validated_evidence_snapshots_update
BEFORE UPDATE ON validated_evidence_snapshots
BEGIN
    SELECT RAISE(ABORT, 'VALIDATED_EVIDENCE_SNAPSHOT_IMMUTABLE');
END;

CREATE TRIGGER trg_validated_evidence_snapshots_delete
BEFORE DELETE ON validated_evidence_snapshots
BEGIN
    SELECT RAISE(ABORT, 'VALIDATED_EVIDENCE_SNAPSHOT_IMMUTABLE');
END;

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

CREATE TRIGGER trg_final_decisions_authorization_insert
BEFORE INSERT ON final_decisions
WHEN json_valid(NEW.payload_json) <> 1
    OR json_extract(NEW.payload_json, '$.decision') NOT IN ('APPROVE', 'REJECT', 'HOLD', 'FAILED')
    OR NOT (
        (json_extract(NEW.payload_json, '$.decision') = 'APPROVE'
            AND json_extract(NEW.payload_json, '$.payment_eligible') = 1)
        OR (json_extract(NEW.payload_json, '$.decision') <> 'APPROVE'
            AND json_extract(NEW.payload_json, '$.payment_eligible') = 0)
    )
    OR NOT EXISTS (
        SELECT 1
        FROM cases c
        JOIN extractions e ON e.case_id = c.case_id
        WHERE c.case_id = NEW.case_id
            AND e.execution_generation = NEW.decision_generation
            AND e.version = (
                SELECT MAX(latest.version) FROM extractions latest
                WHERE latest.case_id = NEW.case_id
                    AND latest.execution_generation = NEW.decision_generation
            )
            AND c.source_id = NEW.source_id
            AND json_extract(e.payload_json, '$.source.source_id') = NEW.source_id
            AND json_extract(e.payload_json, '$.invoice_number.normalized_value')
                IS NEW.invoice_number
            AND json_extract(e.payload_json, '$.vendor.normalized_value') IS NEW.vendor
            AND json_extract(e.payload_json, '$.declared_total') IS NEW.authorized_amount
            AND json_extract(e.payload_json, '$.currency.normalized_value')
                IS NEW.authorized_currency
            AND NEW.payment_idempotency_key = payment_identity_key(
                NEW.vendor, NEW.invoice_number
            )
    )
    OR NOT EXISTS (
        SELECT 1
        FROM validated_evidence_snapshots anchor
        WHERE anchor.case_id = NEW.case_id
            AND anchor.execution_generation = NEW.decision_generation
            AND anchor.evidence_snapshot_digest = NEW.evidence_snapshot_digest
            AND anchor.review_id IS NEW.review_id
            AND (
                json_extract(NEW.payload_json, '$.decision') <> 'APPROVE'
                OR anchor.unresolved_blocker_count = 0
            )
            AND (
                json_extract(NEW.payload_json, '$.decision') <> 'APPROVE'
                OR anchor.critique_disposition = 'APPROVE'
                OR anchor.review_id IS NOT NULL
            )
            AND (
                (
                    anchor.review_id IS NULL
                    AND json_extract(NEW.payload_json, '$.human_outcome') IS NULL
                )
                OR EXISTS (
                    SELECT 1
                    FROM review_requests r
                    JOIN human_decisions h ON h.review_id = r.review_id
                    WHERE r.review_id = anchor.review_id
                        AND json_extract(
                            NEW.payload_json, '$.human_outcome.review_id'
                        ) = h.review_id
                        AND json_extract(
                            NEW.payload_json, '$.human_outcome.reviewer'
                        ) = h.reviewer
                        AND json_extract(
                            NEW.payload_json, '$.human_outcome.decision'
                        ) = h.decision
                        AND json_extract(
                            NEW.payload_json, '$.human_outcome.reason'
                        ) = h.reason
                        AND (
                            (
                                h.decision IN (
                                    'APPROVE', 'ESTABLISH_MAPPING', 'SUPERSEDE_REVISION'
                                )
                                AND (
                                    json_extract(NEW.payload_json, '$.decision') = 'APPROVE'
                                    OR (
                                        json_extract(NEW.payload_json, '$.decision') = 'HOLD'
                                        AND anchor.unresolved_blocker_count > 0
                                    )
                                )
                            )
                            OR (
                                h.decision = 'REJECT'
                                AND json_extract(NEW.payload_json, '$.decision') = 'REJECT'
                            )
                            OR (
                                h.decision = 'REQUEST_CORRECTION'
                                AND json_extract(NEW.payload_json, '$.decision') = 'HOLD'
                            )
                        )
                        AND julianday(json_extract(
                            NEW.payload_json, '$.human_outcome.decided_at'
                        )) = julianday(h.decided_at)
                        AND json_extract(
                            NEW.payload_json, '$.human_outcome.mappings'
                        ) = json_extract(h.payload_json, '$.mappings')
                        AND json_extract(
                            NEW.payload_json, '$.human_outcome.superseded_case_id'
                        ) IS json_extract(h.payload_json, '$.superseded_case_id')
                        AND json_extract(
                            NEW.payload_json, '$.human_outcome.addressed_blocker_ids'
                        ) = json_extract(h.payload_json, '$.addressed_blocker_ids')
                )
            )
    )
BEGIN
    SELECT RAISE(ABORT, 'FINAL_DECISION_AUTHORIZATION_INVALID');
END;

CREATE TRIGGER trg_final_decisions_immutable_update
BEFORE UPDATE ON final_decisions
BEGIN
    SELECT RAISE(ABORT, 'FINAL_DECISION_IMMUTABLE');
END;

CREATE TRIGGER trg_final_decisions_immutable_delete
BEFORE DELETE ON final_decisions
BEGIN
    SELECT RAISE(ABORT, 'FINAL_DECISION_IMMUTABLE');
END;

CREATE TRIGGER trg_payments_authorization_insert
BEFORE INSERT ON payments
WHEN NEW.status NOT IN ('PAID', 'FAILED')
    OR (NEW.status = 'PAID' AND NEW.error IS NOT NULL)
    OR (NEW.status = 'FAILED' AND NEW.error IS NULL)
    OR NOT EXISTS (
        SELECT 1
        FROM final_decisions f
        JOIN cases c ON c.case_id = f.case_id
        JOIN validated_evidence_snapshots anchor
            ON anchor.case_id = f.case_id
            AND anchor.execution_generation = f.decision_generation
            AND anchor.evidence_snapshot_digest = f.evidence_snapshot_digest
        WHERE f.case_id = NEW.case_id
            AND f.decision_generation = NEW.decision_generation
            AND f.evidence_snapshot_digest = NEW.evidence_snapshot_digest
            AND json_valid(f.payload_json) = 1
            AND json_extract(f.payload_json, '$.decision') = 'APPROVE'
            AND json_extract(f.payload_json, '$.payment_eligible') = 1
            AND f.source_id = NEW.source_id
            AND f.invoice_number = NEW.invoice_number
            AND f.vendor = NEW.vendor
            AND f.authorized_amount = NEW.amount
            AND f.authorized_currency = NEW.currency
            AND f.payment_idempotency_key = NEW.idempotency_key
            AND f.review_id IS NEW.review_id
            AND anchor.review_id IS NEW.review_id
            AND anchor.unresolved_blocker_count = 0
            AND (anchor.critique_disposition = 'APPROVE' OR anchor.review_id IS NOT NULL)
            AND c.source_id = NEW.source_id
            AND NEW.idempotency_key = payment_identity_key(NEW.vendor, NEW.invoice_number)
            AND CAST(NEW.amount AS NUMERIC) > 0
    )
BEGIN
    SELECT RAISE(ABORT, 'PAYMENT_AUTHORIZATION_INVALID');
END;

CREATE TRIGGER trg_payments_immutable_update
BEFORE UPDATE ON payments
BEGIN
    SELECT RAISE(ABORT, 'PAYMENT_IMMUTABLE');
END;

CREATE TRIGGER trg_payments_immutable_delete
BEFORE DELETE ON payments
BEGIN
    SELECT RAISE(ABORT, 'PAYMENT_IMMUTABLE');
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
    OR NOT EXISTS (
        SELECT 1 FROM validated_evidence_snapshots
        WHERE case_id = NEW.case_id
            AND execution_generation = NEW.decision_generation
            AND evidence_snapshot_digest = NEW.evidence_snapshot_digest
    )
BEGIN
    SELECT RAISE(ABORT, 'EVIDENCE_SNAPSHOT_DIGEST_INVALID');
END;

CREATE TRIGGER trg_final_decisions_snapshot_digest_update
BEFORE UPDATE OF evidence_snapshot_digest, decision_generation ON final_decisions
WHEN NEW.evidence_snapshot_digest IS NULL
    OR length(NEW.evidence_snapshot_digest) <> 64
    OR NEW.evidence_snapshot_digest GLOB '*[^0-9a-f]*'
    OR NOT EXISTS (
        SELECT 1 FROM validated_evidence_snapshots
        WHERE case_id = NEW.case_id
            AND execution_generation = NEW.decision_generation
            AND evidence_snapshot_digest = NEW.evidence_snapshot_digest
    )
BEGIN
    SELECT RAISE(ABORT, 'EVIDENCE_SNAPSHOT_DIGEST_INVALID');
END;

CREATE TRIGGER trg_payments_snapshot_digest_insert
BEFORE INSERT ON payments
WHEN NEW.evidence_snapshot_digest IS NULL
    OR length(NEW.evidence_snapshot_digest) <> 64
    OR NEW.evidence_snapshot_digest GLOB '*[^0-9a-f]*'
    OR NOT EXISTS (
        SELECT 1 FROM final_decisions f
        JOIN validated_evidence_snapshots anchor
            ON anchor.case_id = f.case_id
            AND anchor.execution_generation = f.decision_generation
            AND anchor.evidence_snapshot_digest = f.evidence_snapshot_digest
        WHERE f.case_id = NEW.case_id
            AND f.decision_generation = NEW.decision_generation
            AND f.evidence_snapshot_digest = NEW.evidence_snapshot_digest
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
        SELECT 1 FROM final_decisions f
        JOIN validated_evidence_snapshots anchor
            ON anchor.case_id = f.case_id
            AND anchor.execution_generation = f.decision_generation
            AND anchor.evidence_snapshot_digest = f.evidence_snapshot_digest
        WHERE f.case_id = NEW.case_id
            AND f.decision_generation = NEW.decision_generation
            AND f.evidence_snapshot_digest = NEW.evidence_snapshot_digest
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

CREATE TABLE schema_migration_history (
    ordinal INTEGER PRIMARY KEY CHECK(typeof(ordinal) = 'integer' AND ordinal >= 1),
    version INTEGER NOT NULL UNIQUE CHECK(typeof(version) = 'integer' AND version >= 1),
    migration_sha256 TEXT NOT NULL CHECK(
        typeof(migration_sha256) = 'text'
        AND length(migration_sha256) = 64
        AND migration_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    applied_at TEXT NOT NULL CHECK(
        typeof(applied_at) = 'text'
        AND substr(applied_at, 1, 4) <> '0000'
        AND (
            (length(applied_at) = 25 AND applied_at GLOB
                '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]+00:00')
            OR (length(applied_at) = 32 AND applied_at GLOB
                '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]+00:00'
                AND substr(applied_at, 21, 6) <> '000000')
        )
        AND datetime(applied_at) IS NOT NULL
        AND strftime('%Y-%m-%dT%H:%M:%S', applied_at) = substr(applied_at, 1, 19)
        AND CAST(substr(applied_at, 12, 2) AS INTEGER) BETWEEN 0 AND 23
        AND CAST(substr(applied_at, 15, 2) AS INTEGER) BETWEEN 0 AND 59
        AND CAST(substr(applied_at, 18, 2) AS INTEGER) BETWEEN 0 AND 59
    )
);

CREATE TRIGGER trg_schema_migration_history_monotonic_insert
BEFORE INSERT ON schema_migration_history
WHEN NEW.ordinal <> COALESCE((SELECT MAX(ordinal) FROM schema_migration_history), 0) + 1
    OR NEW.version <> NEW.ordinal
BEGIN
    SELECT RAISE(ABORT, 'MIGRATION_HISTORY_SEQUENCE_INVALID');
END;

CREATE TRIGGER trg_schema_migration_history_immutable_update
BEFORE UPDATE ON schema_migration_history
BEGIN
    SELECT RAISE(ABORT, 'MIGRATION_HISTORY_IMMUTABLE');
END;

CREATE TRIGGER trg_schema_migration_history_immutable_delete
BEFORE DELETE ON schema_migration_history
BEGIN
    SELECT RAISE(ABORT, 'MIGRATION_HISTORY_IMMUTABLE');
END;
