ALTER TABLE cases ADD COLUMN execution_token TEXT;
ALTER TABLE cases ADD COLUMN execution_generation INTEGER NOT NULL DEFAULT 0;
ALTER TABLE cases ADD COLUMN execution_state TEXT NOT NULL DEFAULT 'IDLE';
ALTER TABLE cases ADD COLUMN lease_expires_at TEXT;

ALTER TABLE final_decisions ADD COLUMN decision_generation INTEGER NOT NULL DEFAULT 0;

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
