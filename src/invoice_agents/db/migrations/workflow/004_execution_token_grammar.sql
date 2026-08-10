-- Reject any pre-existing authority that cannot satisfy the canonical token
-- grammar.  The transient guard deliberately aborts this migration before any
-- durable schema object is changed; the migration runner rolls the transaction
-- back and leaves v3 retryable after explicit reconciliation.
CREATE TABLE execution_token_migration_guard (
    valid INTEGER NOT NULL CHECK(valid = 1)
);

INSERT INTO execution_token_migration_guard(valid)
SELECT 0
FROM cases
WHERE execution_state IN ('RUNNING', 'FINISHED')
AND NOT (
    typeof(execution_token) = 'text'
    AND length(execution_token) = 37
    AND execution_token GLOB 'exec_[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'
)
LIMIT 1;

DROP TABLE execution_token_migration_guard;

CREATE TRIGGER trg_cases_execution_token_grammar_insert
BEFORE INSERT ON cases
WHEN NEW.execution_state IN ('RUNNING', 'FINISHED')
AND NOT (
    typeof(NEW.execution_token) = 'text'
    AND length(NEW.execution_token) = 37
    AND NEW.execution_token GLOB 'exec_[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'
)
BEGIN
    SELECT RAISE(ABORT, 'INVALID_EXECUTION_TOKEN');
END;

CREATE TRIGGER trg_cases_execution_token_grammar_update
BEFORE UPDATE OF execution_token, execution_generation, execution_state, lease_expires_at
ON cases
WHEN NEW.execution_state IN ('RUNNING', 'FINISHED')
AND NOT (
    typeof(NEW.execution_token) = 'text'
    AND length(NEW.execution_token) = 37
    AND NEW.execution_token GLOB 'exec_[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'
)
BEGIN
    SELECT RAISE(ABORT, 'INVALID_EXECUTION_TOKEN');
END;
