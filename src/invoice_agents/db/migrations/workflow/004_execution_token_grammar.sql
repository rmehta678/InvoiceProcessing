-- aa9e43b issued recovery_<uuid4.hex> only while atomically terminalizing an
-- eligible IDLE or expired RUNNING generation.  Record only rows whose complete
-- terminal shape proves that exact historical path; no other noncanonical
-- authority is eligible for automatic reconciliation.
CREATE TABLE execution_token_migration_guard (
    valid INTEGER NOT NULL CHECK(valid = 1)
);

-- Preserve the original fail-fast boundary for authorities that are not even
-- lexical recovery candidates.  Exact recovery tokens continue below to the
-- mandatory strict payload validator before they can be reconciled.
INSERT INTO execution_token_migration_guard(valid)
SELECT 0
FROM cases
WHERE execution_state IN ('RUNNING', 'FINISHED')
AND NOT (
    typeof(execution_token) = 'text'
    AND length(execution_token) = 37
    AND execution_token GLOB 'exec_[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'
)
AND NOT (
    typeof(execution_token) = 'text'
    AND length(execution_token) = 41
    AND execution_token GLOB 'recovery_[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'
)
LIMIT 1;

CREATE TABLE execution_token_recovery_reconciliation (
    case_id TEXT PRIMARY KEY,
    original_token TEXT NOT NULL,
    replacement_token TEXT NOT NULL CHECK(
        typeof(replacement_token) = 'text'
        AND length(replacement_token) = 37
        AND replacement_token GLOB 'exec_[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'
    )
);

INSERT INTO execution_token_recovery_reconciliation(
    case_id,
    original_token,
    replacement_token
)
SELECT
    c.case_id,
    c.execution_token,
    'exec_' || substr(c.execution_token, 10)
FROM cases AS c
WHERE c.execution_state = 'FINISHED'
AND typeof(c.execution_token) = 'text'
AND length(c.execution_token) = 41
AND c.execution_token GLOB 'recovery_[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'
AND c.status = 'INCOMPLETE'
AND c.stop_reason = 'ORPHANED_EXECUTION'
AND typeof(c.execution_generation) = 'integer'
AND c.execution_generation >= 1
AND c.lease_expires_at IS NULL
AND typeof(c.started_at) = 'text'
AND typeof(c.updated_at) = 'text'
AND typeof(c.finished_at) = 'text'
AND substr(c.started_at, -6) = '+00:00'
AND substr(c.updated_at, -6) = '+00:00'
AND substr(c.finished_at, -6) = '+00:00'
AND datetime(c.started_at) IS NOT NULL
AND datetime(c.finished_at) IS NOT NULL
AND julianday(c.finished_at) >= julianday(c.started_at)
AND c.finished_at = c.updated_at
AND (c.source_id IS NULL OR EXISTS (
    SELECT 1 FROM source_artifacts AS source
    WHERE source.source_id = c.source_id
))
-- This deterministic connection-local UDF is mandatory.  An unregistered
-- validator is a SQLite error, so migration cannot silently skip certification.
AND CASE
    WHEN typeof(c.result_json) <> 'text' OR json_valid(c.result_json) <> 1 THEN 0
    ELSE (
        strict_case_result_json(c.result_json) = 1
        AND json_type(c.result_json, '$') = 'object'
        AND (SELECT COUNT(*) FROM json_each(c.result_json)) = 11
        AND NOT EXISTS (
            SELECT 1
            FROM json_each(c.result_json) AS root_field
            WHERE root_field.key NOT IN (
                'case_id',
                'source_id',
                'status',
                'stop_reason',
                'final_decision',
                'review_request',
                'payment',
                'errors',
                'usage',
                'started_at',
                'finished_at'
            )
        )
        AND json_type(c.result_json, '$.case_id') = 'text'
        AND json_extract(c.result_json, '$.case_id') = c.case_id
        AND json_type(c.result_json, '$.source_id') IN ('text', 'null')
        AND json_extract(c.result_json, '$.source_id') IS c.source_id
        AND json_type(c.result_json, '$.status') = 'text'
        AND json_extract(c.result_json, '$.status') = c.status
        AND json_type(c.result_json, '$.stop_reason') = 'text'
        AND json_extract(c.result_json, '$.stop_reason') = c.stop_reason
        AND json_type(c.result_json, '$.started_at') = 'text'
        AND json_extract(c.result_json, '$.started_at') =
            substr(c.started_at, 1, length(c.started_at) - 6) || 'Z'
        AND json_type(c.result_json, '$.finished_at') = 'text'
        AND json_extract(c.result_json, '$.finished_at') =
            substr(c.finished_at, 1, length(c.finished_at) - 6) || 'Z'
        AND json_type(c.result_json, '$.errors') = 'array'
        AND json_array_length(c.result_json, '$.errors') >= 1
        AND json_type(c.result_json, '$.errors[#-1]') = 'object'
        AND (SELECT COUNT(*) FROM json_each(c.result_json, '$.errors[#-1]')) = 6
        AND NOT EXISTS (
            SELECT 1
            FROM json_each(c.result_json, '$.errors[#-1]') AS error_field
            WHERE error_field.key NOT IN (
                'category',
                'message',
                'case_id',
                'stop_reason',
                'provider_request_id',
                'details'
            )
        )
        AND json_type(c.result_json, '$.errors[#-1].category') = 'text'
        AND json_extract(c.result_json, '$.errors[#-1].category') = 'ORCHESTRATION'
        AND json_type(c.result_json, '$.errors[#-1].message') = 'text'
        AND json_extract(c.result_json, '$.errors[#-1].message') =
            'execution lease expired before a terminal result was recorded'
        AND json_type(c.result_json, '$.errors[#-1].case_id') = 'text'
        AND json_extract(c.result_json, '$.errors[#-1].case_id') = c.case_id
        AND json_type(c.result_json, '$.errors[#-1].stop_reason') = 'text'
        AND json_extract(c.result_json, '$.errors[#-1].stop_reason') =
            'ORPHANED_EXECUTION'
        AND json_type(c.result_json, '$.errors[#-1].provider_request_id') = 'null'
        AND json_type(c.result_json, '$.errors[#-1].details') = 'object'
        AND (SELECT COUNT(*) FROM json_each(
            c.result_json,
            '$.errors[#-1].details'
        )) = 1
        AND NOT EXISTS (
            SELECT 1
            FROM json_each(
                c.result_json,
                '$.errors[#-1].details'
            ) AS detail_field
            WHERE detail_field.key <> 'abandoned_execution_generation'
        )
        AND json_type(
            c.result_json,
            '$.errors[#-1].details.abandoned_execution_generation'
        ) = 'integer'
        AND json_extract(
            c.result_json,
            '$.errors[#-1].details.abandoned_execution_generation'
        ) = c.execution_generation - 1
    )
END;

-- Reject every other pre-existing authority that cannot satisfy the canonical
-- grammar.  Both transient tables are inside the migration transaction, so a
-- rejection leaves v3 schema, history, and case data unchanged and retryable.
INSERT INTO execution_token_migration_guard(valid)
SELECT 0
FROM cases
WHERE execution_state IN ('RUNNING', 'FINISHED')
AND NOT (
    typeof(execution_token) = 'text'
    AND length(execution_token) = 37
    AND execution_token GLOB 'exec_[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'
)
AND NOT EXISTS (
    SELECT 1
    FROM execution_token_recovery_reconciliation AS recovery
    WHERE recovery.case_id = cases.case_id
    AND recovery.original_token = cases.execution_token
)
LIMIT 1;

UPDATE cases
SET execution_token = (
    SELECT recovery.replacement_token
    FROM execution_token_recovery_reconciliation AS recovery
    WHERE recovery.case_id = cases.case_id
    AND recovery.original_token = cases.execution_token
)
WHERE EXISTS (
    SELECT 1
    FROM execution_token_recovery_reconciliation AS recovery
    WHERE recovery.case_id = cases.case_id
    AND recovery.original_token = cases.execution_token
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
DROP TABLE execution_token_recovery_reconciliation;

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
