-- Bind a result artifact only to the exact current terminal execution and its
-- durable filesystem identity.  Existing v4 rows intentionally receive no
-- binding: no migration can reconstruct inode, device, or file durability.
CREATE UNIQUE INDEX idx_cases_case_generation
ON cases(case_id, execution_generation);

CREATE TABLE result_artifact_bindings (
    case_id TEXT NOT NULL PRIMARY KEY,
    execution_generation INTEGER NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    artifact_device INTEGER NOT NULL,
    artifact_inode INTEGER NOT NULL,
    artifact_file_type INTEGER NOT NULL,
    artifact_size_bytes INTEGER NOT NULL,
    FOREIGN KEY(case_id, execution_generation)
        REFERENCES cases(case_id, execution_generation)
        ON UPDATE RESTRICT ON DELETE CASCADE,
    CHECK(typeof(execution_generation) = 'integer' AND execution_generation >= 1),
    CHECK(
        typeof(artifact_sha256) = 'text'
        AND length(artifact_sha256) = 64
        AND artifact_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK(typeof(artifact_device) = 'integer' AND artifact_device >= 0),
    CHECK(typeof(artifact_inode) = 'integer' AND artifact_inode > 0),
    CHECK(typeof(artifact_file_type) = 'integer' AND artifact_file_type = 32768),
    CHECK(typeof(artifact_size_bytes) = 'integer' AND artifact_size_bytes >= 0)
);

CREATE TRIGGER trg_result_artifact_bindings_terminal_insert
BEFORE INSERT ON result_artifact_bindings
WHEN NOT EXISTS (
    SELECT 1 FROM cases AS parent
    WHERE parent.case_id = NEW.case_id
    AND parent.execution_generation = NEW.execution_generation
    AND parent.execution_state = 'FINISHED'
    AND typeof(parent.execution_token) = 'text'
    AND length(parent.execution_token) = 37
    AND parent.execution_token GLOB 'exec_[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'
    AND typeof(parent.result_json) = 'text'
    AND parent.lease_expires_at IS NULL
)
BEGIN
    SELECT RAISE(ABORT, 'RESULT_ARTIFACT_BINDING_PARENT_INVALID');
END;

CREATE TRIGGER trg_result_artifact_bindings_terminal_update
BEFORE UPDATE ON result_artifact_bindings
WHEN NOT EXISTS (
    SELECT 1 FROM cases AS parent
    WHERE parent.case_id = NEW.case_id
    AND parent.execution_generation = NEW.execution_generation
    AND parent.execution_state = 'FINISHED'
    AND typeof(parent.execution_token) = 'text'
    AND length(parent.execution_token) = 37
    AND parent.execution_token GLOB 'exec_[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'
    AND typeof(parent.result_json) = 'text'
    AND parent.lease_expires_at IS NULL
)
BEGIN
    SELECT RAISE(ABORT, 'RESULT_ARTIFACT_BINDING_PARENT_INVALID');
END;

CREATE TRIGGER trg_cases_result_artifact_binding_guard_update
BEFORE UPDATE OF execution_token, execution_generation, execution_state,
    lease_expires_at, result_json
ON cases
WHEN EXISTS (
    SELECT 1 FROM result_artifact_bindings AS binding
    WHERE binding.case_id = OLD.case_id
)
BEGIN
    SELECT RAISE(ABORT, 'RESULT_ARTIFACT_BINDING_MUST_BE_INVALIDATED');
END;
