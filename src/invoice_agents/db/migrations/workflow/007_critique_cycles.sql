CREATE TABLE critique_results_v7 (
    critique_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    execution_generation INTEGER NOT NULL DEFAULT 0,
    cycle INTEGER NOT NULL DEFAULT 1 CHECK(cycle IN (1, 2)),
    responds_to_critique_id TEXT REFERENCES critique_results_v7(critique_id),
    CHECK(
        (cycle = 1 AND responds_to_critique_id IS NULL)
        OR (cycle = 2 AND responds_to_critique_id IS NOT NULL)
    )
);

INSERT INTO critique_results_v7(
    critique_id,
    case_id,
    payload_json,
    created_at,
    execution_generation,
    cycle,
    responds_to_critique_id
)
SELECT
    critique_id,
    case_id,
    payload_json,
    created_at,
    execution_generation,
    1,
    NULL
FROM critique_results;

DROP TABLE critique_results;
ALTER TABLE critique_results_v7 RENAME TO critique_results;

CREATE UNIQUE INDEX idx_critique_results_case_cycle
ON critique_results(case_id, cycle);

CREATE TRIGGER trg_critique_results_cycle_parent_insert
BEFORE INSERT ON critique_results
WHEN NEW.cycle = 2 AND NOT EXISTS (
    SELECT 1
    FROM critique_results AS parent
    WHERE parent.critique_id = NEW.responds_to_critique_id
      AND parent.case_id = NEW.case_id
      AND parent.cycle = 1
)
BEGIN
    SELECT RAISE(ABORT, 'CRITIQUE_RESPONSE_INVALID');
END;

CREATE TRIGGER trg_critique_results_cycle_parent_update
BEFORE UPDATE OF case_id, cycle, responds_to_critique_id ON critique_results
WHEN NEW.cycle = 2 AND NOT EXISTS (
    SELECT 1
    FROM critique_results AS parent
    WHERE parent.critique_id = NEW.responds_to_critique_id
      AND parent.case_id = NEW.case_id
      AND parent.cycle = 1
)
BEGIN
    SELECT RAISE(ABORT, 'CRITIQUE_RESPONSE_INVALID');
END;

CREATE TRIGGER trg_critique_results_immutable_after_final_insert
BEFORE INSERT ON critique_results
WHEN EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = NEW.case_id
) OR EXISTS (SELECT 1 FROM payments WHERE case_id = NEW.case_id AND status = 'PAID')
BEGIN
    SELECT RAISE(ABORT, 'AUTHORIZATION_EVIDENCE_IMMUTABLE');
END;

CREATE TRIGGER trg_critique_results_immutable_after_final_update
BEFORE UPDATE ON critique_results
WHEN EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = OLD.case_id
) OR EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = NEW.case_id
) OR EXISTS (SELECT 1 FROM payments WHERE case_id IN (OLD.case_id, NEW.case_id) AND status = 'PAID')
BEGIN
    SELECT RAISE(ABORT, 'AUTHORIZATION_EVIDENCE_IMMUTABLE');
END;

CREATE TRIGGER trg_critique_results_immutable_after_final_delete
BEFORE DELETE ON critique_results
WHEN EXISTS (
    SELECT 1 FROM final_decisions
    WHERE case_id = OLD.case_id
) OR EXISTS (SELECT 1 FROM payments WHERE case_id = OLD.case_id AND status = 'PAID')
BEGIN
    SELECT RAISE(ABORT, 'AUTHORIZATION_EVIDENCE_IMMUTABLE');
END;
