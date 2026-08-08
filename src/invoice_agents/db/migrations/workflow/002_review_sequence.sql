-- Review cycles: review_requests.case_id UNIQUE structurally forbade a second
-- review for a case. Recreate the table with an explicit per-case sequence so a
-- resolved authorizing decision can be followed by another review when blocking
-- evidence remains. Existing rows become sequence 1.
PRAGMA foreign_keys=OFF;
BEGIN;

CREATE TABLE review_requests_v2 (
    review_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    sequence INTEGER NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

INSERT INTO review_requests_v2 (
    review_id, case_id, sequence, status, payload_json, created_at, resolved_at
)
SELECT review_id, case_id, 1, status, payload_json, created_at, resolved_at
FROM review_requests;

DROP TABLE review_requests;

ALTER TABLE review_requests_v2 RENAME TO review_requests;

CREATE UNIQUE INDEX idx_review_requests_case_sequence ON review_requests(case_id, sequence);

COMMIT;
PRAGMA foreign_keys=ON;
