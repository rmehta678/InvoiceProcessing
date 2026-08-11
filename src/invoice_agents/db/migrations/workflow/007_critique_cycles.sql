CREATE TABLE legacy_critique_history (
    source_rowid INTEGER NOT NULL,
    critique_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    execution_generation INTEGER NOT NULL,
    migration_role TEXT NOT NULL CHECK(
        migration_role IN ('PROMOTED_CYCLE_ONE', 'HISTORICAL_SUPERSEDED')
    ),
    PRIMARY KEY(case_id, source_rowid),
    UNIQUE(critique_id)
);

CREATE TRIGGER trg_legacy_critique_history_ambiguous_insert
BEFORE INSERT ON legacy_critique_history
WHEN NEW.migration_role = 'PROMOTED_CYCLE_ONE'
    AND EXISTS (
        SELECT 1 FROM legacy_critique_history prior
        WHERE prior.case_id = NEW.case_id
          AND prior.migration_role = 'PROMOTED_CYCLE_ONE'
    )
BEGIN
    SELECT RAISE(ABORT, 'LEGACY_CRITIQUE_HISTORY_AMBIGUOUS');
END;

INSERT INTO legacy_critique_history(
    source_rowid,
    critique_id,
    case_id,
    payload_json,
    created_at,
    execution_generation,
    migration_role
)
SELECT
    legacy.rowid,
    legacy.critique_id,
    legacy.case_id,
    legacy.payload_json,
    legacy.created_at,
    legacy.execution_generation,
    CASE
        WHEN legacy.created_at = (
            SELECT MAX(candidate.created_at)
            FROM critique_results candidate
            WHERE candidate.case_id = legacy.case_id
        ) THEN 'PROMOTED_CYCLE_ONE'
        ELSE 'HISTORICAL_SUPERSEDED'
    END
FROM critique_results legacy
ORDER BY legacy.case_id, legacy.created_at DESC, legacy.rowid;

CREATE TRIGGER trg_legacy_critique_history_immutable_insert
BEFORE INSERT ON legacy_critique_history
BEGIN
    SELECT RAISE(ABORT, 'LEGACY_CRITIQUE_HISTORY_IMMUTABLE');
END;

CREATE TRIGGER trg_legacy_critique_history_immutable_update
BEFORE UPDATE ON legacy_critique_history
BEGIN
    SELECT RAISE(ABORT, 'LEGACY_CRITIQUE_HISTORY_IMMUTABLE');
END;

CREATE TRIGGER trg_legacy_critique_history_immutable_delete
BEFORE DELETE ON legacy_critique_history
BEGIN
    SELECT RAISE(ABORT, 'LEGACY_CRITIQUE_HISTORY_IMMUTABLE');
END;

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
FROM legacy_critique_history
WHERE migration_role = 'PROMOTED_CYCLE_ONE';

DROP TABLE critique_results;
ALTER TABLE critique_results_v7 RENAME TO critique_results;

CREATE UNIQUE INDEX idx_critique_results_case_cycle
ON critique_results(case_id, cycle);

CREATE TABLE critique_follow_up_evidence (
    critique_id TEXT NOT NULL REFERENCES critique_results(critique_id),
    requested_item TEXT NOT NULL CHECK(length(trim(requested_item)) > 0),
    outcome TEXT NOT NULL CHECK(outcome IN ('SUPPORTED', 'CHALLENGED', 'MISSING')),
    evidence_event_id TEXT NOT NULL REFERENCES events(event_id),
    PRIMARY KEY(critique_id, requested_item, evidence_event_id),
    UNIQUE(critique_id, evidence_event_id)
);

CREATE INDEX idx_critique_follow_up_evidence_event
ON critique_follow_up_evidence(evidence_event_id);

CREATE TRIGGER trg_critique_results_cycle_parent_insert
BEFORE INSERT ON critique_results
WHEN NEW.cycle = 2 AND NOT EXISTS (
    SELECT 1
    FROM critique_results AS parent
    WHERE parent.critique_id = NEW.responds_to_critique_id
      AND parent.case_id = NEW.case_id
      AND parent.cycle = 1
      AND typeof(NEW.execution_generation) = 'integer'
      AND NEW.execution_generation >= parent.execution_generation
      AND strict_canonical_utc_micros(NEW.created_at) IS NOT NULL
      AND strict_canonical_utc_micros(parent.created_at) IS NOT NULL
      AND strict_canonical_utc_micros(NEW.created_at)
          > strict_canonical_utc_micros(parent.created_at)
      AND CASE
          WHEN json_valid(parent.payload_json) = 1 THEN
              COALESCE(json_array_length(parent.payload_json, '$.challenged_findings'), 0)
              + COALESCE(json_array_length(parent.payload_json, '$.missing_evidence'), 0)
              + COALESCE(json_array_length(parent.payload_json, '$.requested_follow_up'), 0)
          ELSE 0
      END > 0
)
BEGIN
    SELECT RAISE(ABORT, 'CRITIQUE_RESPONSE_INVALID');
END;

CREATE TRIGGER trg_critique_results_cycle_parent_update
BEFORE UPDATE ON critique_results
WHEN NEW.cycle = 2 AND NOT EXISTS (
    SELECT 1
    FROM critique_results AS parent
    WHERE parent.critique_id = NEW.responds_to_critique_id
      AND parent.case_id = NEW.case_id
      AND parent.cycle = 1
      AND typeof(NEW.execution_generation) = 'integer'
      AND NEW.execution_generation >= parent.execution_generation
      AND strict_canonical_utc_micros(NEW.created_at) IS NOT NULL
      AND strict_canonical_utc_micros(parent.created_at) IS NOT NULL
      AND strict_canonical_utc_micros(NEW.created_at)
          > strict_canonical_utc_micros(parent.created_at)
      AND CASE
          WHEN json_valid(parent.payload_json) = 1 THEN
              COALESCE(json_array_length(parent.payload_json, '$.challenged_findings'), 0)
              + COALESCE(json_array_length(parent.payload_json, '$.missing_evidence'), 0)
              + COALESCE(json_array_length(parent.payload_json, '$.requested_follow_up'), 0)
          ELSE 0
      END > 0
)
BEGIN
    SELECT RAISE(ABORT, 'CRITIQUE_RESPONSE_INVALID');
END;

CREATE TRIGGER trg_critique_follow_up_evidence_insert
BEFORE INSERT ON critique_follow_up_evidence
WHEN NOT EXISTS (
    SELECT 1
    FROM critique_results child
    JOIN critique_results parent
      ON parent.critique_id = child.responds_to_critique_id
     AND parent.case_id = child.case_id
     AND parent.cycle = 1
    JOIN events evidence
      ON evidence.event_id = NEW.evidence_event_id
     AND evidence.case_id = child.case_id
     AND evidence.created_at > parent.created_at
    WHERE child.critique_id = NEW.critique_id
      AND child.cycle = 2
      AND (
          (
              evidence.event_type = 'tool.critic_line_recompute'
              AND evidence.agent_name = 'independent_critic_agent'
              AND evidence.source_id IS NULL
              AND evidence.tool_call_id IS NULL
              AND evidence.db_evidence_id IS NULL
              AND evidence.review_id IS NULL
              AND evidence.payment_id IS NULL
              AND evidence.provider_request_id IS NULL
              AND strict_critic_follow_up_payload(
                  evidence.event_type,
                  evidence.payload_json,
                  child.execution_generation
              ) = 1
          )
          OR (
              evidence.event_type = 'tool.critic_inventory_recheck'
              AND evidence.agent_name = 'independent_critic_agent'
              AND evidence.source_id IS NULL
              AND evidence.tool_call_id IS NULL
              AND evidence.db_evidence_id IS NULL
              AND evidence.review_id IS NULL
              AND evidence.payment_id IS NULL
              AND evidence.provider_request_id IS NULL
              AND strict_critic_follow_up_payload(
                  evidence.event_type,
                  evidence.payload_json,
                  child.execution_generation
              ) = 1
          )
          OR (
              evidence.event_type = 'tool.identity_candidates'
              AND evidence.agent_name = 'identity_provenance_agent'
              AND evidence.source_id = (
                  SELECT source_id FROM cases WHERE case_id = child.case_id
              )
              AND evidence.tool_call_id IS NULL
              AND evidence.review_id IS NULL
              AND evidence.payment_id IS NULL
              AND evidence.provider_request_id IS NULL
              AND EXISTS (
                  SELECT 1 FROM identity_results identity_evidence
                  WHERE identity_evidence.identity_id = evidence.db_evidence_id
                    AND identity_evidence.case_id = child.case_id
                    AND identity_evidence.execution_generation = child.execution_generation
                    AND identity_evidence.created_at <= evidence.created_at
              )
          )
          OR (
              evidence.event_type = 'tool.inventory_comparison'
              AND evidence.agent_name = 'inventory_comparison_agent'
              AND evidence.source_id = (
                  SELECT source_id FROM cases WHERE case_id = child.case_id
              )
              AND evidence.tool_call_id IS NULL
              AND evidence.review_id IS NULL
              AND evidence.payment_id IS NULL
              AND evidence.provider_request_id IS NULL
              AND EXISTS (
                  SELECT 1 FROM comparison_results inventory_evidence
                  WHERE inventory_evidence.comparison_id = evidence.db_evidence_id
                    AND inventory_evidence.case_id = child.case_id
                    AND inventory_evidence.execution_generation = child.execution_generation
                    AND inventory_evidence.comparison_type = 'inventory'
                    AND inventory_evidence.created_at <= evidence.created_at
              )
          )
          OR (
              evidence.event_type = 'tool.mapping_evidence_recorded'
              AND evidence.agent_name = 'inventory_comparison_agent'
              AND evidence.source_id = (
                  SELECT source_id FROM cases WHERE case_id = child.case_id
              )
              AND evidence.tool_call_id IS NULL
              AND evidence.review_id IS NULL
              AND evidence.payment_id IS NULL
              AND evidence.provider_request_id IS NULL
              AND json_valid(evidence.payload_json) = 1
              AND json_extract(evidence.payload_json, '$.extraction_id') =
                  evidence.db_evidence_id
              AND EXISTS (
                  SELECT 1 FROM extractions extraction_evidence
                  WHERE extraction_evidence.extraction_id = evidence.db_evidence_id
                    AND extraction_evidence.case_id = child.case_id
                    AND extraction_evidence.execution_generation = child.execution_generation
                    AND extraction_evidence.created_at <= evidence.created_at
              )
          )
          OR (
              evidence.event_type = 'tool.financial_risk_assessment'
              AND evidence.agent_name = 'financial_risk_agent'
              AND evidence.source_id = (
                  SELECT source_id FROM cases WHERE case_id = child.case_id
              )
              AND evidence.tool_call_id IS NULL
              AND evidence.review_id IS NULL
              AND evidence.payment_id IS NULL
              AND evidence.provider_request_id IS NULL
              AND EXISTS (
                  SELECT 1 FROM comparison_results risk_evidence
                  WHERE risk_evidence.comparison_id = evidence.db_evidence_id
                    AND risk_evidence.case_id = child.case_id
                    AND risk_evidence.execution_generation = child.execution_generation
                    AND risk_evidence.comparison_type = 'risk'
                    AND risk_evidence.created_at <= evidence.created_at
              )
          )
      )
)
BEGIN
    SELECT RAISE(ABORT, 'CRITIQUE_FOLLOW_UP_EVIDENCE_INVALID');
END;

CREATE TRIGGER trg_critique_follow_up_evidence_immutable_after_authorization_insert
BEFORE INSERT ON critique_follow_up_evidence
WHEN EXISTS (
    SELECT 1
    FROM critique_results critique
    JOIN final_decisions final ON final.case_id = critique.case_id
    WHERE critique.critique_id = NEW.critique_id
) OR EXISTS (
    SELECT 1
    FROM critique_results critique
    JOIN payments payment ON payment.case_id = critique.case_id
    WHERE critique.critique_id = NEW.critique_id
      AND payment.status = 'PAID'
)
BEGIN
    SELECT RAISE(ABORT, 'AUTHORIZATION_EVIDENCE_IMMUTABLE');
END;

CREATE TRIGGER trg_critique_follow_up_evidence_immutable_update
BEFORE UPDATE ON critique_follow_up_evidence
BEGIN
    SELECT RAISE(ABORT, 'CRITIQUE_FOLLOW_UP_EVIDENCE_IMMUTABLE');
END;

CREATE TRIGGER trg_critique_follow_up_evidence_immutable_delete
BEFORE DELETE ON critique_follow_up_evidence
BEGIN
    SELECT RAISE(ABORT, 'CRITIQUE_FOLLOW_UP_EVIDENCE_IMMUTABLE');
END;

CREATE TRIGGER trg_events_bound_critique_evidence_immutable_update
BEFORE UPDATE ON events
WHEN EXISTS (
    SELECT 1 FROM critique_follow_up_evidence evidence
    WHERE evidence.evidence_event_id = OLD.event_id
)
BEGIN
    SELECT RAISE(ABORT, 'CRITIQUE_FOLLOW_UP_EVIDENCE_IMMUTABLE');
END;

CREATE TRIGGER trg_events_bound_critique_evidence_immutable_delete
BEFORE DELETE ON events
WHEN EXISTS (
    SELECT 1 FROM critique_follow_up_evidence evidence
    WHERE evidence.evidence_event_id = OLD.event_id
)
BEGIN
    SELECT RAISE(ABORT, 'CRITIQUE_FOLLOW_UP_EVIDENCE_IMMUTABLE');
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

DROP TRIGGER trg_validated_evidence_snapshots_insert;

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
            AND critique.execution_generation <= NEW.execution_generation
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
            AND critique.cycle = (
                SELECT MAX(latest.cycle) FROM critique_results latest
                WHERE latest.case_id = NEW.case_id
                    AND latest.execution_generation <= NEW.execution_generation
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
                                AND predecessor_critique.execution_generation <=
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
                                AND predecessor_critique.cycle = (
                                    SELECT MAX(latest.cycle) FROM critique_results latest
                                    WHERE latest.case_id = NEW.case_id
                                        AND latest.execution_generation <=
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
