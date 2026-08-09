"""Guard scenarios: stops, unresolved reviews, locks, and bad inputs stay loudly visible."""

import json
import os
import sqlite3
import stat
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import httpx
import openai
import pytest
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import TextMessage

from invoice_agents import orchestration
from invoice_agents.agents.decision_rules import validate_final_decision
from invoice_agents.config import Settings
from invoice_agents.db.core import DatabaseKind, connect_database, verify_database
from invoice_agents.db.store import WorkflowStore
from invoice_agents.errors import (
    DatabaseVerificationError,
    InvoiceAgentsError,
    SourceEvidenceError,
)
from invoice_agents.hitl.service import create_review_request, record_human_decision
from invoice_agents.models import (
    CaseStatus,
    Critique,
    DecisionKind,
    FinalDecision,
    FinancialComparison,
    HumanDecisionKind,
    InventoryComparison,
    InventoryStatus,
    ReviewRequest,
    RiskAssessment,
    SourceArtifact,
)
from invoice_agents.observability.audit import AuditRecorder
from invoice_agents.orchestration import (
    _error_record,
    is_max_messages_stop,
    prepare_case,
    process_invoice,
    run_prepared_case,
)
from invoice_agents.payment.service import mock_payment
from invoice_agents.source_store import snapshot_source
from invoice_agents.tools.evidence import extract_invoice_evidence

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


class StubModelClient:
    """Stand-in for the OpenAI client; run_prepared_case only awaits close()."""

    async def close(self) -> None:
        return None


class FakeTeam:
    """Stream one canned TaskResult, or raise before yielding, with no provider traffic."""

    def __init__(
        self,
        task_result: TaskResult | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._task_result = task_result
        self._error = error

    async def run_stream(self, task: object) -> AsyncIterator[object]:
        if self._error is not None:
            raise self._error
        assert self._task_result is not None
        yield self._task_result

    async def save_state(self) -> dict[str, object]:
        return {}


async def _pinned_max_messages_phrase() -> str:
    """Drive the real AutoGen termination condition and return its stop phrasing."""

    termination = MaxMessageTermination(2, include_agent_event=False)
    stop = await termination(
        [TextMessage(content="a", source="x"), TextMessage(content="b", source="y")]
    )
    assert stop is not None, "MaxMessageTermination(2) did not stop after two chat messages"
    return stop.content


def _prepare(
    invoice_dir: Path, settings: Settings, name: str = "invoice_1001.txt"
) -> tuple[str, datetime]:
    prepared = prepare_case(invoice_dir / name, settings)
    assert isinstance(prepared, tuple), "case preparation must succeed before the team runs"
    return prepared


def _zero_financial() -> FinancialComparison:
    zero = Decimal("0")
    return FinancialComparison(
        calculated_subtotal=zero,
        declared_subtotal=None,
        subtotal_delta=None,
        calculated_tax=zero,
        declared_tax=None,
        tax_delta=None,
        tax_recomputable=False,
        tax_basis="no declared tax to recompute",
        calculated_fees=zero,
        calculated_total=zero,
        declared_total=None,
        total_delta=None,
        line_deltas={},
        exact=True,
    )


def _risk(policy_review_reasons: list[str]) -> RiskAssessment:
    return RiskAssessment(
        financial=_zero_financial(),
        dates=[],
        inventory=[],
        identity_candidates=[],
        suspicious_signals=[],
        unavailable_reconciliations=[],
        policy_review_reasons=policy_review_reasons,
    )


def _critique(disposition: DecisionKind) -> Critique:
    return Critique(
        supported_findings=[],
        challenged_findings=[],
        missing_evidence=[],
        requested_follow_up=[],
        recommended_disposition=disposition,
        rationale=["synthetic critique for guard tests"],
    )


def _synthetic_source() -> SourceArtifact:
    return SourceArtifact(
        source_id="src_guard_test",
        canonical_path=Path("guard-test.txt"),
        sha256="0" * 64,
        source_format="txt",
        size_bytes=1,
        modified_at=datetime.now(UTC),
    )


def _pending_review(case_id: str) -> ReviewRequest:
    source = _synthetic_source()
    return ReviewRequest(
        review_id="rev_guard_test",
        case_id=case_id,
        status="PENDING",
        reasons=["policy trigger"],
        amount=None,
        source=source,
        evidence_bundle={},
        agent_recommendation=DecisionKind.HOLD,
        agent_rationale=["awaiting human review"],
        critic=_critique(DecisionKind.HOLD),
        questions=[],
        created_at=datetime.now(UTC),
    )


def _blocking_inventory() -> InventoryComparison:
    return InventoryComparison(
        sku="SKU-WIDGET-A",
        raw_items=["WidgetA"],
        requested_quantity=Decimal("20"),
        available_stock=15,
        status=InventoryStatus.EXCEEDS_STOCK,
        explanation="stock comparison",
    )


def _persist_blocking_review(invoice_dir: Path, settings: Settings) -> ReviewRequest:
    case_id, _started_at = _prepare(invoice_dir, settings)
    store = WorkflowStore(settings.workflow_db)
    risk = _risk(["blocking evidence requires review"]).model_copy(
        update={"inventory": [_blocking_inventory()]}, deep=True
    )
    return create_review_request(
        case_id,
        store.load_extraction(case_id),
        risk,
        _critique(DecisionKind.HOLD),
        DecisionKind.HOLD,
        ["blocking evidence requires review"],
        store,
    )


def test_review_package_persists_typed_blocking_evidence(
    invoice_dir: Path, settings: Settings
) -> None:
    review = _persist_blocking_review(invoice_dir, settings)
    expected = [
        {
            "blocker_id": "inventory:SKU-WIDGET-A:EXCEEDS_STOCK",
            "kind": "inventory",
            "evidence_id": "SKU-WIDGET-A",
            "description": "inventory EXCEEDS_STOCK: WidgetA requested=20 stock=15",
        }
    ]
    assert review.evidence_bundle["blocking_evidence"] == expected
    stored = WorkflowStore(settings.workflow_db).load_review(review.review_id)
    assert stored.evidence_bundle["blocking_evidence"] == expected


def test_review_decision_persists_explicit_blocker_authorization(
    invoice_dir: Path, settings: Settings
) -> None:
    review = _persist_blocking_review(invoice_dir, settings)
    blocker_ids = [
        entry["blocker_id"] for entry in review.evidence_bundle["blocking_evidence"]
    ]
    resolved = record_human_decision(
        review.review_id,
        "reviewer@example.com",
        HumanDecisionKind.APPROVE,
        "the cited stock exception is authorized",
        WorkflowStore(settings.workflow_db),
        settings.inventory_db,
        addressed_blocker_ids=blocker_ids,
    )
    assert resolved.human_decision is not None
    assert resolved.human_decision.addressed_blocker_ids == [
        "inventory:SKU-WIDGET-A:EXCEEDS_STOCK"
    ]


@pytest.mark.parametrize(
    ("decision", "addressed"),
    [
        (HumanDecisionKind.APPROVE, ["inventory:STALE:UNKNOWN"]),
        (HumanDecisionKind.REJECT, ["inventory:SKU-WIDGET-A:EXCEEDS_STOCK"]),
    ],
    ids=["unknown-id", "non-authorizing-decision"],
)
def test_review_service_rejects_invalid_blocker_authorization(
    decision: HumanDecisionKind,
    addressed: list[str],
    invoice_dir: Path,
    settings: Settings,
) -> None:
    review = _persist_blocking_review(invoice_dir, settings)
    with pytest.raises(InvoiceAgentsError) as excinfo:
        record_human_decision(
            review.review_id,
            "reviewer@example.com",
            decision,
            "submitted authorization",
            WorkflowStore(settings.workflow_db),
            settings.inventory_db,
            addressed_blocker_ids=addressed,
        )
    assert excinfo.value.stop_reason == "BLOCKER_AUTHORIZATION_INVALID"


@pytest.mark.asyncio
async def test_max_messages_maps_to_incomplete(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pin the real AutoGen phrasing; an upgrade that changes it must fail here.
    pinned_phrase = await _pinned_max_messages_phrase()
    assert is_max_messages_stop(pinned_phrase) is True
    # The old over-broad 'max' substring must NOT classify unrelated stops.
    assert is_max_messages_stop("Handoff to human_reviewer") is False
    assert is_max_messages_stop("max tool iterations") is False

    case_id, started_at = _prepare(invoice_dir, settings)
    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: StubModelClient())
    monkeypatch.setattr(
        orchestration,
        "build_team",
        lambda _context, _client: FakeTeam(TaskResult(messages=[], stop_reason=pinned_phrase)),
    )
    monkeypatch.chdir(tmp_path)
    result = await run_prepared_case(case_id, started_at, settings)
    assert result.status is CaseStatus.INCOMPLETE
    assert result.stop_reason == "MAX_MESSAGES_EXHAUSTED"
    assert result.final_decision is None
    assert result.payment is None


def test_unresolved_review_blocks_final_decision() -> None:
    risk = _risk(["amount exceeds review threshold"])
    critique = _critique(DecisionKind.HOLD)

    with pytest.raises(InvoiceAgentsError) as no_review:
        validate_final_decision(
            DecisionKind.HOLD, False, risk, critique, None, case_id="case_guard"
        )
    assert no_review.value.stop_reason == "HUMAN_REVIEW_UNRESOLVED"

    with pytest.raises(InvoiceAgentsError) as pending_review:
        validate_final_decision(
            DecisionKind.HOLD,
            False,
            risk,
            critique,
            _pending_review("case_guard"),
            case_id="case_guard",
        )
    assert pending_review.value.stop_reason == "HUMAN_REVIEW_UNRESOLVED"


def test_critic_disagreement_blocks_approve() -> None:
    risk = _risk([])
    for disposition in (DecisionKind.HOLD, DecisionKind.REJECT):
        with pytest.raises(InvoiceAgentsError) as blocked:
            validate_final_decision(
                DecisionKind.APPROVE,
                True,
                risk,
                _critique(disposition),
                None,
                case_id="case_guard",
            )
        assert blocked.value.stop_reason == "CRITIC_DISAGREEMENT_UNRESOLVED"
    # Critic agreement with APPROVE raises nothing.
    validate_final_decision(
        DecisionKind.APPROVE,
        True,
        risk,
        _critique(DecisionKind.APPROVE),
        None,
        case_id="case_guard",
    )


def test_locked_database_fails_visibly(workflow_db: Path) -> None:
    store = WorkflowStore(workflow_db)
    # comparison_results.case_id has a foreign key: seed the case so the post-release
    # write can succeed and the locked-phase failure is attributable to the lock alone.
    source = _synthetic_source()
    store.register_source(source)
    store.create_case("case_x", source, datetime.now(UTC))
    blocker = sqlite3.connect(workflow_db)
    try:
        blocker.execute("BEGIN EXCLUSIVE")

        # Each blocked call below waits out the 5s busy timeout; that is expected.
        with pytest.raises(sqlite3.OperationalError, match="locked") as save_error:
            store.save_comparison("case_x", "inventory", {"a": 1})
        record = _error_record(save_error.value)
        assert record.category == "DATABASE"
        assert record.stop_reason == "DATABASE_ERROR"
        assert "NOT_FOUND" not in record.stop_reason

        with pytest.raises(DatabaseVerificationError) as verify_error:
            verify_database(workflow_db, DatabaseKind.WORKFLOW)
        assert verify_error.value.category == "DATABASE"
        assert verify_error.value.stop_reason == "DATABASE_VERIFICATION_ERROR"
        assert "NOT_FOUND" not in str(verify_error.value)
    finally:
        blocker.rollback()
        blocker.close()

    # Once the exclusive lock is released both operations succeed.
    assert store.save_comparison("case_x", "inventory", {"a": 1}).startswith("cmp_")
    verified = verify_database(workflow_db, DatabaseKind.WORKFLOW)
    assert verified["integrity"] == "ok"


@pytest.mark.asyncio
async def test_malformed_provider_response_categorized(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("POST", "https://api.x.ai/v1/chat/completions")
    validation_error = openai.APIResponseValidationError(
        response=httpx.Response(200, request=request),
        body=None,
        message="response validation sentinel",
    )
    record = _error_record(validation_error)
    assert record.category == "PROVIDER"
    assert record.stop_reason == "PROVIDER_RESPONSE_INVALID"
    assert "response validation sentinel" in record.message

    with pytest.raises(json.JSONDecodeError) as decode_error:
        json.loads("{not json")
    decode_record = _error_record(decode_error.value)
    assert decode_record.category == "SCHEMA"
    assert decode_record.stop_reason == "RESPONSE_DECODE_FAILED"
    assert decode_record.message == str(decode_error.value)

    case_id, started_at = _prepare(invoice_dir, settings)
    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: StubModelClient())
    monkeypatch.setattr(
        orchestration, "build_team", lambda _context, _client: FakeTeam(error=validation_error)
    )
    monkeypatch.chdir(tmp_path)
    result = await run_prepared_case(case_id, started_at, settings)
    assert result.status is CaseStatus.FAILED
    assert result.errors[0].category == "PROVIDER"
    assert result.errors[0].stop_reason == "PROVIDER_RESPONSE_INVALID"
    assert "response validation sentinel" in result.errors[0].message


def test_payment_ledger_write_failure(invoice_dir: Path, settings: Settings) -> None:
    store = WorkflowStore(settings.workflow_db)
    case_id, _ = _prepare(invoice_dir, settings)
    invoice = store.load_extraction(case_id)
    store.save_final_decision(
        case_id,
        FinalDecision(
            decision=DecisionKind.APPROVE,
            reasons=["test"],
            critic_disposition=DecisionKind.APPROVE,
            payment_eligible=True,
        ),
    )
    os.chmod(settings.workflow_db, stat.S_IREAD)
    try:
        with pytest.raises(sqlite3.OperationalError):
            mock_payment(case_id, invoice, store, settings.workflow_db)
    finally:
        os.chmod(settings.workflow_db, stat.S_IWRITE | stat.S_IREAD)
    # The failed write recorded nothing: no PAID row may exist.
    with connect_database(settings.workflow_db, read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM payments").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_preflight_failure_writes_result_json(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings_missing_key = Settings(
        xai_api_key=None,
        inventory_db=settings.inventory_db,
        workflow_db=settings.workflow_db,
    )
    result = await process_invoice(invoice_dir / "invoice_1001.txt", settings_missing_key)
    assert result.status is CaseStatus.FAILED
    result_path = tmp_path / "artifacts" / "results" / f"{result.case_id}.json"
    assert result_path.exists()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["status"] == "FAILED"
    assert payload["stop_reason"] == "PROVIDER_PREFLIGHT_FAILED"


@pytest.mark.asyncio
async def test_retry_events_count_into_usage(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepare(invoice_dir, settings)
    recorder = AuditRecorder(settings.workflow_db, case_id)
    recorder.record("provider.retry", {"message": "retry 1"})
    recorder.record("provider.retry", {"message": "retry 2"})

    pinned_phrase = await _pinned_max_messages_phrase()
    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: StubModelClient())
    monkeypatch.setattr(
        orchestration,
        "build_team",
        lambda _context, _client: FakeTeam(TaskResult(messages=[], stop_reason=pinned_phrase)),
    )
    monkeypatch.chdir(tmp_path)
    result = await run_prepared_case(case_id, started_at, settings)
    assert result.usage.retries == 2
    assert WorkflowStore(settings.workflow_db).count_events(case_id, "provider.retry") == 2


def test_synthetic_fixture_sources_fail_visibly(tmp_path: Path) -> None:
    with pytest.raises(SourceEvidenceError) as corrupt_pdf_error:
        snapshot_source(FIXTURES_DIR / "corrupt.pdf", tmp_path / "sources", 10_485_760)
    assert corrupt_pdf_error.value.category == "TOOL"
    assert corrupt_pdf_error.value.stop_reason == "PDF_WORKER_FAILED"

    with pytest.raises(SourceEvidenceError) as malformed_json_error:
        source = snapshot_source(FIXTURES_DIR / "malformed.json", tmp_path / "sources", 10_485_760)
        extract_invoice_evidence(source)
    assert malformed_json_error.value.category == "PARSE"
    assert malformed_json_error.value.stop_reason == "JSON_PARSE_FAILED"

    with pytest.raises(SourceEvidenceError) as empty_error:
        source = snapshot_source(FIXTURES_DIR / "empty.txt", tmp_path / "sources", 10_485_760)
        extract_invoice_evidence(source)
    assert empty_error.value.category == "PARSE"
    assert empty_error.value.stop_reason == "SOURCE_EMPTY"


@pytest.mark.parametrize(
    "raw_total",
    [
        "100.00 Bearer secret-provider-credential",
        "100.00 sk-proj-sensitive-provider-credential",
        "100.00 xai-sensitive-provider-credential",
    ],
)
def test_malformed_money_error_preserves_safe_evidence_context(
    raw_total: str, tmp_path: Path
) -> None:
    submitted = tmp_path / "malformed-money.txt"
    submitted.write_text(
        "\n".join(
            (
                "INVOICE",
                "Vendor: Numeric Supplies",
                "Invoice Number: INV-4242",
                "Date: 2026-01-15",
                "Due Date: 2026-02-01",
                "WidgetA qty: 2 unit price: $50.00 $100.00",
                f"Total: {raw_total}",
            )
        ),
        encoding="utf-8",
    )
    source = snapshot_source(submitted, tmp_path / "sources", 10_485_760)

    with pytest.raises(SourceEvidenceError) as excinfo:
        extract_invoice_evidence(source)

    error = _error_record(excinfo.value)
    assert error.category == "PARSE"
    assert error.stop_reason == "MALFORMED_MONEY_FIELD"
    assert error.details["field"] == "declared total"
    assert error.details["locator"] == "line:7"
    assert error.details["source_id"] == source.source_id
    assert error.details["raw_value"] == "[REDACTED]"
    assert "secret-provider-credential" not in error.message
    assert "secret-provider-credential" not in str(error.details)

    bad_database = tmp_path / "not_a_database.db"
    bad_database.write_bytes(b"not a sqlite database\n")
    with pytest.raises(DatabaseVerificationError) as bad_db_error:
        verify_database(bad_database, DatabaseKind.WORKFLOW)
    assert bad_db_error.value.stop_reason == "DATABASE_SIGNATURE_INVALID"
