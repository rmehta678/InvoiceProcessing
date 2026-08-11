"""Guard scenarios: stops, unresolved reviews, locks, and bad inputs stay loudly visible."""

import json
import os
import sqlite3
import stat
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
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
    CaseResult,
    CaseStatus,
    Critique,
    DecisionKind,
    ExtractedInvoice,
    FinalDecision,
    FinancialComparison,
    HumanDecisionKind,
    ReviewRequest,
    RiskAssessment,
    RiskPolicy,
    SourceArtifact,
)
from invoice_agents.observability.audit import AuditRecorder
from invoice_agents.orchestration import (
    _error_record,
    is_max_messages_stop,
    prepare_case,
    process_invoice,
)
from invoice_agents.orchestration import (
    _run_prepared_case_in_process as run_prepared_case,
)
from invoice_agents.payment.service import mock_payment
from invoice_agents.source_store import snapshot_source
from invoice_agents.tools.comparison import (
    InventoryReader,
    apply_mapping_evidence,
    build_risk_assessment,
    compare_inventory_evidence,
    compute_invoice_totals,
    find_prior_invoice_candidates,
)
from invoice_agents.tools.evidence import extract_invoice_evidence
from tests.support.pdf_policy import TEST_PDF_POLICY


@pytest.fixture(autouse=True)
def _forbid_unstubbed_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(_settings: Settings) -> object:
        raise AssertionError("non-live guard test reached an unstubbed provider boundary")

    monkeypatch.setattr(orchestration, "create_model_client", forbidden)


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


async def _run_prepared_with_new_claim(
    case_id: str, started_at: datetime, settings: Settings
) -> CaseResult:
    claim = WorkflowStore(settings).claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        orchestration.EXECUTION_LEASE_SECONDS,
    )
    return await run_prepared_case(case_id, started_at, settings, claim=claim)


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
        policy=RiskPolicy(
            review_threshold_amount=Decimal("10000.00"),
            review_threshold_currency="USD",
            review_threshold_effective_date=date(2026, 8, 6),
            due_date_tolerance_days=3,
        ),
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


def _extract_json_tax_invoice(
    tmp_path: Path,
    tax_fields: dict[str, str],
) -> ExtractedInvoice:
    path = tmp_path / "tax-evidence.json"
    path.write_text(
        json.dumps(
            {
                "invoice_number": "INV-4242",
                "vendor": {"name": "Tax Evidence Supplies"},
                "date": "2026-01-15",
                "due_date": "2026-02-14",
                "payment_terms": "Net 30",
                "currency": "USD",
                "line_items": [
                    {
                        "item": "WidgetA",
                        "quantity": "2",
                        "unit_price": "50.00",
                        "amount": "100.00",
                    }
                ],
                "subtotal": "100.00",
                "total": "100.00",
                **tax_fields,
            }
        ),
        encoding="utf-8",
    )
    source = snapshot_source(
        path,
        tmp_path / "sources",
        max_bytes=10_485_760,
        pdf_policy=TEST_PDF_POLICY,
    )
    return extract_invoice_evidence(source, TEST_PDF_POLICY)


def test_missing_tax_cannot_become_exact_or_approval_green(
    tmp_path: Path,
    settings: Settings,
) -> None:
    invoice = _extract_json_tax_invoice(tmp_path, {})
    financial = compute_invoice_totals(invoice)
    risk = build_risk_assessment(invoice, [], [], financial, settings)

    assert invoice.missing_fields == ["tax"]
    assert financial.calculated_tax == Decimal("0")
    assert financial.tax_recomputable is False
    assert financial.exact is False
    assert any(
        "required fields are missing: tax" in reason for reason in risk.policy_review_reasons
    )
    assert any(
        "financial evidence is incomplete" in reason for reason in risk.policy_review_reasons
    )
    with pytest.raises(InvoiceAgentsError) as blocked:
        validate_final_decision(
            DecisionKind.APPROVE,
            True,
            risk,
            _critique(DecisionKind.APPROVE),
            None,
            case_id="case_missing_tax",
        )
    assert blocked.value.stop_reason == "HUMAN_REVIEW_UNRESOLVED"


def test_explicit_zero_tax_preserves_clean_decision_path(
    tmp_path: Path,
    settings: Settings,
) -> None:
    invoice = _extract_json_tax_invoice(
        tmp_path,
        {"tax_rate": "0", "tax_amount": "0.00"},
    )
    financial = compute_invoice_totals(invoice)
    risk = build_risk_assessment(invoice, [], [], financial, settings)

    assert invoice.missing_fields == []
    assert financial.calculated_tax == Decimal("0")
    assert financial.tax_recomputable is True
    assert financial.exact is True
    assert risk.policy_review_reasons == []
    validate_final_decision(
        DecisionKind.APPROVE,
        True,
        risk,
        _critique(DecisionKind.APPROVE),
        None,
        case_id="case_explicit_zero_tax",
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
        critic_disagreement_reason=None,
        questions=["Does the evidence support this decision?"],
        created_at=datetime.now(UTC),
    )


def _persist_blocking_review(invoice_dir: Path, settings: Settings) -> ReviewRequest:
    case_id, _started_at = _prepare(invoice_dir, settings)
    store = WorkflowStore(settings)
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    invoice = store.promote_predecessor_extraction(claim)
    with connect_database(settings.inventory_db) as connection:
        connection.execute(
            "UPDATE inventory SET available_stock = ? WHERE sku = ?",
            (5, "SKU-WIDGET-A"),
        )
        connection.commit()
    mappings, comparisons, unresolved = compare_inventory_evidence(
        invoice, InventoryReader(settings.inventory_db)
    )
    invoice = apply_mapping_evidence(invoice, mappings, unresolved)
    store.save_extraction(case_id, invoice, claim)
    identity = find_prior_invoice_candidates(case_id, invoice, store)
    risk = build_risk_assessment(
        invoice, comparisons, identity, compute_invoice_totals(invoice), settings
    )
    case_critique = _critique(DecisionKind.HOLD)
    store.save_identity(
        case_id,
        [candidate.model_dump(mode="json") for candidate in identity],
        claim,
    )
    store.save_comparison(
        case_id,
        "inventory",
        {
            "comparisons": [item.model_dump(mode="json") for item in comparisons],
            "unresolved_candidates": {
                item: result.model_dump(mode="json") for item, result in unresolved.items()
            },
        },
        claim,
    )
    store.save_comparison(case_id, "risk", risk.model_dump(mode="json"), claim)
    store.save_critique(case_id, case_critique, claim)
    review = create_review_request(
        case_id,
        invoice,
        risk,
        case_critique,
        DecisionKind.HOLD,
        ["blocking evidence requires review"],
        store,
        claim,
        pdf_policy=settings.pdf_policy(),
    )
    store.release_case_execution(claim)
    return review


def test_review_package_persists_typed_blocking_evidence(
    invoice_dir: Path, settings: Settings
) -> None:
    review = _persist_blocking_review(invoice_dir, settings)
    expected = [
        {
            "blocker_id": "inventory:SKU-WIDGET-A:EXCEEDS_STOCK",
            "kind": "inventory",
            "evidence_id": "SKU-WIDGET-A",
            "description": "inventory EXCEEDS_STOCK: WidgetA requested=10 stock=5",
        }
    ]
    assert review.evidence_bundle["blocking_evidence"] == expected
    stored = WorkflowStore(settings.workflow_db).load_review(review.review_id)
    assert stored.evidence_bundle["blocking_evidence"] == expected


def test_review_decision_persists_explicit_blocker_authorization(
    invoice_dir: Path, settings: Settings
) -> None:
    review = _persist_blocking_review(invoice_dir, settings)
    blocker_ids = [entry["blocker_id"] for entry in review.evidence_bundle["blocking_evidence"]]
    resolved = record_human_decision(
        review.review_id,
        "reviewer@example.com",
        HumanDecisionKind.APPROVE,
        "the cited stock exception is authorized",
        WorkflowStore(settings),
        settings.inventory_db,
        addressed_blocker_ids=blocker_ids,
    )
    assert resolved.human_decision is not None
    assert resolved.human_decision.addressed_blocker_ids == ["inventory:SKU-WIDGET-A:EXCEEDS_STOCK"]


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
            WorkflowStore(settings),
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
    result = await _run_prepared_with_new_claim(case_id, started_at, settings)
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


def test_locked_database_fails_visibly(workflow_db: Path, inventory_db: Path) -> None:
    store = WorkflowStore(workflow_db)
    settings = Settings(workflow_db=workflow_db, inventory_db=inventory_db)
    # comparison_results.case_id has a foreign key: seed the case so the post-release
    # write can succeed and the locked-phase failure is attributable to the lock alone.
    source = _synthetic_source()
    store.register_source(source)
    store.create_case("case_x", source, datetime.now(UTC))
    claim = store.claim_case_execution(
        "case_x", frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    blocker = sqlite3.connect(workflow_db)
    try:
        blocker.execute("BEGIN EXCLUSIVE")
        blocker.execute(
            "UPDATE cases SET updated_at = 'locked-uncommitted' WHERE case_id = 'case_x'"
        )

        # Each blocked call below waits out the 5s busy timeout; that is expected.
        with pytest.raises(sqlite3.OperationalError, match="locked") as save_error:
            store.save_comparison("case_x", "inventory", {"a": 1}, claim)
        record = _error_record(save_error.value)
        assert record.category == "DATABASE"
        assert record.stop_reason == "DATABASE_ERROR"
        assert "NOT_FOUND" not in record.stop_reason

        with pytest.raises(DatabaseVerificationError) as verify_error:
            verify_database(workflow_db, DatabaseKind.WORKFLOW, settings=settings)
        assert verify_error.value.category == "DATABASE"
        assert verify_error.value.stop_reason == "DATABASE_SIDECAR_UNSUPPORTED"
        assert "NOT_FOUND" not in str(verify_error.value)
    finally:
        blocker.rollback()
        blocker.close()

    # Once the exclusive lock is released both operations succeed.
    assert store.save_comparison("case_x", "inventory", {"a": 1}, claim).startswith("cmp_")
    verified = verify_database(workflow_db, DatabaseKind.WORKFLOW, settings=settings)
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
    assert record.message == "provider response failed schema validation"
    assert "response validation sentinel" not in record.message

    with pytest.raises(json.JSONDecodeError) as decode_error:
        json.loads("{not json")
    decode_record = _error_record(decode_error.value)
    assert decode_record.category == "SCHEMA"
    assert decode_record.stop_reason == "RESPONSE_DECODE_FAILED"
    assert decode_record.message == "response JSON decoding failed"
    assert str(decode_error.value) not in decode_record.message

    case_id, started_at = _prepare(invoice_dir, settings)
    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: StubModelClient())
    monkeypatch.setattr(
        orchestration, "build_team", lambda _context, _client: FakeTeam(error=validation_error)
    )
    monkeypatch.chdir(tmp_path)
    result = await _run_prepared_with_new_claim(case_id, started_at, settings)
    assert result.status is CaseStatus.FAILED
    assert result.errors[0].category == "PROVIDER"
    assert result.errors[0].stop_reason == "PROVIDER_RESPONSE_INVALID"
    assert result.errors[0].message == "provider response failed schema validation"
    assert "response validation sentinel" not in result.model_dump_json()


def test_payment_ledger_write_failure(invoice_dir: Path, settings: Settings) -> None:
    store = WorkflowStore(settings)
    case_id, _ = _prepare(invoice_dir, settings, "invoice_1004.json")
    invoice = store.load_extraction(case_id)
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    invoice = store.promote_predecessor_extraction(claim)
    mappings, comparisons, unresolved = compare_inventory_evidence(
        invoice, InventoryReader(settings.inventory_db)
    )
    invoice = apply_mapping_evidence(invoice, mappings, unresolved)
    store.save_extraction(case_id, invoice, claim)
    identity = find_prior_invoice_candidates(case_id, invoice, store)
    risk = build_risk_assessment(
        invoice, comparisons, identity, compute_invoice_totals(invoice), settings
    )
    store.save_identity(
        case_id,
        [candidate.model_dump(mode="json") for candidate in identity],
        claim,
    )
    store.save_comparison(
        case_id,
        "inventory",
        {
            "comparisons": [item.model_dump(mode="json") for item in comparisons],
            "unresolved_candidates": {
                item: result.model_dump(mode="json") for item, result in unresolved.items()
            },
        },
        claim,
    )
    store.save_comparison(case_id, "risk", risk.model_dump(mode="json"), claim)
    store.save_critique(case_id, _critique(DecisionKind.APPROVE), claim)
    store.save_final_decision(
        case_id,
        FinalDecision(
            decision=DecisionKind.APPROVE,
            reasons=["test"],
            evidence=[reference for line in invoice.lines for reference in line.evidence[:1]],
            critic_disposition=DecisionKind.APPROVE,
            payment_eligible=True,
        ),
        claim,
    )
    os.chmod(settings.workflow_db, stat.S_IREAD)
    try:
        with pytest.raises(sqlite3.OperationalError):
            mock_payment(case_id, invoice, store, settings.workflow_db, claim)
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
    result = await _run_prepared_with_new_claim(case_id, started_at, settings)
    assert result.usage.retries == 2
    assert WorkflowStore(settings.workflow_db).count_events(case_id, "provider.retry") == 2


def test_synthetic_fixture_sources_fail_visibly(tmp_path: Path) -> None:
    with pytest.raises(SourceEvidenceError) as corrupt_pdf_error:
        snapshot_source(
            FIXTURES_DIR / "corrupt.pdf",
            tmp_path / "sources",
            10_485_760,
            pdf_policy=TEST_PDF_POLICY,
        )
    assert corrupt_pdf_error.value.category == "TOOL"
    assert corrupt_pdf_error.value.stop_reason == "PDF_WORKER_FAILED"

    with pytest.raises(SourceEvidenceError) as malformed_json_error:
        source = snapshot_source(
            FIXTURES_DIR / "malformed.json",
            tmp_path / "sources",
            10_485_760,
            pdf_policy=TEST_PDF_POLICY,
        )
        extract_invoice_evidence(source, TEST_PDF_POLICY)
    assert malformed_json_error.value.category == "PARSE"
    assert malformed_json_error.value.stop_reason == "JSON_PARSE_FAILED"

    with pytest.raises(SourceEvidenceError) as empty_error:
        source = snapshot_source(
            FIXTURES_DIR / "empty.txt",
            tmp_path / "sources",
            10_485_760,
            pdf_policy=TEST_PDF_POLICY,
        )
        extract_invoice_evidence(source, TEST_PDF_POLICY)
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
    source = snapshot_source(
        submitted,
        tmp_path / "sources",
        10_485_760,
        pdf_policy=TEST_PDF_POLICY,
    )

    with pytest.raises(SourceEvidenceError) as excinfo:
        extract_invoice_evidence(source, TEST_PDF_POLICY)

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
