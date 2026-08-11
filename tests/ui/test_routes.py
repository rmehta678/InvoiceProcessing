"""Route tests over an ephemeral migrated workflow DB; model runs are stubbed.

Mutations exercise the real services (record_human_decision, prepare_invoice);
only run_prepared_case / resume_case - the paid model boundary - are replaced.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

import pytest
from factories import (
    FIXTURE_DIR,
    make_failed_case,
    make_pending_review_case,
    make_succeeded_case,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from markupsafe import escape

from invoice_agents.agents.decision_rules import unaddressed_blockers
from invoice_agents.config import Settings
from invoice_agents.db.core import connect_database
from invoice_agents.db.store import ExecutionClaim, WorkflowStore
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.models import CaseResult, CaseStatus, RiskAssessment
from invoice_agents.observability.audit import AuditRecorder
from invoice_agents.ui import queries


def wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition was not reached in time")


class BlockerAuthorizationControls(HTMLParser):
    """Collect real blocker checkbox names under each rendered decision group."""

    def __init__(self) -> None:
        super().__init__()
        self.active_decision: str | None = None
        self.names_by_decision: dict[str, list[str]] = {}
        self.disabled_by_decision: dict[str, list[bool]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "fieldset" and attributes.get("data-blocker-authorization-for"):
            self.active_decision = str(attributes["data-blocker-authorization-for"])
            self.names_by_decision[self.active_decision] = []
            self.disabled_by_decision[self.active_decision] = []
        elif (
            tag == "input"
            and self.active_decision is not None
            and attributes.get("type") == "checkbox"
            and attributes.get("name")
        ):
            self.names_by_decision[self.active_decision].append(str(attributes["name"]))
            self.disabled_by_decision[self.active_decision].append("disabled" in attributes)

    def handle_endtag(self, tag: str) -> None:
        if tag == "fieldset" and self.active_decision is not None:
            self.active_decision = None


def blocker_control_names(html: str) -> dict[str, list[str]]:
    parser = BlockerAuthorizationControls()
    parser.feed(html)
    return parser.names_by_decision


def stub_runs(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    """Replace the model-run boundary with an instant stored SUCCEEDED result."""

    async def fake_run(
        case_id: str,
        started_at: datetime,
        settings: Settings,
        *,
        claim: ExecutionClaim | None = None,
    ) -> CaseResult:
        calls.append(case_id)
        store = WorkflowStore(settings.workflow_db)
        invoice = store.load_extraction(case_id)
        assert claim is not None
        result = CaseResult(
            case_id=case_id,
            source_id=invoice.source.source_id,
            status=CaseStatus.SUCCEEDED,
            stop_reason="STUB_RUN_RECORDED",
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )
        store.finish_case(result, claim)
        return result

    monkeypatch.setattr("invoice_agents.ui.runs.run_prepared_case", fake_run)


# --------------------------------------------------------------------------- dashboard


def test_dashboard_empty_state(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "No cases yet" in response.text
    assert "invoice-agents process" in response.text
    assert "schema v5, integrity ok" in response.text


def test_dashboard_lists_stored_case(client: TestClient, settings: Settings) -> None:
    case_id = make_succeeded_case(settings)
    response = client.get("/")
    assert response.status_code == 200
    assert "INV-1001" in response.text
    assert "Widgets Inc." in response.text
    assert "5,000.00 USD" in response.text
    assert "SUCCEEDED" in response.text
    assert "APPROVE" in response.text
    assert "PAID" in response.text
    assert f"/cases/{case_id}" in response.text


def test_dashboard_filters_and_search(client: TestClient, settings: Settings) -> None:
    make_succeeded_case(settings)
    make_pending_review_case(settings)
    everything = client.get("/").text
    assert "INV-1001" in everything and "1002" in everything
    only_pending = client.get("/?status=NEEDS_HUMAN").text
    assert "INV-1001" not in only_pending
    searched = client.get("/?q=widgets").text
    assert "INV-1001" in searched and "Gadgets" not in searched


def test_dashboard_htmx_returns_fragment(client: TestClient, settings: Settings) -> None:
    make_succeeded_case(settings)
    response = client.get("/?status=SUCCEEDED", headers={"HX-Request": "true"})
    assert response.status_code == 200
    assert "<html" not in response.text
    assert "case-table-region" in response.text
    assert "INV-1001" in response.text


def test_missing_database_aborts_startup_without_creating_storage(
    settings: Settings, ui_workdir: Path
) -> None:
    from invoice_agents.ui.server import create_app

    missing_database = ui_workdir / "missing-workflow.db"
    broken = Settings(
        xai_api_key="test-only-not-a-real-key",
        inventory_db=settings.inventory_db,
        workflow_db=missing_database,
    )
    with pytest.raises(InvoiceAgentsError) as error, TestClient(create_app(broken)):
        pass
    assert error.value.stop_reason == "DATABASE_MISSING"
    assert not missing_database.exists()


# ------------------------------------------------------------------------- case detail


def test_case_detail_renders_full_narrative(client: TestClient, settings: Settings) -> None:
    case_id = make_succeeded_case(settings)
    response = client.get(f"/cases/{case_id}")
    assert response.status_code == 200
    text = response.text
    assert "APPROVED_PAYMENT_RECORDED" in text
    assert "Raw" in text and "Normalized" in text
    assert text.index(">Raw<") < text.index(">Normalized<"), "raw column comes first"
    assert "SKU-WIDGET-A" in text
    assert "Reconciliations not performed" in text
    assert "vendor master reconciliation unavailable" in text
    assert "mock" in text, "the payment section always carries the word mock"
    assert "Event timeline" in text
    assert "case.prepared" in text
    assert "Usage:" in text


def test_case_detail_missing_case_is_404(client: TestClient) -> None:
    response = client.get("/cases/case_does_not_exist")
    assert response.status_code == 404
    assert "CASE_NOT_FOUND" in response.text


def test_failed_case_shows_errors_at_top(client: TestClient, settings: Settings) -> None:
    case_id = make_failed_case(settings)
    response = client.get(f"/cases/{case_id}")
    assert response.status_code == 200
    text = response.text
    assert "PROVIDER_TIMEOUT" in text
    assert "provider request exceeded the configured timeout" in text
    assert "req_fixture_123" in text
    assert text.index("provider request exceeded") < text.index("Extraction"), (
        "error records render before the evidence narrative"
    )


def test_ui_exception_handler_sanitizes_embedded_credentials(client: TestClient) -> None:
    secret = "plainUiCredential"

    async def sensitive_error() -> None:
        raise InvoiceAgentsError(
            category="PROVIDER",
            message=f"provider exception AUTHORIZATION: Bearer {secret}",
            stop_reason="PROVIDER_REQUEST_FAILED",
            provider_request_id="raw provider body round7-request-marker",
        )

    client.app.add_api_route("/_task13-sensitive-error", sensitive_error)
    response = client.get("/_task13-sensitive-error")

    assert response.status_code == 400
    assert "[REDACTED]" in response.text
    assert secret not in response.text
    assert "round7-request-marker" not in response.text
    assert "provider request ID" not in response.text


def test_case_detail_displays_matching_tool_call_rows_without_secrets(
    client: TestClient, settings: Settings
) -> None:
    case_id = make_succeeded_case(settings)
    secret = "plainUiEventCredential"
    audit = AuditRecorder(settings.workflow_db, case_id)
    audit.record(
        "autogen.ToolCallRequestEvent",
        {"content": [{"name": "lookup", "arguments": f"api_key={secret}"}]},
        tool_call_id="call_shared_1",
    )
    audit.record(
        "autogen.ToolCallExecutionEvent",
        {"content": [{"name": "lookup", "content": "inventory matched"}]},
        tool_call_id="call_shared_1",
    )

    response = client.get(f"/cases/{case_id}")

    assert response.status_code == 200
    assert response.text.count("call_shared_1") == 2
    assert "ToolCallRequestEvent" in response.text
    assert "ToolCallExecutionEvent" in response.text
    assert "[REDACTED]" in response.text
    assert secret not in response.text


def test_needs_human_case_links_review(client: TestClient, settings: Settings) -> None:
    case_id, review = make_pending_review_case(settings)
    response = client.get(f"/cases/{case_id}")
    assert response.status_code == 200
    assert "HUMAN_REVIEW_REQUESTED" in response.text
    assert f"/reviews/{review.review_id}" in response.text
    assert "waiting for a human decision" in response.text


def test_case_result_json_requires_a_durable_artifact_binding(
    client: TestClient, settings: Settings, ui_workdir: Path
) -> None:
    case_id = make_succeeded_case(settings)
    missing = client.get(f"/cases/{case_id}/result.json")
    assert missing.status_code == 409
    assert missing.json() == {
        "case_id": case_id,
        "status": "SUCCEEDED",
        "stop_reason": "RESULT_ARTIFACT_BINDING_UNRESOLVED",
    }
    artifact_dir = ui_workdir / "artifacts" / "results"
    artifact_dir.mkdir(parents=True)
    still_unbound = client.get(f"/cases/{case_id}/result.json")
    assert still_unbound.status_code == 409
    assert still_unbound.json() == missing.json()
    store = WorkflowStore(settings.workflow_db)
    result = store.load_result(case_id)
    assert result is not None
    (artifact_dir / f"{case_id}.json").write_text(result.model_dump_json(), encoding="utf-8")
    response = client.get(f"/cases/{case_id}/result.json")
    assert response.status_code == 409
    assert response.json()["stop_reason"] == "RESULT_ARTIFACT_BINDING_UNRESOLVED"
    assert client.get("/cases/../secrets/result.json").status_code in {400, 404}


@pytest.mark.parametrize("corruption", ["authority", "result", "binding"])
def test_case_result_json_propagates_persisted_corruption_to_error_boundary(
    corruption: str,
    client: TestClient,
    settings: Settings,
) -> None:
    case_id = make_succeeded_case(settings)
    with connect_database(settings.workflow_db) as connection:
        if corruption in {"authority", "result"}:
            triggers = connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'trigger' AND tbl_name = 'cases'"
            ).fetchall()
            for trigger in triggers:
                connection.execute(f'DROP TRIGGER "{trigger["name"]}"')
            if corruption == "authority":
                connection.execute(
                    "UPDATE cases SET execution_token = 'exec_not_canonical' "
                    "WHERE case_id = ?",
                    (case_id,),
                )
            else:
                connection.execute(
                    "UPDATE cases SET result_json = '{\"corrupt\":true}' WHERE case_id = ?",
                    (case_id,),
                )
        else:
            generation = connection.execute(
                "SELECT execution_generation FROM cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()["execution_generation"]
            connection.execute("PRAGMA ignore_check_constraints = ON")
            connection.execute(
                "INSERT INTO result_artifact_bindings("
                "case_id, execution_generation, artifact_sha256, artifact_device, "
                "artifact_inode, artifact_file_type, artifact_size_bytes) "
                "VALUES (?, ?, 'not-a-sha256', 1, 1, 32768, 1)",
                (case_id, generation),
            )
            connection.execute("PRAGMA ignore_check_constraints = OFF")
        connection.commit()

    response = client.get(f"/cases/{case_id}/result.json")

    assert response.status_code == 400
    assert "PERSISTED_RESULT_INVALID" in response.text
    assert "RESULT_ARTIFACT_BINDING_DURABILITY_UNRESOLVED" not in response.text


def test_case_result_json_propagates_unrelated_database_failure_to_error_boundary(
    client: TestClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = make_succeeded_case(settings)
    loader_calls: list[str] = []

    def fail_unrelated_database_read(
        _store: WorkflowStore,
        _case_id: str,
    ) -> tuple[CaseResult | None, int, object | None]:
        loader_calls.append(_case_id)
        raise sqlite3.OperationalError("unrelated result database sentinel")

    monkeypatch.setattr(
        WorkflowStore,
        "load_result_with_artifact_binding",
        fail_unrelated_database_read,
    )

    response = client.get(f"/cases/{case_id}/result.json")

    assert loader_calls == [case_id]
    assert response.status_code == 500
    assert "DATABASE_ERROR" in response.text
    assert "unrelated result database sentinel" in response.text
    assert "RESULT_ARTIFACT_BINDING_DURABILITY_UNRESOLVED" not in response.text


def test_case_result_json_maps_only_recognized_binding_ambiguity_to_409(
    client: TestClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = make_succeeded_case(settings)
    loader_calls: list[str] = []

    def fail_with_recognized_binding_ambiguity(
        _store: WorkflowStore,
        _case_id: str,
    ) -> tuple[CaseResult | None, int, object | None]:
        loader_calls.append(_case_id)
        raise InvoiceAgentsError(
            ErrorCategory.DATABASE,
            "result-artifact binding transaction outcome could not be proven",
            case_id=case_id,
            stop_reason="RESULT_ARTIFACT_BINDING_DURABILITY_UNRESOLVED",
        )

    monkeypatch.setattr(
        WorkflowStore,
        "load_result_with_artifact_binding",
        fail_with_recognized_binding_ambiguity,
    )

    response = client.get(f"/cases/{case_id}/result.json")

    assert loader_calls == [case_id]
    assert response.status_code == 409
    assert response.json() == {
        "case_id": case_id,
        "status": "INCOMPLETE",
        "stop_reason": "RESULT_ARTIFACT_BINDING_DURABILITY_UNRESOLVED",
    }


# ----------------------------------------------------------------------------- reviews


def test_review_queue_pending_and_all(client: TestClient, settings: Settings) -> None:
    case_id, review = make_pending_review_case(settings)
    pending = client.get("/reviews")
    assert pending.status_code == 200
    assert review.review_id[:10] in pending.text
    assert "PENDING" in pending.text
    assert case_id[:10] in pending.text
    assert "15,000.00 USD" in pending.text
    everything = client.get("/reviews?all=1")
    assert review.review_id[:10] in everything.text


def test_review_queue_empty_state(client: TestClient) -> None:
    response = client.get("/reviews")
    assert response.status_code == 200
    assert "No pending reviews" in response.text


def test_review_detail_shows_package_and_unbiased_form(
    client: TestClient, settings: Settings
) -> None:
    _, review = make_pending_review_case(settings)
    response = client.get(f"/reviews/{review.review_id}")
    assert response.status_code == 200
    text = response.text
    assert (
        "Do the source evidence, normalized values, and calculated deltas support payment?" in text
    )
    for reason in review.reasons:
        assert str(escape(reason)) in text, "every review reason renders verbatim"
    assert "EXCEEDS_STOCK" in text
    assert 'type="radio" name="decision"' in text
    assert "checked" not in text, "no decision may be preselected"
    for kind in (
        "APPROVE",
        "REJECT",
        "REQUEST_CORRECTION",
        "ESTABLISH_MAPPING",
        "SUPERSEDE_REVISION",
    ):
        assert kind in text
    assert "Forces final decision REJECT" in text
    assert "Record decision" in text
    blockers = review.evidence_bundle["blocking_evidence"]
    assert blockers
    for blocker in blockers:
        assert blocker["blocker_id"] in text
        assert blocker["description"] in text
    for kind in ("APPROVE", "ESTABLISH_MAPPING", "SUPERSEDE_REVISION"):
        assert f'data-blocker-authorization-for="{kind}"' in text
    for kind in ("REJECT", "REQUEST_CORRECTION"):
        assert f'data-blocker-authorization-for="{kind}"' not in text


def test_switching_authorizing_decisions_uses_distinct_blocker_control_names(
    client: TestClient, settings: Settings
) -> None:
    _, review = make_pending_review_case(settings)
    response = client.get(f"/reviews/{review.review_id}")
    assert response.status_code == 200
    names_by_decision = blocker_control_names(response.text)
    assert set(names_by_decision) == {
        "APPROVE",
        "ESTABLISH_MAPPING",
        "SUPERSEDE_REVISION",
    }
    one_name_per_decision = {
        decision: set(names) for decision, names in names_by_decision.items()
    }
    assert all(len(names) == 1 for names in one_name_per_decision.values())
    assert len({next(iter(names)) for names in one_name_per_decision.values()}) == 3


def test_blocker_controls_render_disabled_except_for_selected_decision(
    client: TestClient, settings: Settings
) -> None:
    _, review = make_pending_review_case(settings)
    initial = client.get(f"/reviews/{review.review_id}")
    initial_controls = BlockerAuthorizationControls()
    initial_controls.feed(initial.text)
    assert all(
        all(disabled) for disabled in initial_controls.disabled_by_decision.values()
    )

    rerender = client.post(
        f"/reviews/{review.review_id}/decision",
        data={"reviewer": "vp@example.com", "decision": "APPROVE", "reason": "   "},
    )
    assert rerender.status_code == 400
    rerendered_controls = BlockerAuthorizationControls()
    rerendered_controls.feed(rerender.text)
    assert not any(rerendered_controls.disabled_by_decision["APPROVE"])
    assert all(rerendered_controls.disabled_by_decision["ESTABLISH_MAPPING"])
    assert all(rerendered_controls.disabled_by_decision["SUPERSEDE_REVISION"])


def test_review_detail_missing_is_404(client: TestClient) -> None:
    response = client.get("/reviews/rev_missing")
    assert response.status_code == 404
    assert "REVIEW_NOT_FOUND" in response.text


# ------------------------------------------------------------------- decision recording


def test_decision_requires_selection(client: TestClient, settings: Settings) -> None:
    _, review = make_pending_review_case(settings)
    response = client.post(
        f"/reviews/{review.review_id}/decision",
        data={"reviewer": "vp@example.com", "reason": "text"},
    )
    assert response.status_code == 400
    assert "Select a decision" in response.text
    assert WorkflowStore(settings.workflow_db).load_review(review.review_id).status == "PENDING"


def test_decision_missing_reason_rejected(client: TestClient, settings: Settings) -> None:
    _, review = make_pending_review_case(settings)
    response = client.post(
        f"/reviews/{review.review_id}/decision",
        data={"reviewer": "vp@example.com", "decision": "REJECT", "reason": "   "},
    )
    assert response.status_code == 400
    assert "reviewer and reason are required" in response.text
    assert WorkflowStore(settings.workflow_db).load_review(review.review_id).status == "PENDING"


def test_authorizing_decision_records_selected_blocker_ids(
    client: TestClient, settings: Settings
) -> None:
    _, review = make_pending_review_case(settings)
    blocker_ids = [
        entry["blocker_id"] for entry in review.evidence_bundle["blocking_evidence"]
    ]
    assert blocker_ids
    detail = client.get(f"/reviews/{review.review_id}")
    selected_field = blocker_control_names(detail.text)["APPROVE"][0]
    response = client.post(
        f"/reviews/{review.review_id}/decision",
        data={
            "reviewer": "vp@example.com",
            "decision": "APPROVE",
            "reason": "explicitly authorizing the cited blockers",
            selected_field: blocker_ids,
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    stored = WorkflowStore(settings.workflow_db).load_review(review.review_id)
    assert stored.human_decision is not None
    assert stored.human_decision.addressed_blocker_ids == blocker_ids


def test_inactive_blocker_selections_cannot_complete_selected_decision_authorization(
    client: TestClient, settings: Settings
) -> None:
    case_id, review = make_pending_review_case(settings)
    blocker_ids = [
        entry["blocker_id"] for entry in review.evidence_bundle["blocking_evidence"]
    ]
    assert blocker_ids
    selected_ids = blocker_ids[:-1]
    detail = client.get(f"/reviews/{review.review_id}")
    names_by_decision = blocker_control_names(detail.text)
    selected_field = names_by_decision["APPROVE"][0]
    inactive_field = names_by_decision["ESTABLISH_MAPPING"][0]
    data: dict[str, str | list[str]] = {
        "reviewer": "vp@example.com",
        "decision": "APPROVE",
        "reason": "only the visible selected blockers are authorized",
    }
    if selected_field == inactive_field:
        data[selected_field] = [*blocker_ids, *selected_ids]
    else:
        data[selected_field] = selected_ids
        data[inactive_field] = blocker_ids

    response = client.post(
        f"/reviews/{review.review_id}/decision", data=data, follow_redirects=False
    )

    assert response.status_code == 303
    store = WorkflowStore(settings.workflow_db)
    stored = store.load_review(review.review_id)
    assert stored.human_decision is not None
    assert stored.human_decision.addressed_blocker_ids == selected_ids
    risk = RiskAssessment.model_validate(store.load_comparison(case_id, "risk"))
    assert unaddressed_blockers(risk, stored.human_decision)


def test_mapping_decision_requires_mapping(client: TestClient, settings: Settings) -> None:
    _, review = make_pending_review_case(settings)
    response = client.post(
        f"/reviews/{review.review_id}/decision",
        data={
            "reviewer": "vp@example.com",
            "decision": "ESTABLISH_MAPPING",
            "reason": "mapping the alias",
        },
    )
    assert response.status_code == 400
    assert "requires at least one explicit mapping" in response.text


def test_mapping_fields_submitted_with_reject_reach_service_and_are_rejected(
    client: TestClient, settings: Settings
) -> None:
    _, review = make_pending_review_case(
        settings, source=FIXTURE_DIR / "invoice_2001_bulk_alias.txt"
    )
    before = WorkflowStore(settings.workflow_db).load_review(review.review_id)

    response = client.post(
        f"/reviews/{review.review_id}/decision",
        data={
            "reviewer": "vp@example.com",
            "decision": "REJECT",
            "reason": "a stale browser retained mapping inputs",
            "mapping_raw": ["WidgetA (bulk)"],
            "mapping_sku": ["SKU-WIDGET-A"],
        },
    )

    assert response.status_code == 400
    assert "mappings are permitted only for ESTABLISH_MAPPING" in response.text
    assert WorkflowStore(settings.workflow_db).load_review(review.review_id) == before
    with connect_database(settings.inventory_db, read_only=True) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM item_aliases WHERE alias_normalized = ?", ("widgetabulk",)
        ).fetchone()[0] == 0


def test_mapping_dropdown_is_only_a_hint_and_unknown_raw_evidence_is_rejected(
    client: TestClient, settings: Settings
) -> None:
    _, review = make_pending_review_case(
        settings, source=FIXTURE_DIR / "invoice_2001_bulk_alias.txt"
    )
    before = WorkflowStore(settings.workflow_db).load_review(review.review_id)

    response = client.post(
        f"/reviews/{review.review_id}/decision",
        data={
            "reviewer": "vp@example.com",
            "decision": "ESTABLISH_MAPPING",
            "reason": "a forged raw item must not be authorized by the dropdown SKU",
            "mapping_raw": ["Invented browser alias"],
            "mapping_sku": ["SKU-WIDGET-A"],
        },
    )

    assert response.status_code == 400
    assert "unresolved inventory evidence" in response.text
    assert WorkflowStore(settings.workflow_db).load_review(review.review_id) == before
    with connect_database(settings.inventory_db, read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM item_aliases").fetchone()[0] == 0


def test_reject_decision_recorded_and_resumable(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_id, review = make_pending_review_case(settings)
    response = client.post(
        f"/reviews/{review.review_id}/decision",
        data={
            "reviewer": "vp@example.com",
            "decision": "REJECT",
            "reason": "Requested quantity is not authorized.",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == f"/reviews/{review.review_id}?decided=1"
    assert unquote(client.cookies.get("ui_reviewer") or "") == "vp@example.com"

    stored = WorkflowStore(settings.workflow_db).load_review(review.review_id)
    assert stored.status == "RESOLVED"
    assert stored.human_decision is not None
    assert stored.human_decision.decision == "REJECT"
    assert stored.human_decision.reviewer == "vp@example.com"
    assert stored.human_decision.reason == "Requested quantity is not authorized."

    decided_page = client.get(f"/reviews/{review.review_id}?decided=1")
    assert "Decision recorded." in decided_page.text
    assert "Resume case now" in decided_page.text

    resumed: list[str] = []

    async def fake_resume(
        resume_case_id: str,
        resume_settings: Settings,
        *,
        claim: ExecutionClaim | None = None,
    ) -> CaseResult:
        resumed.append(resume_case_id)
        store = WorkflowStore(resume_settings.workflow_db)
        assert claim is not None
        result = store.load_result(resume_case_id)
        assert result is not None
        finished = result.model_copy(
            update={
                "status": CaseStatus.SUCCEEDED,
                "stop_reason": "DECISION_REJECT",
                "finished_at": datetime.now(UTC),
            },
            deep=True,
        )
        store.finish_case(finished, claim)
        return finished

    monkeypatch.setattr("invoice_agents.ui.runs.resume_case", fake_resume)
    resume_response = client.post(f"/cases/{case_id}/resume", follow_redirects=False)
    assert resume_response.status_code == 303
    assert resume_response.headers["location"] == "/reviews"
    wait_for(lambda: resumed == [case_id])
    wait_for(
        lambda: (
            WorkflowStore(settings.workflow_db).load_result(case_id).status is CaseStatus.SUCCEEDED
        )
    )


def test_resolved_review_rejects_second_decision(client: TestClient, settings: Settings) -> None:
    _, review = make_pending_review_case(settings)
    first = client.post(
        f"/reviews/{review.review_id}/decision",
        data={"reviewer": "vp@example.com", "decision": "REJECT", "reason": "no"},
        follow_redirects=False,
    )
    assert first.status_code == 303
    second = client.post(
        f"/reviews/{review.review_id}/decision",
        data={"reviewer": "other@example.com", "decision": "APPROVE", "reason": "yes"},
    )
    assert second.status_code == 400
    assert "already resolved" in second.text
    stored = WorkflowStore(settings.workflow_db).load_review(review.review_id)
    assert stored.human_decision is not None
    assert stored.human_decision.decision == "REJECT", "the first decision stands"


def test_mapping_decision_writes_alias_and_resolves(client: TestClient, settings: Settings) -> None:
    _, review = make_pending_review_case(
        settings, source=FIXTURE_DIR / "invoice_2001_bulk_alias.txt"
    )
    page = client.get(f"/reviews/{review.review_id}")
    assert "WidgetA (bulk)" in page.text, "unresolved raw items feed the mapping rows"
    assert "SKU-WIDGET-A" in page.text, "SKU dropdown is backed by real inventory rows"
    response = client.post(
        f"/reviews/{review.review_id}/decision",
        data={
            "reviewer": "vp@example.com",
            "decision": "ESTABLISH_MAPPING",
            "reason": "bulk WidgetA is the same physical item",
            "mapping_raw": ["WidgetA (bulk)"],
            "mapping_sku": ["SKU-WIDGET-A"],
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    stored = WorkflowStore(settings.workflow_db).load_review(review.review_id)
    assert stored.status == "RESOLVED"
    assert stored.human_decision is not None
    assert [(mapping.raw_item, mapping.sku) for mapping in stored.human_decision.mappings] == [
        ("WidgetA (bulk)", "SKU-WIDGET-A")
    ]
    with connect_database(settings.inventory_db, read_only=True) as connection:
        row = connection.execute(
            "SELECT sku, approved_by, source FROM item_aliases WHERE alias_normalized = ?",
            ("widgetabulk",),
        ).fetchone()
    assert row is not None
    assert row["sku"] == "SKU-WIDGET-A"
    assert row["approved_by"] == "vp@example.com"
    assert review.review_id in row["source"]


def test_supersede_requires_prior_case(client: TestClient, settings: Settings) -> None:
    _, review = make_pending_review_case(settings)
    response = client.post(
        f"/reviews/{review.review_id}/decision",
        data={
            "reviewer": "vp@example.com",
            "decision": "SUPERSEDE_REVISION",
            "reason": "this is a revision",
        },
    )
    assert response.status_code == 400
    assert "requires a superseded case ID" in response.text


# -------------------------------------------------------------------------------- resume


def test_resume_rejects_non_needs_human_case(client: TestClient, settings: Settings) -> None:
    case_id = make_succeeded_case(settings)
    response = client.post(f"/cases/{case_id}/resume")
    assert response.status_code == 409
    assert "CASE_NOT_RESUMABLE" in response.text


def test_resume_requires_recorded_decision(client: TestClient, settings: Settings) -> None:
    case_id, _ = make_pending_review_case(settings)
    response = client.post(f"/cases/{case_id}/resume")
    assert response.status_code == 409
    assert "HUMAN_DECISION_MISSING" in response.text


# ------------------------------------------------------------------------ submit & batch


def test_submit_existing_runs_in_background(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    stub_runs(monkeypatch, calls)
    response = client.post(
        "/submit",
        data={
            "submission_id": "submission_route_single_existing",
            "existing": "invoice_1001.txt",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("/cases/") and location.endswith("/live")
    case_id = location.split("/")[2]
    wait_for(lambda: calls == [case_id])
    wait_for(
        lambda: (
            (result := WorkflowStore(settings.workflow_db).load_result(case_id)) is not None
            and result.stop_reason == "STUB_RUN_RECORDED"
        )
    )
    live = client.get(location)
    assert live.status_code == 200
    assert case_id in live.text


def test_submit_upload_lands_in_uploads_dir(
    client: TestClient,
    settings: Settings,
    ui_workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    stub_runs(monkeypatch, calls)
    content = (ui_workdir / "data" / "invoices" / "invoice_1001.txt").read_bytes()
    response = client.post(
        "/submit",
        data={"submission_id": "submission_route_single_upload"},
        files={"upload": ("uploaded_invoice.txt", content, "text/plain")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    saved = ui_workdir / "data" / "invoices" / "uploads" / "uploaded_invoice.txt"
    assert saved.is_file()
    assert saved.read_bytes() == content
    case_id = response.headers["location"].split("/")[2]
    header = queries.case_header(settings.workflow_db, case_id)
    assert header is not None


def test_submit_rejects_upload_one_byte_over_limit_without_partial_file(
    client: TestClient,
    settings: Settings,
    ui_workdir: Path,
) -> None:
    content = (ui_workdir / "data" / "invoices" / "invoice_1001.txt").read_bytes()
    settings.source_max_bytes = len(content) - 1

    response = client.post(
        "/submit",
        data={"submission_id": "submission_route_oversized_upload"},
        files={"upload": ("oversized.txt", content, "text/plain")},
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert "SOURCE_TOO_LARGE" in response.text
    upload_dir = ui_workdir / "data" / "invoices" / "uploads"
    assert upload_dir.is_dir()
    assert list(upload_dir.iterdir()) == []


def test_submit_rejects_unknown_and_missing_choice(client: TestClient) -> None:
    missing = client.post(
        "/submit",
        data={"submission_id": "submission_route_unknown", "existing": "nope.txt"},
    )
    assert missing.status_code == 404
    assert "SOURCE_NOT_FOUND" in missing.text
    nothing = client.post(
        "/submit",
        data={"submission_id": "submission_route_missing_choice"},
    )
    assert nothing.status_code == 400
    assert "choose at least one invoice file or upload one" in nothing.text


def test_submit_multiple_existing_runs_as_batch(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    stub_runs(monkeypatch, calls)
    response = client.post(
        "/submit",
        data={
            "submission_id": "submission_route_multiple",
            "existing": ["invoice_1001.txt", "invoice_1002.txt"],
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    batch_url = response.headers["location"]
    assert batch_url.startswith("/batches/")
    page = client.get(batch_url)
    assert page.status_code == 200
    assert "invoice_1001.txt" in page.text
    assert "invoice_1002.txt" in page.text
    rows_url = f"{batch_url}/rows"
    wait_for(lambda: client.get(rows_url).status_code == 286, timeout=10.0)
    assert sorted(calls) == sorted(set(calls)) and len(calls) == 2, (
        "both selected files ran exactly once"
    )


def test_submit_multiple_with_unknown_name_starts_nothing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    stub_runs(monkeypatch, calls)
    response = client.post(
        "/submit",
        data={
            "submission_id": "submission_route_multiple_unknown",
            "existing": ["invoice_1001.txt", "nope.txt"],
        },
    )
    assert response.status_code == 404
    assert "SOURCE_NOT_FOUND" in response.text
    assert calls == [], "resolution fails atomically before any run starts"


def test_submit_traversal_is_rejected(client: TestClient, ui_workdir: Path) -> None:
    secret = ui_workdir / "secret.txt"
    secret.write_text("outside the invoice dir", encoding="utf-8")
    response = client.post(
        "/submit",
        data={
            "submission_id": "submission_route_traversal",
            "existing": "../../secret.txt",
        },
    )
    assert response.status_code == 404


def test_batch_runs_matrix_until_terminal(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    stub_runs(monkeypatch, calls)
    response = client.post(
        "/batch",
        data={"submission_id": "submission_route_batch", "concurrency": "1"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    batch_url = response.headers["location"]
    page = client.get(batch_url)
    assert page.status_code == 200
    assert "invoice_1001.txt" in page.text, "the matrix is server-rendered, one row per file"
    assert "invoice_1002.txt" in page.text

    rows_url = f"{batch_url}/rows"

    def finished() -> bool:
        return client.get(rows_url).status_code == 286

    wait_for(finished, timeout=10.0)
    final = client.get(rows_url)
    assert final.status_code == 286, "HTTP 286 stops the htmx poll"
    assert "STUB_RUN_RECORDED" in final.text
    assert "SUCCEEDED" in final.text
    assert len(calls) == 2, "both prepared files ran exactly once"


@pytest.mark.parametrize(
    ("value", "expected_status"),
    [("0", 400), ("-1", 400), ("9", 400), ("true", 422), ("1.5", 422)],
)
def test_batch_rejects_invalid_concurrency_before_registry_mutation(
    client: TestClient,
    app: FastAPI,
    value: str,
    expected_status: int,
) -> None:
    registry = app.state.registry
    assert registry._batches == {}

    response = client.post(
        "/batch",
        data={"submission_id": "submission_route_invalid_batch", "concurrency": value},
        follow_redirects=False,
    )

    assert response.status_code == expected_status
    assert registry._batches == {}
    assert registry._runs == {}


def test_batch_unknown_is_404(client: TestClient) -> None:
    assert client.get("/batches/batch_missing").status_code == 404


# ----------------------------------------------------------------------------------- SSE


def test_event_stream_replays_and_terminates(client: TestClient, settings: Settings) -> None:
    case_id = make_succeeded_case(settings)
    events: list[str] = []
    payloads: list[str] = []
    with client.stream("GET", f"/cases/{case_id}/events") as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            elif line.startswith("data:"):
                payloads.append(line.split(":", 1)[1].strip())
                if events and events[-1] == "terminal":
                    break
    assert "case-event" in events
    assert events[-1] == "terminal"
    terminal = json.loads(payloads[-1])
    assert terminal["status"] == "SUCCEEDED"
    assert terminal["stop_reason"] == "APPROVED_PAYMENT_RECORDED"
    joined = "\n".join(payloads)
    assert "case.prepared" in joined


# -------------------------------------------------------------------------------- system


def test_system_page_reports_without_leaking_key(client: TestClient, settings: Settings) -> None:
    response = client.get("/system")
    assert response.status_code == 200
    text = response.text
    assert "grok-4.5" in text
    assert "https://api.x.ai/v1" in text
    assert "present" in text
    assert "test-only-not-a-real-key" not in text, "the key value must never render"
    assert "10,000.00 USD" in text
    assert str(escape('uv run pytest -m "not live"')) in text


# ------------------------------------------------------------------------------ queries


def test_queries_event_cursor_and_prior_cases(settings: Settings, ui_workdir: Path) -> None:
    case_id = make_succeeded_case(settings)
    all_events = queries.events_after(settings.workflow_db, case_id, 0)
    assert all_events, "prepare_case records audit events"
    tail = queries.events_after(settings.workflow_db, case_id, all_events[0].seq)
    assert [event.seq for event in tail] == [event.seq for event in all_events[1:]]

    other_case, _ = make_pending_review_case(settings)
    priors = queries.prior_cases_for_invoice(settings.workflow_db, "INV-1001", "case_other")
    assert [prior.case_id for prior in priors] == [case_id]
    assert priors[0].payment_status == "PAID"
    assert priors[0].declared_total == "5000.00"
    assert queries.prior_cases_for_invoice(settings.workflow_db, None, "x") == []
    assert other_case != case_id
