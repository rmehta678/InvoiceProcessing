"""Playwright browser smokes, one per UI delivery-phase exit criterion.

Opt-in like the live suite: set RUN_UI_SMOKE=1 and install browsers first with
`uv run playwright install chromium`. Model runs stay stubbed - the smokes prove
the browser flows, not the paid pipeline.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from factories import make_pending_review_case, make_succeeded_case

from invoice_agents.config import Settings
from invoice_agents.db.store import WorkflowStore
from invoice_agents.models import CaseResult, CaseStatus

pytestmark = pytest.mark.ui_smoke

playwright_sync = pytest.importorskip("playwright.sync_api", reason="playwright is not installed")

if os.getenv("RUN_UI_SMOKE") != "1":
    pytest.skip(
        "UI browser smokes NOT RUN; set RUN_UI_SMOKE=1 and install chromium",
        allow_module_level=True,
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def server_url(
    settings: Settings, ui_workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[str]:
    """The real app served by uvicorn in-process, with the model boundary stubbed."""

    import uvicorn

    from invoice_agents.ui.server import create_app

    async def fake_run(case_id: str, started_at: datetime, run_settings: Settings) -> CaseResult:
        store = WorkflowStore(run_settings.workflow_db)
        invoice = store.load_extraction(case_id)
        claim = store.claim_case_execution(
            case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
        )
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

    async def fake_resume(case_id: str, run_settings: Settings) -> CaseResult:
        store = WorkflowStore(run_settings.workflow_db)
        claim = store.claim_case_execution(
            case_id, frozenset({CaseStatus.NEEDS_HUMAN}), lease_seconds=60
        )
        previous = store.load_result(case_id)
        assert previous is not None
        result = previous.model_copy(
            update={
                "status": CaseStatus.SUCCEEDED,
                "stop_reason": "DECISION_REJECT",
                "finished_at": datetime.now(UTC),
            },
            deep=True,
        )
        store.finish_case(result, claim)
        return result

    monkeypatch.setattr("invoice_agents.ui.runs.run_prepared_case", fake_run)
    monkeypatch.setattr("invoice_agents.ui.runs.resume_case", fake_resume)

    port = _free_port()
    config = uvicorn.Config(create_app(settings), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise AssertionError("uvicorn did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def page(server_url: str) -> Iterator[Any]:
    from playwright.sync_api import Error, sync_playwright

    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Error:
            pytest.skip("chromium not installed; run `uv run playwright install chromium`")
        page = browser.new_page()
        yield page
        browser.close()


def test_u0_dashboard_and_case_detail_render_stored_state(
    page: Any, server_url: str, settings: Settings
) -> None:
    case_id = make_succeeded_case(settings)
    page.goto(server_url + "/")
    assert "Galatiq" in page.title()
    row = page.locator(f"tr[data-href='/cases/{case_id}']")
    assert row.count() == 1
    row.click()
    page.wait_for_url(f"**/cases/{case_id}")
    assert "APPROVED_PAYMENT_RECORDED" in page.content()
    assert "mock" in page.content()


def test_u1_decide_reject_and_resume_in_browser(
    page: Any, server_url: str, settings: Settings
) -> None:
    case_id, review = make_pending_review_case(settings)
    page.goto(f"{server_url}/reviews/{review.review_id}")
    page.fill("#reviewer", "vp@example.com")
    page.check("input[name='decision'][value='REJECT']")
    page.fill("#reason", "Requested quantity is not authorized.")
    page.click("button:has-text('Record decision')")
    page.wait_for_url("**?decided=1")
    assert "Decision recorded." in page.content()
    stored = WorkflowStore(settings.workflow_db).load_review(review.review_id)
    assert stored.status == "RESOLVED"
    assert stored.human_decision is not None
    assert stored.human_decision.decision == "REJECT"
    page.click("button:has-text('Resume case now')")
    page.wait_for_url("**/reviews")
    page.goto(f"{server_url}/cases/{case_id}/live")
    page.wait_for_selector(".terminal-banner", timeout=15000)
    banner = page.inner_text(".terminal-banner")
    assert "SUCCEEDED" in banner and "DECISION_REJECT" in banner


def test_u2_submit_from_browser_reaches_terminal_banner(
    page: Any, server_url: str, settings: Settings
) -> None:
    page.goto(server_url + "/submit")
    page.check("input[name='existing'][value='invoice_1001.txt']")
    page.click("button:has-text('Process invoice')")
    page.wait_for_url("**/live")
    page.wait_for_selector(".terminal-banner", timeout=15000)
    assert "STUB_RUN_RECORDED" in page.inner_text(".terminal-banner")
