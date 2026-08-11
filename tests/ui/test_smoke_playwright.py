"""Playwright browser smokes, one per UI delivery-phase exit criterion.

Opt-in like the live suite: set RUN_UI_SMOKE=1 and install browsers first with
`uv run playwright install chromium`. Model runs stay stubbed - the smokes prove
the browser flows, not the paid pipeline.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

import pytest
from factories import make_pending_review_case, make_succeeded_case

from invoice_agents.config import Settings
from invoice_agents.db.store import ExecutionClaim, WorkflowStore
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

    async def fake_run(
        case_id: str,
        started_at: datetime,
        run_settings: Settings,
        *,
        claim: ExecutionClaim | None = None,
    ) -> CaseResult:
        assert claim is not None
        store = WorkflowStore(run_settings.workflow_db)
        invoice = store.load_current_extraction(claim)
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

    async def fake_resume(
        case_id: str,
        run_settings: Settings,
        *,
        claim: ExecutionClaim | None = None,
    ) -> CaseResult:
        assert claim is not None
        store = WorkflowStore(run_settings.workflow_db)
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
    config = uvicorn.Config(
        create_app(
            settings,
            allowed_hosts=("127.0.0.1",),
            allowed_origins=(f"http://127.0.0.1:{port}",),
        ),
        host="127.0.0.1",
        port=port,
        log_level="error",
    )
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


def _csrf_diagnostics(
    page: Any,
    server_url: str,
    request: Any,
    hidden_tokens: list[str],
) -> dict[str, Any]:
    """Describe CSRF metadata without exposing cookie or token values."""

    headers = request.all_headers()
    content_type = headers.get("content-type", "")
    post_data = request.post_data or ""
    if content_type.startswith("application/x-www-form-urlencoded"):
        posted_tokens = [value for name, value in parse_qsl(post_data) if name == "csrf_token"]
    else:
        marker = 'name="csrf_token"\r\n\r\n'
        posted_tokens = [part.split("\r\n", 1)[0] for part in post_data.split(marker)[1:]]
    session_cookies = [
        cookie
        for cookie in page.context.cookies([server_url])
        if cookie["name"] == "galatiq_session"
    ]
    cookie_metadata = [
        {
            "name": cookie["name"],
            "domain": cookie["domain"],
            "path": cookie["path"],
            "sameSite": cookie["sameSite"],
            "secure": cookie["secure"],
        }
        for cookie in session_cookies
    ]
    combined_token_count = len(posted_tokens) + int("x-csrf-token" in headers)
    posted_tokens_match_hidden = (
        len(posted_tokens) == 1
        and posted_tokens[0] in hidden_tokens
        and len(set(hidden_tokens)) == 1
    )
    origin = headers.get("origin")
    referer = headers.get("referer")
    if origin != server_url and not (origin is None and referer and referer.startswith(server_url)):
        inferred_reject = "strict same-origin metadata"
    elif "galatiq_session=" not in headers.get("cookie", ""):
        inferred_reject = "signed session cookie absent from request"
    elif combined_token_count != 1:
        inferred_reject = "submitted CSRF token multiplicity"
    elif not posted_tokens_match_hidden:
        inferred_reject = "submitted CSRF token differs from rendered token"
    else:
        inferred_reject = "session signature or server-side token validation"
    return {
        "origin": origin,
        "referer": referer,
        "host": headers.get("host"),
        "content_type": content_type,
        "hidden_token_count": len(hidden_tokens),
        "posted_csrf_field_count": len(posted_tokens),
        "csrf_header_present": "x-csrf-token" in headers,
        "posted_tokens_match_hidden": posted_tokens_match_hidden,
        "request_session_cookie_present": "galatiq_session=" in headers.get("cookie", ""),
        "session_cookies": cookie_metadata,
        "inferred_reject": inferred_reject,
    }


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


def test_sse_client_deduplicates_exact_decimal_event_ids_without_losing_precision(
    page: Any,
) -> None:
    case_id = "case-browser-cursor"
    last_event_id_headers: list[list[str]] = []
    request_lock = threading.Lock()

    def case_event(event_id: str, event_type: str, payload_seq: int) -> str:
        payload = json.dumps(
            {
                "seq": payload_seq,
                "event_type": event_type,
                "created_at": "2026-08-10T12:00:00Z",
            },
            separators=(",", ":"),
        )
        return f"id: {event_id}\nevent: case-event\ndata: {payload}\n\n"

    first_response = "retry: 1000\n\n" + "".join(
        [
            case_event("9", "cursor.nine", 9_007_199_254_740_993),
            case_event("10", "cursor.ten", 9),
            case_event(
                "9007199254740992",
                "cursor.safe-boundary-plus-one",
                9_007_199_254_740_993,
            ),
            case_event(
                "9007199254740993",
                "cursor.adjacent-above-safe-boundary",
                9_007_199_254_740_992,
            ),
        ]
    )
    terminal_event = (
        "event: terminal\n"
        'data: {"case_id":"case-browser-cursor","status":"SUCCEEDED",'
        '"stop_reason":"CURSOR_TEST_COMPLETE"}\n\n'
    )
    second_response = "".join(
        [
            case_event(
                "9007199254740992",
                "cursor.stale-duplicate",
                9_223_372_036_854_775_807,
            ),
            case_event(
                "9007199254740993",
                "cursor.exact-duplicate",
                9_223_372_036_854_775_806,
            ),
            case_event("9223372036854775807", "cursor.sqlite-rowid-maximum", 10),
            terminal_event,
            terminal_event,
        ]
    )

    app_script = (
        Path(__file__).resolve().parents[2] / "src/invoice_agents/ui/static/app.js"
    ).read_bytes()
    live_page = (
        "<!doctype html><html><body>"
        f'<section data-live-events data-case-id="{case_id}">'
        '<p id="stream-note" class="pulsing"></p>'
        '<div id="terminal-host"></div>'
        '<ul id="live-timeline"></ul>'
        "</section>"
        '<script src="/app.js"></script>'
        "</body></html>"
    ).encode()

    class CursorServerHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            self.close_connection = True

        def do_GET(self) -> None:
            if self.path == "/live":
                self._send(live_page, "text/html; charset=utf-8")
                return
            if self.path == "/app.js":
                self._send(app_script, "text/javascript; charset=utf-8")
                return
            if self.path == f"/cases/{case_id}/events":
                with request_lock:
                    last_event_id_headers.append(
                        self.headers.get_all("Last-Event-ID", failobj=[])
                    )
                    response_number = len(last_event_id_headers)
                if response_number > 2:
                    self.send_error(409, "terminal must prevent a third SSE request")
                    return
                body = first_response if response_number == 1 else second_response
                self._send(body.encode(), "text/event-stream")
                return
            self.send_error(404)

        def log_message(self, _format: str, *args: object) -> None:
            del args

    cursor_server = ThreadingHTTPServer(("127.0.0.1", 0), CursorServerHandler)
    cursor_thread = threading.Thread(target=cursor_server.serve_forever)
    cursor_thread.start()
    cursor_url = f"http://127.0.0.1:{cursor_server.server_port}"
    page.add_init_script(
        """
        (() => {
          const NativeEventSource = window.EventSource;
          window.nativeEventSourceInstances = [];
          window.EventSource = new Proxy(NativeEventSource, {
            construct(target, argumentsList) {
              const source = Reflect.construct(target, argumentsList);
              window.nativeEventSourceInstances.push(source);
              return source;
            }
          });
        })();
        """
    )
    try:
        page.goto(f"{cursor_url}/live")
        page.wait_for_function("window.nativeEventSourceInstances.length === 1")
        page.evaluate(
            """
            () => {
              window.reconnectedEventStates = [];
              window.terminalBannerEffects = 0;
              new MutationObserver((records) => {
                records.forEach((record) => {
                  record.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE &&
                        node.matches(".terminal-banner")) {
                      window.terminalBannerEffects += 1;
                    }
                  });
                });
              }).observe(document.getElementById("terminal-host"), { childList: true });
              window.nativeEventSourceInstances[0].addEventListener("case-event", (event) => {
                window.reconnectedEventStates.push({
                  id: event.lastEventId,
                  readyState: event.currentTarget.readyState
                });
              });
            }
            """
        )
        page.wait_for_selector("#terminal-host > .terminal-banner")

        rows = page.locator("#live-timeline > li")
        assert rows.count() == 5
        assert rows.locator(".dim").all_inner_texts() == [
            "cursor.nine",
            "cursor.ten",
            "cursor.safe-boundary-plus-one",
            "cursor.adjacent-above-safe-boundary",
            "cursor.sqlite-rowid-maximum",
        ]
        with request_lock:
            assert last_event_id_headers == [[], ["9007199254740993"]]
        assert page.evaluate(
            "JSON.parse('{\"seq\":9007199254740993}').seq === 9007199254740992"
        )
        assert page.evaluate("window.nativeEventSourceInstances.length") == 1
        assert page.evaluate(
            "Object.prototype.toString.call(window.nativeEventSourceInstances[0])"
        ) == "[object EventSource]"
        assert page.evaluate(
            """
            window.reconnectedEventStates.find(
              (event) => event.id === "9223372036854775807"
            ).readyState
            """
        ) == 1
        assert page.locator("#terminal-host > .terminal-banner").count() == 1
        assert page.locator("#terminal-host").get_by_text("CURSOR_TEST_COMPLETE").count() == 1
        assert page.evaluate("window.terminalBannerEffects") == 1
        assert page.evaluate("window.nativeEventSourceInstances[0].readyState") == 2
    finally:
        cursor_server.shutdown()
        cursor_server.server_close()
        cursor_thread.join(timeout=5)
    assert not cursor_thread.is_alive()


def test_u1_decide_reject_and_resume_in_browser(
    page: Any, server_url: str, settings: Settings
) -> None:
    console_messages: list[str] = []
    page.on("console", lambda message: console_messages.append(f"{message.type}: {message.text}"))
    case_id, review = make_pending_review_case(settings)
    page.goto(f"{server_url}/reviews/{review.review_id}")
    hidden_tokens = page.locator("input[name='csrf_token']").evaluate_all(
        "nodes => nodes.map(node => node.value)"
    )
    page.fill("#reviewer", "vp@example.com")
    page.check("input[name='decision'][value='REJECT']")
    page.fill("#reason", "Requested quantity is not authorized.")
    with page.expect_response(
        lambda response: (
            response.request.method == "POST"
            and response.url.endswith(f"/reviews/{review.review_id}/decision")
        )
    ) as post_info:
        page.click("button:has-text('Record decision')")
    post = post_info.value
    assert post.status == 303, (
        f"decision POST status={post.status} final_url={page.url!r} "
        f"body={post.text()[:1000]!r} console={console_messages!r} "
        f"csrf={_csrf_diagnostics(page, server_url, post.request, hidden_tokens)!r}"
    )
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
    console_messages: list[str] = []
    page.on("console", lambda message: console_messages.append(f"{message.type}: {message.text}"))
    loaded = page.goto(server_url + "/submit")
    assert loaded is not None
    hidden_tokens = page.locator("input[name='csrf_token']").evaluate_all(
        "nodes => nodes.map(node => node.value)"
    )
    page.check("input[name='existing'][value='invoice_1001.txt']")
    with page.expect_response(
        lambda response: response.request.method == "POST" and response.url.endswith("/submit")
    ) as post_info:
        page.click("button:has-text('Process invoice')")
    post = post_info.value
    assert post.status == 303, (
        f"submit POST status={post.status} final_url={page.url!r} "
        f"body={post.text()[:1000]!r} console={console_messages!r} "
        f"csrf={_csrf_diagnostics(page, server_url, post.request, hidden_tokens)!r}"
    )
    request_headers = post.request.all_headers()
    assert request_headers["origin"] == server_url
    assert request_headers["referer"] == server_url + "/submit"
    assert loaded.headers["referrer-policy"] == "same-origin"
    page.wait_for_url("**/live")
    assert page.url.endswith("/live")
    page.wait_for_selector(".terminal-banner", timeout=15000)
    assert "STUB_RUN_RECORDED" in page.inner_text(".terminal-banner")


def test_recovery_error_event_closes_eventsource_without_retry(
    page: Any,
    server_url: str,
    settings: Settings,
) -> None:
    case_id = make_succeeded_case(settings)
    event_requests = 0
    payload = json.dumps(
        {
            "case_id": case_id,
            "status": "UNAVAILABLE",
            "stop_reason": "EXECUTION_RECOVERY_FAILED",
            "recovery_verified": False,
        }
    )

    def recovery_error(route: Any) -> None:
        nonlocal event_requests
        event_requests += 1
        route.fulfill(
            status=200,
            content_type="text/event-stream",
            body=f"retry: 10\nevent: recovery-error\ndata: {payload}\n\n",
        )

    page.route(f"**/cases/{case_id}/events", recovery_error)
    page.goto(f"{server_url}/cases/{case_id}/live")
    page.wait_for_selector(".terminal-banner", timeout=2_000)
    assert "Execution recovery unavailable" in page.inner_text(".terminal-banner")
    page.wait_for_timeout(250)
    assert event_requests == 1


def _assert_narrow_document_is_contained(page: Any) -> None:
    dimensions = page.evaluate(
        """() => ({
          documentClientWidth: document.documentElement.clientWidth,
          documentScrollWidth: document.documentElement.scrollWidth,
          navLinksContained: Array.from(document.querySelectorAll('.topbar nav a')).every(
            (link) => {
              const rect = link.getBoundingClientRect();
              return rect.left >= 0 && rect.right <= document.documentElement.clientWidth;
            }
          )
        })"""
    )
    assert dimensions["documentScrollWidth"] == dimensions["documentClientWidth"]
    assert dimensions["navLinksContained"] is True

    table = page.locator(".table-wrap").first
    table_dimensions = table.evaluate(
        """(node) => ({
          clientWidth: node.clientWidth,
          scrollWidth: node.scrollWidth,
          right: node.getBoundingClientRect().right,
          documentClientWidth: document.documentElement.clientWidth
        })"""
    )
    assert table_dimensions["right"] <= table_dimensions["documentClientWidth"]
    assert table_dimensions["scrollWidth"] > table_dimensions["clientWidth"]


def _tab_to_href(page: Any, href: str, *, limit: int = 40) -> None:
    for _ in range(limit):
        page.keyboard.press("Tab")
        focused = page.evaluate(
            """() => ({
              tag: document.activeElement && document.activeElement.tagName,
              href: document.activeElement && document.activeElement.getAttribute('href'),
              rowLink: document.activeElement && document.activeElement.hasAttribute('data-row-link')
            })"""
        )
        if focused["href"] == href:
            assert focused == {"tag": "A", "href": href, "rowLink": True}
            return
    raise AssertionError(f"Tab did not reach semantic row link {href!r}")


def test_320px_dashboard_and_reviews_contain_wide_tables_and_literal_statuses(
    page: Any, server_url: str, settings: Settings
) -> None:
    make_succeeded_case(settings)
    make_pending_review_case(settings)
    page.set_viewport_size({"width": 320, "height": 720})

    page.goto(server_url + "/")
    _assert_narrow_document_is_contained(page)
    assert page.locator("#case-table").get_by_text("SUCCEEDED", exact=True).is_visible()

    page.goto(server_url + "/reviews")
    _assert_narrow_document_is_contained(page)
    assert page.locator("table.data").get_by_text("PENDING", exact=True).is_visible()


def test_tab_and_enter_activate_case_and_review_row_links_natively(
    page: Any, server_url: str, settings: Settings
) -> None:
    case_id = make_succeeded_case(settings)
    _, review = make_pending_review_case(settings)

    page.goto(server_url + "/")
    case_href = f"/cases/{case_id}"
    _tab_to_href(page, case_href)
    page.keyboard.press("Enter")
    page.wait_for_url(f"**{case_href}")

    page.goto(server_url + "/reviews")
    review_href = f"/reviews/{review.review_id}"
    _tab_to_href(page, review_href)
    page.keyboard.press("Enter")
    page.wait_for_url(f"**{review_href}")


def test_j_and_k_move_real_focus_between_row_links_before_native_enter(
    page: Any, server_url: str, settings: Settings
) -> None:
    make_succeeded_case(settings, "invoice_1001.txt")
    make_succeeded_case(settings, "invoice_1004.json")
    page.goto(server_url + "/")
    links = page.locator("#case-table tbody a[data-row-link]")
    assert links.count() == 2
    hrefs = [links.nth(index).get_attribute("href") for index in range(2)]

    page.keyboard.press("j")
    assert page.evaluate("document.activeElement.matches('a[data-row-link]')") is True
    assert page.evaluate("document.activeElement.getAttribute('href')") == hrefs[0]
    focus_style = links.nth(0).evaluate(
        """(node) => ({
          style: getComputedStyle(node).outlineStyle,
          width: getComputedStyle(node).outlineWidth
        })"""
    )
    assert focus_style["style"] != "none"
    assert focus_style["width"] != "0px"

    page.keyboard.press("j")
    assert page.evaluate("document.activeElement.getAttribute('href')") == hrefs[1]
    page.keyboard.press("k")
    assert page.evaluate("document.activeElement.getAttribute('href')") == hrefs[0]

    assert hrefs[0] is not None
    page.keyboard.press("Enter")
    page.wait_for_url(f"**{hrefs[0]}")


def test_clicking_a_secondary_row_link_does_not_trigger_the_primary_row_link(
    page: Any, server_url: str, settings: Settings
) -> None:
    case_id, review = make_pending_review_case(settings)
    page.goto(server_url + "/reviews")

    row = page.locator(f"tr[data-href='/reviews/{review.review_id}']")
    row.locator(f"a[href='/cases/{case_id}']").click()

    page.wait_for_url(f"**/cases/{case_id}")


def test_row_click_never_synthesizes_its_primary_link_from_hostile_descendants(
    page: Any, server_url: str, settings: Settings
) -> None:
    make_succeeded_case(settings)
    page.goto(server_url + "/")
    page.evaluate(
        """() => {
          const row = document.querySelector('#case-table tbody tr[data-href]');
          const primary = row.querySelector('a[data-row-link]');
          window.primaryRowLinkClicks = 0;
          primary.addEventListener('click', (event) => {
            window.primaryRowLinkClicks += 1;
            event.preventDefault();
          });

          const cell = document.createElement('td');
          cell.innerHTML = `
            <a href="#secondary" data-hostile="anchor">secondary</a>
            <button type="button" data-hostile="button">button</button>
            <input type="text" data-hostile="input" value="input">
            <select data-hostile="select"><option>option</option></select>
            <textarea data-hostile="textarea">textarea</textarea>
            <label data-hostile="label">label</label>
            <form><span data-hostile="form-child">form child</span></form>
            <details open>
              <summary data-hostile="summary">summary</summary>
              <span data-hostile="details-child">details child</span>
            </details>
            <audio controls data-hostile="audio"></audio>
            <video controls data-hostile="video"></video>
            <span contenteditable data-hostile="contenteditable-empty">editable empty</span>
            <span contenteditable="plaintext-only" data-hostile="contenteditable-plaintext">editable text</span>
            <span contenteditable="false" data-hostile="contenteditable-false">editable attribute</span>
            <span tabindex="-1" data-hostile="tabindex">programmatic target</span>
            <span role="switch" data-hostile="role-switch">switch</span>
            <span role="link" data-hostile="role-link">role link</span>
          `;
          row.appendChild(cell);
        }"""
    )

    hostile_kinds = (
        "anchor",
        "button",
        "input",
        "select",
        "textarea",
        "label",
        "form-child",
        "summary",
        "details-child",
        "audio",
        "video",
        "contenteditable-empty",
        "contenteditable-plaintext",
        "contenteditable-false",
        "tabindex",
        "role-switch",
        "role-link",
    )
    unguarded: list[str] = []
    for kind in hostile_kinds:
        if kind == "details-child":
            page.locator("details").evaluate("node => { node.open = true; }")
        before = page.evaluate("window.primaryRowLinkClicks")
        descendant = page.locator(f"[data-hostile='{kind}']")
        descendant.click(force=True)
        if kind in {"audio", "video"}:
            descendant.dispatch_event("click")
        after = page.evaluate("window.primaryRowLinkClicks")
        if after != before:
            unguarded.append(kind)
    assert not unguarded, f"unguarded hostile descendants: {unguarded!r}"
