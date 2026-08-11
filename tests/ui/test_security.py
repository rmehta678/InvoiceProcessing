"""Fail-closed request-security contracts for every console mutation."""

from __future__ import annotations

import secrets
from html.parser import HTMLParser
from http.cookies import SimpleCookie
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from factories import make_pending_review_case
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError
from starlette.responses import Response

from invoice_agents.config import Settings
from invoice_agents.db.store import WorkflowStore
from invoice_agents.hitl.service import record_human_decision
from invoice_agents.models import HumanDecisionKind
from invoice_agents.ui.security import (
    CSRF_SCOPE_KEY,
    SESSION_COOKIE_NAME,
    CSRFMiddleware,
    ExactTrustedHostMiddleware,
)
from invoice_agents.ui.server import create_app

Attack = Literal["hostile_origin", "missing_token", "wrong_token"]


class _MutationFormTokens(HTMLParser):
    """Collect CSRF inputs by their real form action."""

    def __init__(self) -> None:
        super().__init__()
        self.action: str | None = None
        self.meta_tokens: list[str] = []
        self.tokens: dict[str, list[str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "form":
            self.action = attributes.get("action")
        elif tag == "meta" and attributes.get("name") == "csrf-token":
            self.meta_tokens.append(attributes.get("content") or "")
        elif (
            tag == "input"
            and self.action is not None
            and attributes.get("type") == "hidden"
            and attributes.get("name") == "csrf_token"
        ):
            self.tokens.setdefault(self.action, []).append(attributes.get("value") or "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self.action = None


def _attack_request(
    client: TestClient,
    *,
    page_path: str,
    action: str,
    attack: Attack,
    data: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any], list[str]]:
    page = client.get(page_path)
    assert page.status_code == 200
    parser = _MutationFormTokens()
    parser.feed(page.text)
    rendered_tokens = parser.tokens.get(action, [])
    # Before the implementation exists, the placeholder lets the POST reach the
    # unprotected boundary and produce a meaningful RED mutation failure.
    token = rendered_tokens[0] if len(rendered_tokens) == 1 else "A" * 43
    headers = {"Origin": "http://testserver"}
    submitted = dict(data)
    if attack == "hostile_origin":
        headers["Origin"] = "https://evil.example"
        submitted["csrf_token"] = token
    elif attack == "wrong_token":
        replacement = "A" if token[0] != "A" else "B"
        submitted["csrf_token"] = replacement + token[1:]
    return headers, submitted, rendered_tokens


def _database_snapshot(settings: Settings) -> tuple[bytes, bytes]:
    return settings.workflow_db.read_bytes(), settings.inventory_db.read_bytes()


def _rendered_token(client: TestClient, *, path: str = "/submit", action: str = "/submit") -> str:
    response = client.get(path)
    assert response.status_code == 200
    parser = _MutationFormTokens()
    parser.feed(response.text)
    tokens = parser.tokens.get(action, [])
    assert len(tokens) == 1
    assert tokens[0]
    return tokens[0]


def _assert_security_headers(response: Any) -> None:
    policy = response.headers["content-security-policy"]
    assert "'unsafe-inline'" not in policy
    assert "script-src 'self'" in policy
    assert "style-src 'self'" in policy
    assert "frame-ancestors 'none'" in policy
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "same-origin"


@pytest.mark.parametrize("attack", ["hostile_origin", "missing_token", "wrong_token"])
def test_submit_rejects_csrf_attacks_before_registry_or_workflow_mutation(
    raw_client: TestClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    attack: Attack,
) -> None:
    calls: list[Path] = []

    async def forbidden_start_process(path: Path, _settings: Settings) -> str:
        calls.append(path)
        return "case_security_boundary_crossed"

    monkeypatch.setattr(app.state.registry, "start_process", forbidden_start_process)
    headers, data, rendered_tokens = _attack_request(
        raw_client,
        page_path="/submit",
        action="/submit",
        attack=attack,
        data={"existing": "invoice_1001.txt"},
    )
    before = _database_snapshot(settings)

    response = raw_client.post("/submit", data=data, headers=headers, follow_redirects=False)

    assert calls == []
    assert app.state.registry._runs == {}
    assert app.state.registry._batches == {}
    assert _database_snapshot(settings) == before
    assert response.status_code == 403
    assert len(rendered_tokens) == 1 and rendered_tokens[0]


@pytest.mark.parametrize("attack", ["hostile_origin", "missing_token", "wrong_token"])
def test_batch_rejects_csrf_attacks_before_batch_or_workflow_mutation(
    raw_client: TestClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    attack: Attack,
) -> None:
    calls: list[list[Path]] = []

    async def forbidden_start_batch(
        paths: list[Path], _settings: Settings, _concurrency: int | None
    ) -> object:
        calls.append(paths)
        return SimpleNamespace(batch_id="batch_security_boundary_crossed")

    monkeypatch.setattr(app.state.registry, "start_batch", forbidden_start_batch)
    headers, data, rendered_tokens = _attack_request(
        raw_client,
        page_path="/submit",
        action="/batch",
        attack=attack,
        data={"concurrency": "1"},
    )
    before = _database_snapshot(settings)

    response = raw_client.post("/batch", data=data, headers=headers, follow_redirects=False)

    assert calls == []
    assert app.state.registry._runs == {}
    assert app.state.registry._batches == {}
    assert _database_snapshot(settings) == before
    assert response.status_code == 403
    assert len(rendered_tokens) == 1 and rendered_tokens[0]


@pytest.mark.parametrize("attack", ["hostile_origin", "missing_token", "wrong_token"])
def test_review_decision_rejects_csrf_attacks_before_human_or_inventory_mutation(
    raw_client: TestClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    attack: Attack,
) -> None:
    _, review = make_pending_review_case(settings)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def forbidden_record(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr("invoice_agents.ui.routes.record_human_decision", forbidden_record)
    action = f"/reviews/{review.review_id}/decision"
    headers, data, rendered_tokens = _attack_request(
        raw_client,
        page_path=f"/reviews/{review.review_id}",
        action=action,
        attack=attack,
        data={
            "reviewer": "vp@example.com",
            "decision": "REJECT",
            "reason": "The evidence does not authorize payment.",
        },
    )
    before = _database_snapshot(settings)

    response = raw_client.post(action, data=data, headers=headers, follow_redirects=False)

    assert calls == []
    assert _database_snapshot(settings) == before
    assert WorkflowStore(settings.workflow_db).load_review(review.review_id).status == "PENDING"
    assert response.status_code == 403
    assert len(rendered_tokens) == 1 and rendered_tokens[0]


@pytest.mark.parametrize("attack", ["hostile_origin", "missing_token", "wrong_token"])
def test_resume_rejects_csrf_attacks_before_registry_or_workflow_mutation(
    raw_client: TestClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    attack: Attack,
) -> None:
    case_id, review = make_pending_review_case(settings)
    record_human_decision(
        review.review_id,
        "vp@example.com",
        HumanDecisionKind.REJECT,
        "The evidence does not authorize payment.",
        WorkflowStore(settings),
        settings.inventory_db,
    )
    calls: list[str] = []

    async def forbidden_start_resume(resume_case_id: str, _settings: Settings) -> object:
        calls.append(resume_case_id)
        return object()

    monkeypatch.setattr(app.state.registry, "start_resume", forbidden_start_resume)
    action = f"/cases/{case_id}/resume"
    headers, data, rendered_tokens = _attack_request(
        raw_client,
        page_path=f"/reviews/{review.review_id}?decided=1",
        action=action,
        attack=attack,
        data={},
    )
    before = _database_snapshot(settings)

    response = raw_client.post(action, data=data, headers=headers, follow_redirects=False)

    assert calls == []
    assert app.state.registry._runs == {}
    assert _database_snapshot(settings) == before
    assert response.status_code == 403
    assert len(rendered_tokens) == 1 and rendered_tokens[0]


# ---------------------------------------------------------------- sessions and token channels


def test_session_cookie_is_http_only_strict_and_token_is_stable_per_session(
    raw_client: TestClient,
) -> None:
    first = raw_client.get("/submit")
    assert first.status_code == 200
    parser = _MutationFormTokens()
    parser.feed(first.text)
    assert len(parser.tokens["/submit"]) == 1
    assert len(parser.tokens["/batch"]) == 1
    assert parser.tokens["/submit"] == parser.tokens["/batch"]
    token = parser.tokens["/submit"][0]
    assert len(token) == 43
    set_cookie = first.headers["set-cookie"].lower()
    assert f"{SESSION_COOKIE_NAME}=" in set_cookie
    assert "httponly" in set_cookie
    assert "samesite=strict" in set_cookie
    assert "path=/" in set_cookie
    assert "secure" not in set_cookie

    second = raw_client.get("/submit")
    assert _rendered_token(raw_client) == token
    assert SESSION_COOKIE_NAME not in second.headers.get("set-cookie", "")


def test_two_sessions_receive_distinct_tokens(app: FastAPI) -> None:
    with TestClient(app, base_url="http://testserver") as first:
        first_token = _rendered_token(first)
    with TestClient(app, base_url="http://testserver") as second:
        second_token = _rendered_token(second)
    assert first_token != second_token


def test_tampered_session_cookie_rejects_old_token_and_rotates_on_safe_request(
    raw_client: TestClient,
) -> None:
    token = _rendered_token(raw_client)
    cookie = raw_client.cookies.get(SESSION_COOKIE_NAME)
    assert cookie is not None
    replacement = "A" if cookie[-1] != "A" else "B"
    raw_client.cookies.clear()
    raw_client.cookies.set(SESSION_COOKIE_NAME, cookie[:-1] + replacement)

    rejected = raw_client.post(
        "/submit",
        data={"existing": "missing.txt", "csrf_token": token},
        headers={"Origin": "http://testserver"},
        follow_redirects=False,
    )
    assert rejected.status_code == 403

    rotated = _rendered_token(raw_client)
    assert rotated != token


def test_non_loopback_factory_requires_a_shared_secret_before_app_construction(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import invoice_agents.ui.server as ui_server

    construction_calls = 0

    def forbidden_app_construction(*_args: Any, **_kwargs: Any) -> FastAPI:
        nonlocal construction_calls
        construction_calls += 1
        raise AssertionError("remote configuration must fail before FastAPI construction")

    monkeypatch.setattr(ui_server, "FastAPI", forbidden_app_construction)
    local_settings = settings.model_copy(update={"ui_session_secret": None})

    with pytest.raises(ValueError, match="INVOICE_UI_SESSION_SECRET"):
        ui_server.create_app(
            local_settings,
            allowed_hosts=("testserver",),
            allowed_origins=("http://testserver",),
        )

    assert construction_calls == 0


def test_new_local_application_secret_invalidates_an_old_signed_session(
    settings: Settings,
) -> None:
    local_settings = settings.model_copy(update={"ui_session_secret": None})
    first_app = create_app(
        local_settings,
        allowed_hosts=("127.0.0.2",),
        allowed_origins=("http://127.0.0.2:8787",),
    )
    with TestClient(first_app, base_url="http://127.0.0.2:8787") as first:
        token = _rendered_token(first)
        cookie = first.cookies.get(SESSION_COOKIE_NAME)
    assert cookie is not None

    second_app = create_app(
        local_settings,
        allowed_hosts=("127.0.0.2",),
        allowed_origins=("http://127.0.0.2:8787",),
    )
    with TestClient(second_app, base_url="http://127.0.0.2:8787") as second:
        second.cookies.set(SESSION_COOKIE_NAME, cookie)
        response = second.post(
            "/submit",
            data={"existing": "missing.txt", "csrf_token": token},
            headers={"Origin": "http://127.0.0.2:8787"},
            follow_redirects=False,
        )
    assert response.status_code == 403


def test_explicit_shared_session_secret_survives_application_recreation(
    settings: Settings,
) -> None:
    shared_settings = settings.model_copy(update={"ui_session_secret": SecretStr("S" * 43)})
    first_app = create_app(
        shared_settings,
        allowed_hosts=("testserver",),
        allowed_origins=("https://testserver",),
    )
    with TestClient(first_app, base_url="https://testserver") as first:
        token = _rendered_token(first)
        cookie = first.cookies.get(SESSION_COOKIE_NAME)
    assert cookie is not None

    second_app = create_app(
        shared_settings,
        allowed_hosts=("testserver",),
        allowed_origins=("https://testserver",),
    )
    with TestClient(second_app, base_url="https://testserver") as second:
        second.cookies.set(SESSION_COOKIE_NAME, cookie)
        response = second.post(
            "/submit",
            data={
                "submission_id": "submission_security_shared_session",
                "existing": "missing.txt",
                "csrf_token": token,
            },
            headers={"Origin": "https://testserver"},
            follow_redirects=False,
        )
    assert response.status_code == 404


@pytest.mark.parametrize(
    ("allowed_hosts", "allowed_origins"),
    [
        (("127.0.0.1", "console.example"), ("http://127.0.0.1:8787",)),
        (("127.0.0.1",), ("http://127.0.0.1:8787", "http://console.example:8787")),
        (("::1",), ("http://[::1]:8787", "http://192.0.2.10:8787")),
    ],
)
def test_any_non_loopback_host_or_origin_requires_a_shared_session_secret(
    settings: Settings,
    allowed_hosts: tuple[str, ...],
    allowed_origins: tuple[str, ...],
) -> None:
    local_settings = settings.model_copy(update={"ui_session_secret": None})

    with pytest.raises(ValueError, match="INVOICE_UI_SESSION_SECRET"):
        create_app(
            local_settings,
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
        )


def test_short_explicit_session_secret_is_rejected() -> None:
    with pytest.raises(ValidationError, match="session secret"):
        Settings(ui_session_secret=SecretStr("predictable-short-secret"))


def test_factory_rechecks_a_bypassed_short_shared_session_secret(settings: Settings) -> None:
    bypassed_settings = settings.model_copy(update={"ui_session_secret": SecretStr("S" * 31)})

    with pytest.raises(ValueError, match="at least 32 bytes"):
        create_app(
            bypassed_settings,
            allowed_hosts=("testserver",),
            allowed_origins=("https://testserver",),
        )


def test_blank_session_secret_keeps_local_random_key_mode() -> None:
    assert Settings(ui_session_secret="").ui_session_secret is None


@pytest.mark.parametrize(
    ("origin", "host", "secure"),
    [
        ("http://127.0.0.1:8787", "127.0.0.1", False),
        ("https://testserver", "testserver", True),
    ],
)
def test_cookie_secure_attribute_is_derived_from_the_single_allowed_scheme(
    ui_settings: Settings,
    origin: str,
    host: str,
    secure: bool,
) -> None:
    scheme_app = create_app(
        ui_settings,
        allowed_hosts=(host,),
        allowed_origins=(origin,),
    )
    with TestClient(scheme_app, base_url=origin) as client:
        response = client.get("/submit")
    cookie = SimpleCookie()
    cookie.load(response.headers["set-cookie"])
    assert bool(cookie[SESSION_COOKIE_NAME]["secure"]) is secure


def test_mixed_http_and_https_allowed_origins_are_rejected(settings: Settings) -> None:
    with pytest.raises(ValueError, match="single scheme"):
        create_app(
            settings,
            allowed_hosts=("testserver",),
            allowed_origins=("http://testserver", "https://testserver"),
        )


@pytest.mark.parametrize(
    ("origin", "host", "secure"),
    [
        ("http://127.0.0.1:8787", "127.0.0.1", False),
        ("https://testserver", "testserver", True),
    ],
)
def test_reviewer_preference_cookie_uses_the_configured_origin_scheme(
    settings: Settings,
    ui_settings: Settings,
    origin: str,
    host: str,
    secure: bool,
) -> None:
    _, review = make_pending_review_case(settings)
    scheme_app = create_app(
        ui_settings,
        allowed_hosts=(host,),
        allowed_origins=(origin,),
    )
    action = f"/reviews/{review.review_id}/decision"
    with TestClient(scheme_app, base_url=origin) as client:
        token = _rendered_token(
            client,
            path=f"/reviews/{review.review_id}",
            action=action,
        )
        response = client.post(
            action,
            data={
                "reviewer": "vp@example.com",
                "decision": "REJECT",
                "reason": "The evidence does not authorize payment.",
                "csrf_token": token,
            },
            headers={"Origin": origin},
            follow_redirects=False,
        )
    assert response.status_code == 303
    reviewer_header = next(
        value
        for value in response.headers.get_list("set-cookie")
        if value.startswith("ui_reviewer=")
    )
    reviewer_cookie = SimpleCookie()
    reviewer_cookie.load(reviewer_header)
    assert bool(reviewer_cookie["ui_reviewer"]["secure"]) is secure


@pytest.mark.parametrize(
    ("headers", "expected_status"),
    [
        ({"Origin": "http://testserver"}, 404),
        ({"Origin": "http://testserver:80"}, 404),
        ({"Referer": "http://testserver/submit?from=review"}, 404),
        (
            {
                "Origin": "http://testserver",
                "Referer": "http://testserver/submit?from=review",
            },
            404,
        ),
        ({}, 403),
        ({"Origin": "https://testserver"}, 403),
        ({"Origin": "http://testserver:81"}, 403),
        ({"Origin": "https://evil.example"}, 403),
        ({"Referer": "https://evil.example/review"}, 403),
        (
            {"Origin": "http://testserver", "Referer": "https://evil.example/review"},
            403,
        ),
        (
            {"Origin": "https://evil.example", "Referer": "http://testserver/submit"},
            403,
        ),
        (
            {"Origin": "null", "Referer": "http://testserver/submit?from=review"},
            404,
        ),
        ({"Origin": "null"}, 403),
        ({"Origin": "null", "Referer": ""}, 403),
        ({"Origin": "null", "Referer": "not-an-origin"}, 403),
        ({"Origin": "null", "Referer": "https://evil.example/review"}, 403),
        ({"Origin": "null", "Referer": "http://testserver@evil.example/review"}, 403),
        ({"Origin": "null", "Referer": "http://testserver.evil.example/review"}, 403),
        ({"Origin": "http://testserver/path"}, 403),
        ({"Origin": "http://user@testserver"}, 403),
        ({"Origin": "http://testserver#fragment"}, 403),
        ({"Origin": "http://testserver:"}, 403),
        ({"Origin": "http://testserver?"}, 403),
        ({"Origin": "http://testserver#"}, 403),
        ({"Origin": "http://testserver https://evil.example"}, 403),
        ({"Origin": "http://testserver", "Forwarded": "host=evil.example"}, 403),
        ({"Origin": "http://testserver", "X-Forwarded-Host": "evil.example"}, 403),
        ({"Origin": "http://testserver", "X-Forwarded-Proto": "https"}, 403),
        ({"Origin": "http://testserver", "X-Forwarded-Port": "443"}, 403),
        ({"Host": "testserver:81", "Origin": "http://testserver:81"}, 403),
    ],
)
def test_origin_and_referer_evidence_is_unambiguous_and_exact(
    raw_client: TestClient,
    settings: Settings,
    headers: dict[str, str],
    expected_status: int,
) -> None:
    token = _rendered_token(raw_client)
    before = _database_snapshot(settings)
    response = raw_client.post(
        "/submit",
        data={
            "submission_id": "submission_security_origin_evidence",
            "existing": "missing.txt",
            "csrf_token": token,
        },
        headers=headers,
        follow_redirects=False,
    )
    assert response.status_code == expected_status
    assert _database_snapshot(settings) == before


def test_duplicate_origin_headers_are_rejected(raw_client: TestClient) -> None:
    token = _rendered_token(raw_client)
    response = raw_client.post(
        "/submit",
        data={"existing": "missing.txt", "csrf_token": token},
        headers=[
            ("Origin", "http://testserver"),
            ("Origin", "http://testserver"),
        ],
        follow_redirects=False,
    )
    assert response.status_code == 403


def test_duplicate_host_headers_are_rejected(raw_client: TestClient) -> None:
    response = raw_client.get(
        "/submit",
        headers=[("Host", "testserver"), ("Host", "testserver")],
    )
    assert response.status_code == 400


def test_header_only_token_supports_htmx_mutations(raw_client: TestClient) -> None:
    token = _rendered_token(raw_client)
    response = raw_client.post(
        "/submit",
        data={
            "submission_id": "submission_security_header_token",
            "existing": "missing.txt",
        },
        headers={"Origin": "http://testserver", "X-CSRF-Token": token},
        follow_redirects=False,
    )
    assert response.status_code == 404


def test_base_exposes_the_session_token_for_htmx_headers(raw_client: TestClient) -> None:
    page = raw_client.get("/submit")
    parser = _MutationFormTokens()
    parser.feed(page.text)
    assert len(parser.meta_tokens) == 1
    assert parser.meta_tokens == parser.tokens["/submit"]


def test_form_and_header_token_together_are_rejected_as_ambiguous(
    raw_client: TestClient,
    settings: Settings,
) -> None:
    token = _rendered_token(raw_client)
    before = _database_snapshot(settings)
    response = raw_client.post(
        "/submit",
        data={"existing": "missing.txt", "csrf_token": token},
        headers={"Origin": "http://testserver", "X-CSRF-Token": token},
        follow_redirects=False,
    )
    assert response.status_code == 403
    assert _database_snapshot(settings) == before


def test_duplicate_form_tokens_are_rejected(
    raw_client: TestClient,
    settings: Settings,
) -> None:
    token = _rendered_token(raw_client)
    before = _database_snapshot(settings)
    response = raw_client.post(
        "/submit",
        data={"existing": "missing.txt", "csrf_token": [token, token]},
        headers={"Origin": "http://testserver"},
        follow_redirects=False,
    )
    assert response.status_code == 403
    assert _database_snapshot(settings) == before


def test_duplicate_header_tokens_are_rejected(
    raw_client: TestClient,
    settings: Settings,
) -> None:
    token = _rendered_token(raw_client)
    before = _database_snapshot(settings)
    response = raw_client.post(
        "/submit",
        data={"existing": "missing.txt"},
        headers=[
            ("Origin", "http://testserver"),
            ("X-CSRF-Token", token),
            ("X-CSRF-Token", token),
        ],
        follow_redirects=False,
    )
    assert response.status_code == 403
    assert _database_snapshot(settings) == before


@pytest.mark.parametrize("malformed", ["", "short", "A" * 42, "A" * 44, "A" * 42 + "!", "é" * 43])
def test_malformed_tokens_are_rejected(
    raw_client: TestClient,
    settings: Settings,
    malformed: str,
) -> None:
    _rendered_token(raw_client)
    before = _database_snapshot(settings)
    response = raw_client.post(
        "/submit",
        data={"existing": "missing.txt", "csrf_token": malformed},
        headers={"Origin": "http://testserver"},
        follow_redirects=False,
    )
    assert response.status_code == 403
    assert _database_snapshot(settings) == before


def test_query_string_token_is_not_accepted(
    raw_client: TestClient,
    settings: Settings,
) -> None:
    token = _rendered_token(raw_client)
    before = _database_snapshot(settings)
    response = raw_client.post(
        f"/submit?csrf_token={token}",
        data={"existing": "missing.txt"},
        headers={"Origin": "http://testserver"},
        follow_redirects=False,
    )
    assert response.status_code == 403
    assert _database_snapshot(settings) == before


# ---------------------------------------------------------------- hosts, headers, and body ordering


def test_untrusted_host_is_rejected(
    raw_client: TestClient,
    settings: Settings,
) -> None:
    before = _database_snapshot(settings)
    response = raw_client.get("/submit", headers={"Host": "evil.example"})
    assert response.status_code == 400
    assert _database_snapshot(settings) == before


@pytest.mark.parametrize(
    "host",
    [
        "testserver:",
        "testserver:not-a-port",
        "testserver:0",
        "testserver:65536",
        "testserver:80:81",
        "user@testserver:80",
        "[::1",
        "[::1]extra",
    ],
)
def test_malformed_host_authorities_are_rejected(
    raw_client: TestClient,
    host: str,
) -> None:
    assert raw_client.get("/submit", headers={"Host": host}).status_code == 400


def test_dns_host_matching_is_case_insensitive(ui_settings: Settings) -> None:
    dns_app = create_app(
        ui_settings,
        allowed_hosts=("console.example",),
        allowed_origins=("http://console.example",),
    )
    with TestClient(dns_app, base_url="http://console.example") as client:
        response = client.get("/submit", headers={"Host": "CONSOLE.EXAMPLE"})
    assert response.status_code == 200


def test_dns_host_and_origin_case_normalize_together(ui_settings: Settings) -> None:
    dns_app = create_app(
        ui_settings,
        allowed_hosts=("console.example",),
        allowed_origins=("http://console.example",),
    )
    with TestClient(dns_app, base_url="http://console.example") as client:
        token = _rendered_token(client)
        response = client.post(
            "/submit",
            data={
                "submission_id": "submission_security_dns_case",
                "existing": "missing.txt",
                "csrf_token": token,
            },
            headers={"Host": "CONSOLE.EXAMPLE", "Origin": "http://CONSOLE.EXAMPLE"},
            follow_redirects=False,
        )
    assert response.status_code == 404


def test_configured_dns_host_case_is_normalized() -> None:
    settings = Settings(ui_allowed_hosts=("Console.Example",))
    assert settings.ui_allowed_hosts == ("console.example",)


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1:8787",
        "console.example:8787",
        "[::1]:8787",
        "user@console.example",
        "fe80::1%lo0",
    ],
)
def test_configured_host_authority_variants_are_rejected(host: str) -> None:
    with pytest.raises(ValidationError, match="invalid UI allowed host"):
        Settings(ui_allowed_hosts=(host,))


def test_ipv6_loopback_host_and_origin_are_accepted(settings: Settings) -> None:
    ipv6_app = create_app(
        settings,
        allowed_hosts=("::1",),
        allowed_origins=("http://[::1]:8787",),
    )
    with TestClient(ipv6_app, base_url="http://test-transport") as client:
        page = client.get("/submit", headers={"Host": "[::1]:8787"})
        parser = _MutationFormTokens()
        parser.feed(page.text)
        token = parser.tokens["/submit"][0]
        assert page.status_code == 200
        response = client.post(
            "/submit",
            data={
                "submission_id": "submission_security_ipv6_loopback",
                "existing": "missing.txt",
                "csrf_token": token,
            },
            headers={"Host": "[::1]:8787", "Origin": "http://[::1]:8787"},
            follow_redirects=False,
        )
    assert response.status_code == 404


@pytest.mark.parametrize(
    "host",
    [
        "::1",
        "[::2]:8787",
        "[::1%25lo0]:8787",
        "user@[::1]:8787",
        "[::1]:",
        "[::1]:not-a-port",
        "[::1]:0",
        "[::1]:65536",
        "[::1]:8787:8788",
    ],
)
def test_ipv6_host_authority_variants_fail_closed(
    settings: Settings,
    host: str,
) -> None:
    ipv6_app = create_app(
        settings,
        allowed_hosts=("::1",),
        allowed_origins=("http://[::1]:8787",),
    )
    with TestClient(ipv6_app, base_url="http://test-transport") as client:
        response = client.get("/submit", headers={"Host": host})
    assert response.status_code == 400


@pytest.mark.parametrize(
    "origin",
    [
        "http://[::2]:8787",
        "http://::1:8787",
        "http://[::1%25lo0]:8787",
        "http://user@[::1]:8787",
        "http://[::1]:",
        "http://[::1]:8788",
        "http://[::1]:08787",
    ],
)
def test_ipv6_origin_variants_fail_closed(
    settings: Settings,
    origin: str,
) -> None:
    ipv6_app = create_app(
        settings,
        allowed_hosts=("::1",),
        allowed_origins=("http://[::1]:8787",),
    )
    with TestClient(ipv6_app, base_url="http://test-transport") as client:
        page = client.get("/submit", headers={"Host": "[::1]:8787"})
        parser = _MutationFormTokens()
        parser.feed(page.text)
        token = parser.tokens["/submit"][0]
        response = client.post(
            "/submit",
            data={"existing": "missing.txt", "csrf_token": token},
            headers={"Host": "[::1]:8787", "Origin": origin},
            follow_redirects=False,
        )
    assert response.status_code == 403


@pytest.mark.parametrize(
    "origin",
    [
        "http://console.example:",
        "http://console.example?",
        "http://console.example#",
        "http://console.example?#",
        "http://[::1]:",
        "http://::1:8787",
        "http://user@[::1]:8787",
        "http://[::1%25lo0]:8787",
    ],
)
def test_delimiter_only_and_empty_port_allowed_origins_are_rejected(
    settings: Settings,
    origin: str,
) -> None:
    allowed_host = "::1" if "[::1]" in origin else "console.example"
    with pytest.raises(ValueError, match="invalid UI origin"):
        create_app(
            settings,
            allowed_hosts=(allowed_host,),
            allowed_origins=(origin,),
        )


def test_default_app_allows_loopback_names_but_not_testclient_host(settings: Settings) -> None:
    local_app = create_app(settings)
    with TestClient(local_app, base_url="http://127.0.0.1:8787") as ipv4:
        assert ipv4.get("/submit").status_code == 200
    with TestClient(local_app, base_url="http://localhost:8787") as localhost:
        assert localhost.get("/submit").status_code == 200
    with TestClient(local_app, base_url="http://test-transport") as ipv6:
        assert ipv6.get("/submit", headers={"Host": "[::1]:8787"}).status_code == 200
    with TestClient(local_app, base_url="http://testserver") as implicit_test_host:
        assert implicit_test_host.get("/submit").status_code == 400


def test_security_headers_cover_success_and_handled_errors(raw_client: TestClient) -> None:
    token = _rendered_token(raw_client)
    responses = [
        raw_client.get("/submit"),
        raw_client.get("/cases/case_missing"),
        raw_client.post(
            "/submit",
            data={"existing": "missing.txt"},
            headers={"Origin": "http://testserver"},
        ),
        raw_client.post(
            "/batch",
            data={"concurrency": "true", "csrf_token": token},
            headers={"Origin": "http://testserver"},
        ),
        raw_client.get("/submit", headers={"Host": "evil.example"}),
    ]
    assert [response.status_code for response in responses] == [200, 404, 403, 422, 400]
    for response in responses:
        _assert_security_headers(response)


def test_security_headers_cover_unhandled_errors(ui_settings: Settings) -> None:
    broken = create_app(
        ui_settings,
        allowed_hosts=("testserver",),
        allowed_origins=("http://testserver",),
    )

    @broken.get("/security-test-unhandled")
    async def security_test_unhandled() -> None:
        raise RuntimeError("test-only unhandled failure")

    with TestClient(
        broken,
        base_url="http://testserver",
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/security-test-unhandled")
    assert response.status_code == 500
    _assert_security_headers(response)


@pytest.mark.asyncio
async def test_foreign_origin_is_rejected_without_reading_a_large_body(
    app: FastAPI,
) -> None:
    receive_calls = 0
    sent: list[dict[str, Any]] = []

    async def unread_body() -> dict[str, Any]:
        nonlocal receive_calls
        receive_calls += 1
        raise AssertionError("the hostile body must not be read")

    async def capture(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/submit",
            "raw_path": b"/submit",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"host", b"testserver"),
                (b"origin", b"https://evil.example"),
                (b"content-length", b"999999999"),
                (b"content-type", b"multipart/form-data; boundary=never-read"),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        },
        unread_body,
        capture,
    )

    assert receive_calls == 0
    start = next(message for message in sent if message["type"] == "http.response.start")
    assert start["status"] == 403
    response_headers = {name.decode(): value.decode() for name, value in start["headers"]}
    assert response_headers["content-security-policy"] == (
        "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'"
    )


@pytest.mark.asyncio
async def test_csrf_middleware_itself_rejects_a_missing_token_before_inner_app() -> None:
    inner_methods: list[str] = []

    async def inner(
        scope: dict[str, Any],
        _receive: Any,
        send: Any,
    ) -> None:
        inner_methods.append(scope["method"])
        await Response(
            status_code=204,
            headers={"X-Issued-CSRF": str(scope[CSRF_SCOPE_KEY])},
        )(scope, _receive, send)

    middleware = CSRFMiddleware(
        inner,
        secret=secrets.token_bytes(32),
        allowed_origins=("http://testserver",),
    )

    def scope(method: str, headers: list[tuple[bytes, bytes]]) -> dict[str, Any]:
        return {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": "/mutation",
            "raw_path": b"/mutation",
            "query_string": b"",
            "root_path": "",
            "headers": headers,
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        }

    get_messages: list[dict[str, Any]] = []

    async def empty_request() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def capture_get(message: dict[str, Any]) -> None:
        get_messages.append(message)

    await middleware(scope("GET", [(b"host", b"testserver")]), empty_request, capture_get)
    get_start = next(
        message for message in get_messages if message["type"] == "http.response.start"
    )
    get_headers = {name.decode(): value.decode() for name, value in get_start["headers"]}
    cookie = SimpleCookie()
    cookie.load(get_headers["set-cookie"])
    session_cookie = cookie[SESSION_COOKIE_NAME].value

    post_messages: list[dict[str, Any]] = []

    async def tokenless_body() -> dict[str, Any]:
        return {
            "type": "http.request",
            "body": b"existing=invoice_1001.txt",
            "more_body": False,
        }

    async def capture_post(message: dict[str, Any]) -> None:
        post_messages.append(message)

    await middleware(
        scope(
            "POST",
            [
                (b"host", b"testserver"),
                (b"origin", b"http://testserver"),
                (b"content-type", b"application/x-www-form-urlencoded"),
                (b"cookie", f"{SESSION_COOKIE_NAME}={session_cookie}".encode()),
            ],
        ),
        tokenless_body,
        capture_post,
    )
    post_start = next(
        message for message in post_messages if message["type"] == "http.response.start"
    )
    assert post_start["status"] == 403
    assert inner_methods == ["GET"]


@pytest.mark.asyncio
async def test_exact_host_validation_also_covers_websocket_scopes() -> None:
    inner_calls = 0
    sent: list[dict[str, Any]] = []

    async def inner(_scope: Any, _receive: Any, _send: Any) -> None:
        nonlocal inner_calls
        inner_calls += 1

    async def receive() -> dict[str, Any]:
        return {"type": "websocket.disconnect"}

    async def capture(message: dict[str, Any]) -> None:
        sent.append(message)

    middleware = ExactTrustedHostMiddleware(inner, allowed_hosts=("testserver",))
    await middleware(
        {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "scheme": "ws",
            "path": "/socket",
            "raw_path": b"/socket",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"host", b"evil.example")],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
            "subprotocols": [],
        },
        receive,
        capture,
    )
    assert inner_calls == 0
    assert sent[0]["status"] == 400
