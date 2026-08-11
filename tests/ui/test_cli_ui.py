"""The ui command initializes databases by default and stays loopback-only."""

from __future__ import annotations

import logging
import re
import socket
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI
from typer.testing import CliRunner

from invoice_agents.cli import app
from invoice_agents.observability.audit import RedactingFilter

runner = CliRunner()


def test_ui_uvicorn_logging_redacts_unhandled_exception_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real CLI/Uvicorn configuration must redact its ASGI exception handler."""

    canary = "sk-proj-uvicorn-exception-canary-12345678"
    failing_app = FastAPI()

    @failing_app.get("/_uvicorn_logging_failure")
    async def uvicorn_logging_failure() -> None:
        raise RuntimeError(f"api_key={canary}")

    responses: list[tuple[int, str]] = []

    def run_one_real_request(application: Any, **kwargs: Any) -> None:
        server_socket = socket.socket()
        server_socket.bind(("127.0.0.1", 0))
        server_socket.listen(5)
        port = int(server_socket.getsockname()[1])
        server = uvicorn.Server(uvicorn.Config(application, **kwargs))
        for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            current: logging.Logger | None = logging.getLogger(logger_name)
            effective_handlers: list[logging.Handler] = []
            while current is not None:
                effective_handlers.extend(current.handlers)
                if not current.propagate:
                    break
                current = current.parent
            assert effective_handlers
            assert all(
                any(isinstance(item, RedactingFilter) for item in handler.filters)
                for handler in effective_handlers
            )
        logging.getLogger("uvicorn.access").warning(
            '%s - "%s %s HTTP/%s" %d',
            "127.0.0.1",
            "GET",
            f"/_uvicorn_logging_failure?api_key={canary}",
            "1.1",
            599,
        )
        thread = threading.Thread(
            target=server.run,
            kwargs={"sockets": [server_socket]},
            daemon=True,
        )
        thread.start()
        deadline = time.monotonic() + 5
        while not server.started:
            if not thread.is_alive():
                raise AssertionError("Uvicorn exited before accepting the test request")
            if time.monotonic() > deadline:
                raise AssertionError("Uvicorn did not start")
            time.sleep(0.01)
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/_uvicorn_logging_failure",
                timeout=5,
            )
        except urllib.error.HTTPError as exc:
            responses.append((exc.code, exc.read().decode("utf-8")))
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            server_socket.close()
        assert not thread.is_alive()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "invoice_agents.ui.server.create_app", lambda *_args, **_kwargs: failing_app
    )
    monkeypatch.setattr("uvicorn.run", run_one_real_request)

    result = runner.invoke(app, ["ui", "--no-init-db"])

    assert result.exit_code == 0, result.output
    assert responses == [(500, "Internal Server Error")]
    assert "Exception in ASGI application" in result.output
    assert "…[TRACEBACK FRAMES TRUNCATED]" in result.output
    assert "RuntimeError: api_key=[REDACTED]" in result.output
    assert "GET /_uvicorn_logging_failure?api_key=[REDACTED] HTTP/1.1" in result.output
    assert "--- Logging error ---" not in result.output
    assert canary not in result.output


def test_ui_refuses_non_loopback_host_without_flag() -> None:
    server_calls = 0

    def forbidden_run(*_args: object, **_kwargs: object) -> None:
        nonlocal server_calls
        server_calls += 1

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("uvicorn.run", forbidden_run)
        result = runner.invoke(app, ["ui", "--host", "0.0.0.0"])
    assert result.exit_code == 1
    assert "category=CONFIGURATION" in result.output
    assert "stop_reason=UI_REMOTE_BIND_REQUIRES_ACKNOWLEDGEMENT" in result.output
    assert "allow-remote-i-understand" in result.output
    assert "no authentication" in result.output
    assert server_calls == 0


def test_ui_remote_acknowledgement_also_requires_explicit_allowed_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_calls = 0

    def forbidden_run(*_args: object, **_kwargs: object) -> None:
        nonlocal server_calls
        server_calls += 1

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("uvicorn.run", forbidden_run)
    result = runner.invoke(
        app,
        ["ui", "--host", "0.0.0.0", "--allow-remote-i-understand", "--no-init-db"],
    )
    assert result.exit_code == 1
    assert "INVOICE_UI_ALLOWED_HOSTS" in result.output
    assert server_calls == 0


def test_ui_remote_binding_passes_explicit_hosts_and_origins_to_the_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from invoice_agents.ui.security import CSRFMiddleware

    captured: list[tuple[object, dict[str, object]]] = []

    def fake_run(application: object, **kwargs: object) -> None:
        captured.append((application, kwargs))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INVOICE_UI_ALLOWED_HOSTS", '["console.example"]')
    monkeypatch.setenv("INVOICE_UI_SESSION_SECRET", "S" * 43)
    monkeypatch.setattr("uvicorn.run", fake_run)
    result = runner.invoke(
        app,
        ["ui", "--host", "0.0.0.0", "--allow-remote-i-understand", "--no-init-db"],
    )
    assert result.exit_code == 0
    assert len(captured) == 1
    application, kwargs = captured[0]
    assert kwargs["host"] == "0.0.0.0"
    trusted = next(
        middleware
        for middleware in application.user_middleware
        if middleware.cls.__name__ == "ExactTrustedHostMiddleware"
    )
    csrf = next(
        middleware for middleware in application.user_middleware if middleware.cls is CSRFMiddleware
    )
    assert trusted.kwargs["allowed_hosts"] == ("console.example",)
    assert csrf.kwargs["allowed_origins"] == ("http://console.example:8787",)


def test_ui_remote_binding_requires_a_shared_session_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_calls = 0

    def forbidden_run(*_args: object, **_kwargs: object) -> None:
        nonlocal server_calls
        server_calls += 1

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INVOICE_UI_ALLOWED_HOSTS", '["console.example"]')
    monkeypatch.setattr("uvicorn.run", forbidden_run)
    result = runner.invoke(
        app,
        ["ui", "--host", "0.0.0.0", "--allow-remote-i-understand", "--no-init-db"],
    )
    assert result.exit_code == 1
    assert "INVOICE_UI_SESSION_SECRET" in result.output
    assert server_calls == 0


def test_ui_ipv6_loopback_needs_no_remote_ack_and_uses_bracketed_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from invoice_agents.ui.security import CSRFMiddleware

    captured: list[tuple[object, dict[str, object]]] = []

    def fake_run(application: object, **kwargs: object) -> None:
        captured.append((application, kwargs))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("uvicorn.run", fake_run)
    result = runner.invoke(app, ["ui", "--host", "::1", "--no-init-db"])
    assert result.exit_code == 0
    assert "http://[::1]:8787" in result.output
    assert len(captured) == 1
    application, kwargs = captured[0]
    assert kwargs["host"] == "::1"
    trusted = next(
        middleware
        for middleware in application.user_middleware
        if middleware.cls.__name__ == "ExactTrustedHostMiddleware"
    )
    csrf = next(
        middleware for middleware in application.user_middleware if middleware.cls is CSRFMiddleware
    )
    assert trusted.kwargs["allowed_hosts"] == ("::1",)
    assert csrf.kwargs["allowed_origins"] == ("http://[::1]:8787",)


def test_ui_remote_binding_rejects_wildcard_allowed_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_calls = 0

    def forbidden_run(*_args: object, **_kwargs: object) -> None:
        nonlocal server_calls
        server_calls += 1

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INVOICE_UI_ALLOWED_HOSTS", '["*"]')
    monkeypatch.setattr("uvicorn.run", forbidden_run)
    result = runner.invoke(
        app,
        ["ui", "--host", "0.0.0.0", "--allow-remote-i-understand", "--no-init-db"],
    )
    assert result.exit_code == 1
    assert "allowed host" in result.output.lower()
    assert server_calls == 0


def test_ui_help_documents_loopback_default() -> None:
    result = runner.invoke(app, ["ui", "--help"])
    assert result.exit_code == 0
    assert "loopback" in result.output.lower()
    assert "init-db" in result.output


def test_ui_initializes_databases_before_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    result = runner.invoke(app, ["ui"])
    assert result.exit_code == 0
    assert (tmp_path / "inventory.db").is_file()
    assert (tmp_path / "workflow.db").is_file()
    assert "inventory database ready" in result.output
    assert "workflow database ready" in result.output

    again = runner.invoke(app, ["ui"])
    assert again.exit_code == 0
    assert "already migrated" in again.output


def test_ui_no_init_db_skips_database_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    result = runner.invoke(app, ["ui", "--no-init-db"])
    assert result.exit_code == 0
    assert not (tmp_path / "inventory.db").exists()
    assert not (tmp_path / "workflow.db").exists()


def test_ui_database_setup_failure_uses_the_shared_sanitized_cli_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-proj-ui-database-secret"
    raw_path = str((tmp_path / "private-customer-workflow.db").resolve())
    monkeypatch.setattr(
        "invoice_agents.cli._settings",
        lambda: SimpleNamespace(workflow_db=tmp_path / "workflow.db"),
    )

    def fail_setup(_settings: object) -> None:
        raise sqlite3.OperationalError(f"api_key={secret} at {raw_path}")

    monkeypatch.setattr("invoice_agents.cli.ensure_databases", fail_setup)

    result = runner.invoke(app, ["ui"])

    assert result.exit_code == 1
    lines = result.output.splitlines()
    assert len(lines) == 1, result.output
    line = lines[0]
    match = re.fullmatch(
        r"category=([A-Z_]+) stop_reason=([A-Z0-9_]+) message=(.+)",
        line,
    )
    assert match is not None, result.output
    category, stop_reason, message = match.groups()
    assert category == "DATABASE"
    assert stop_reason == "DATABASE_OPERATION_FAILED"
    assert re.search(r"\b[A-Za-z_][A-Za-z0-9_-]*=", message) is None, result.output
    for forbidden_field in (
        "debug=",
        "debug_stack=",
        "exception=",
        "exception_type=",
        "stack=",
        "traceback=",
        "details=",
        "source=",
        "source_path=",
        "path=",
    ):
        assert forbidden_field not in line, result.output
    assert "[REDACTED]" in line
    assert secret not in result.output
    assert raw_path not in result.output
    assert "Traceback (most recent call last)" not in result.output
