"""Web-console fixtures: ephemeral migrated DBs, isolated cwd, in-process app."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from invoice_agents.config import Settings
from invoice_agents.ui.server import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "invoices"
TEST_ORIGIN = "http://testserver"


class _CSRFTokenParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tokens: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if (
            tag == "input"
            and attributes.get("type") == "hidden"
            and attributes.get("name") == "csrf_token"
            and attributes.get("value")
        ):
            self.tokens.append(str(attributes["value"]))


class CSRFTestClient(TestClient):
    """Exercise the real token flow while keeping legacy route calls concise."""

    def _csrf_token(self) -> str:
        page = super().get("/submit")
        parser = _CSRFTokenParser()
        parser.feed(page.text)
        assert parser.tokens
        assert len(set(parser.tokens)) == 1
        return parser.tokens[0]

    def request(self, method: str, url: str, **kwargs: Any):  # type: ignore[no-untyped-def]
        if method.upper() not in {"GET", "HEAD", "OPTIONS"}:
            headers = dict(kwargs.pop("headers", {}) or {})
            headers.setdefault("Origin", TEST_ORIGIN)
            kwargs["headers"] = headers
            if not any(name.lower() == "x-csrf-token" for name in headers):
                data = dict(kwargs.pop("data", {}) or {})
                data.setdefault("csrf_token", self._csrf_token())
                kwargs["data"] = data
        return super().request(method, url, **kwargs)


@pytest.fixture(autouse=True)
def _reset_sse_app_status() -> Iterator[None]:
    """sse_starlette's module-level AppStatus lazily binds its exit Event to the
    first event loop that serves SSE; each test here runs the app on a fresh
    loop, so a stale Event raises "bound to a different event loop"."""

    from sse_starlette.sse import AppStatus

    AppStatus.should_exit = False
    AppStatus.should_exit_event = None
    yield
    AppStatus.should_exit = False
    AppStatus.should_exit_event = None


@pytest.fixture
def ui_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated cwd with its own data/invoices corpus subset; artifacts stay here."""

    invoice_dir = tmp_path / "data" / "invoices"
    invoice_dir.mkdir(parents=True)
    for name in ("invoice_1001.txt", "invoice_1002.txt"):
        shutil.copy(DATA_DIR / name, invoice_dir / name)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def app(settings: Settings, ui_workdir: Path) -> FastAPI:
    return create_app(
        settings,
        allowed_hosts=("testserver",),
        allowed_origins=(TEST_ORIGIN,),
    )


@pytest.fixture
def raw_client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, base_url=TEST_ORIGIN) as test_client:
        yield test_client


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with CSRFTestClient(app, base_url=TEST_ORIGIN) as test_client:
        yield test_client
