"""FastAPI app factory: templates, display filters, static assets, error pages.

Display filters format stored values (thousands separators, timestamps,
truncated hashes); none of them alters, softens, or recomputes a stored status.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import sqlite3
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.trustedhost import TrustedHostMiddleware

from invoice_agents.config import Settings
from invoice_agents.db.store import WorkflowStore
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.ui.routes import router
from invoice_agents.ui.runs import RunRegistry
from invoice_agents.ui.security import (
    DEFAULT_ALLOWED_HOSTS,
    DEFAULT_ALLOWED_ORIGINS,
    SECURITY_HEADERS,
    CSRFMiddleware,
    SecurityHeadersMiddleware,
    csrf_token,
    validate_allowed_hosts,
)

PACKAGE_DIR = Path(__file__).resolve().parent

NOT_FOUND_STOPS = {
    "CASE_NOT_FOUND",
    "REVIEW_NOT_FOUND",
    "EXTRACTION_NOT_FOUND",
    "RESULT_ARTIFACT_MISSING",
    "BATCH_NOT_FOUND",
}

# Visual tone per stored value; REJECT is a completed, healthy outcome and is
# deliberately neutral, never red (UI principle 2).
TONES: dict[str, dict[str, str]] = {
    "case": {
        "SUCCEEDED": "ok",
        "NEEDS_HUMAN": "warn",
        "FAILED": "fail",
        "INCOMPLETE": "pause",
    },
    "decision": {"APPROVE": "ok", "REJECT": "pause", "HOLD": "warn", "FAILED": "fail"},
    "payment": {"PAID": "ok", "DUPLICATE": "pause", "NOT_ELIGIBLE": "pause", "FAILED": "fail"},
    "inventory": {
        "AVAILABLE": "ok",
        "EXCEEDS_STOCK": "warn",
        "OUT_OF_STOCK": "warn",
        "UNKNOWN": "warn",
        "AMBIGUOUS": "warn",
        "INVALID_QUANTITY": "warn",
        "ERROR": "fail",
    },
    "review": {"PENDING": "warn", "RESOLVED": "ok"},
    "human": {
        "APPROVE": "ok",
        "REJECT": "pause",
        "REQUEST_CORRECTION": "warn",
        "ESTABLISH_MAPPING": "accent",
        "SUPERSEDE_REVISION": "accent",
    },
    "date": {
        "EXACT": "ok",
        "AMBIGUOUS": "warn",
        "RELATIVE": "warn",
        "INVALID": "warn",
        "MISSING": "warn",
    },
    "run": {"queued": "pause", "running": "accent", "done": "pause"},
}


def tone(kind: str, value: Any) -> str:
    return TONES.get(kind, {}).get(str(value), "pause")


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if value in (None, ""):
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def fmt_dt(value: Any) -> str:
    parsed = _as_datetime(value)
    if parsed is None:
        return str(value) if value else "-"
    return parsed.strftime("%Y-%m-%d %H:%M:%S UTC")


def fmt_duration(start: Any, finish: Any) -> str:
    start_dt, finish_dt = _as_datetime(start), _as_datetime(finish)
    if start_dt is None or finish_dt is None:
        return "-"
    total = (finish_dt - start_dt).total_seconds()
    if total < 0:
        return "-"
    if total < 60:
        return f"{total:.1f}s"
    minutes, seconds = divmod(round(total), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {seconds}s" if hours else f"{minutes}m {seconds}s"


def _as_decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def fmt_amount(value: Any) -> str:
    """Group digits for reading; the stored precision is preserved exactly."""

    decimal_value = _as_decimal(value)
    if decimal_value is None:
        return str(value) if value not in (None, "") else "-"
    return f"{decimal_value:,}"


def fmt_signed(value: Any) -> str:
    """Deltas always carry an explicit sign and the exact stored amount."""

    decimal_value = _as_decimal(value)
    if decimal_value is None:
        return str(value) if value not in (None, "") else "-"
    formatted = f"{decimal_value:,}"
    return f"+{formatted}" if decimal_value > 0 else formatted


def is_nonzero(value: Any) -> bool:
    """True when a stored delta differs from zero; unparseable stays visible."""

    if value is None or value == "":
        return False
    decimal_value = _as_decimal(value)
    return True if decimal_value is None else decimal_value != 0


def json_pretty(value: Any) -> str:
    if isinstance(value, str):
        try:
            parsed: Any = json.loads(value)
        except json.JSONDecodeError:
            return value
        value = parsed
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str)


def middle(value: Any, keep: int = 10, tail: int = 6) -> str:
    text = str(value)
    if len(text) <= keep + tail + 1:
        return text
    return f"{text[:keep]}…{text[-tail:]}"


def build_templates() -> Jinja2Templates:
    templates = Jinja2Templates(
        directory=PACKAGE_DIR / "templates",
        context_processors=[lambda request: {"csrf_token": csrf_token(request)}],
    )
    templates.env.filters["fmt_dt"] = fmt_dt
    templates.env.filters["fmt_amount"] = fmt_amount
    templates.env.filters["fmt_signed"] = fmt_signed
    templates.env.filters["json_pretty"] = json_pretty
    templates.env.filters["middle"] = middle
    templates.env.filters["nonzero"] = is_nonzero
    templates.env.globals["tone"] = tone
    templates.env.globals["fmt_duration"] = fmt_duration
    return templates


def create_app(
    settings: Settings | None = None,
    *,
    allowed_hosts: Sequence[str] = DEFAULT_ALLOWED_HOSTS,
    allowed_origins: Sequence[str] = DEFAULT_ALLOWED_ORIGINS,
) -> FastAPI:
    """Build the console app; no docs endpoints, localhost intent, no auth layer."""

    selected_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await asyncio.to_thread(WorkflowStore(selected_settings).recover_expired_executions)
        yield

    app = FastAPI(
        title="Galatiq Invoice Console",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    configured_hosts = validate_allowed_hosts(allowed_hosts)
    configured_origins = tuple(allowed_origins)
    app.add_middleware(
        CSRFMiddleware,
        secret=secrets.token_bytes(32),
        allowed_origins=configured_origins,
        max_body_bytes=selected_settings.source_max_bytes + 1_048_576,
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=configured_hosts,
        www_redirect=False,
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.state.settings = selected_settings
    app.state.registry = RunRegistry()
    app.state.templates = build_templates()
    app.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")
    app.include_router(router)

    @app.exception_handler(InvoiceAgentsError)
    async def invoice_error(request: Request, exc: InvoiceAgentsError) -> Response:
        status_code = 404 if exc.stop_reason in NOT_FOUND_STOPS else 400
        response: Response = app.state.templates.TemplateResponse(
            request,
            "error.html",
            {"nav": None, "error": exc},
            status_code=status_code,
        )
        return response

    @app.exception_handler(sqlite3.Error)
    async def sqlite_error(request: Request, exc: sqlite3.Error) -> Response:
        # Verbatim database failure; the console never repairs or retries.
        error = InvoiceAgentsError(ErrorCategory.DATABASE, str(exc), stop_reason="DATABASE_ERROR")
        response: Response = app.state.templates.TemplateResponse(
            request,
            "error.html",
            {"nav": None, "error": error},
            status_code=500,
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled_error(_request: Request, _exc: Exception) -> Response:
        return PlainTextResponse(
            "Internal Server Error",
            status_code=500,
            headers=SECURITY_HEADERS,
        )

    return app
