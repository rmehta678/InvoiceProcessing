"""Session-bound CSRF and strict same-origin request protection."""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from http.cookies import CookieError, SimpleCookie
from ipaddress import IPv6Address, ip_address
from typing import Any

from fastapi import Request
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from invoice_agents.config import is_ui_loopback_host, normalize_ui_host

CSRF_FIELD_NAME = "csrf_token"
CSRF_HEADER_NAME = "x-csrf-token"
SESSION_COOKIE_NAME = "galatiq_session"
CSRF_SCOPE_KEY = "invoice_agents.csrf_token"
COOKIE_SECURE_SCOPE_KEY = "invoice_agents.cookie_secure"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{43}$")
DEFAULT_ALLOWED_ORIGINS = (
    "http://127.0.0.1:8787",
    "http://localhost:8787",
    "http://[::1]:8787",
)
DEFAULT_ALLOWED_HOSTS = ("127.0.0.1", "localhost", "::1")
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'"
    ),
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
_PROXY_ORIGIN_HEADERS = (
    "forwarded",
    "x-forwarded-host",
    "x-forwarded-port",
    "x-forwarded-proto",
)
_NATIVE_FORM_FETCH_METADATA = {
    "sec-fetch-site": "same-origin",
    "sec-fetch-mode": "navigate",
    "sec-fetch-dest": "document",
    "sec-fetch-user": "?1",
}
DEFAULT_MUTATION_BODY_MAX_BYTES = 16_777_216

Origin = tuple[str, str, int]
Authority = tuple[str, int | None]


@dataclass(frozen=True)
class OriginPolicy:
    origins: frozenset[Origin]
    scheme: str

    @property
    def cookie_secure(self) -> bool:
        return self.scheme == "https"


def validate_allowed_hosts(values: Sequence[str]) -> tuple[str, ...]:
    """Return exact bare hosts; wildcards, ports, paths, and empty defaults fail."""

    if not values:
        raise ValueError("at least one explicit UI allowed host is required")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("UI allowed hosts must be strings")
        host = normalize_ui_host(value)
        if host not in normalized:
            normalized.append(host)
    return tuple(normalized)


def _encoded_digest(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _parsed_port(value: str) -> int | None:
    if not value or not value.isascii() or not value.isdigit():
        return None
    port = int(value)
    if not 1 <= port <= 65_535 or str(port) != value:
        return None
    return port


def _parsed_authority(value: str) -> Authority | None:
    """Parse one RFC-style host[:port], requiring brackets around IPv6."""

    if (
        not value
        or value != value.strip()
        or any(ord(character) <= 0x20 or ord(character) == 0x7F for character in value)
        or any(delimiter in value for delimiter in ("/", "\\", "?", "#", "@"))
    ):
        return None

    port: int | None = None
    if value.startswith("["):
        closing = value.find("]")
        if closing <= 1 or value.find("[", 1) != -1 or value.find("]", closing + 1) != -1:
            return None
        serialized_host = value[1:closing]
        suffix = value[closing + 1 :]
        if suffix:
            if not suffix.startswith(":"):
                return None
            port = _parsed_port(suffix[1:])
            if port is None:
                return None
        try:
            host = normalize_ui_host(serialized_host)
            if not isinstance(ip_address(host), IPv6Address):
                return None
        except ValueError:
            return None
        return host, port

    if "[" in value or "]" in value or value.count(":") > 1:
        return None
    serialized_host = value
    if ":" in value:
        serialized_host, serialized_port = value.split(":", 1)
        port = _parsed_port(serialized_port)
        if port is None:
            return None
    try:
        host = normalize_ui_host(serialized_host)
    except ValueError:
        return None
    try:
        address = ip_address(host)
    except ValueError:
        return host, port
    if isinstance(address, IPv6Address):
        return None
    return host, port


def _parsed_origin(value: str, *, referer: bool) -> Origin | None:
    if (
        not value
        or value != value.strip()
        or any(ord(character) <= 0x20 or ord(character) == 0x7F for character in value)
        or "\\" in value
    ):
        return None
    if value.startswith("http://"):
        scheme = "http"
        remainder = value[len("http://") :]
    elif value.startswith("https://"):
        scheme = "https"
        remainder = value[len("https://") :]
    else:
        return None

    delimiter_positions = [
        position for delimiter in "/?#" if (position := remainder.find(delimiter)) >= 0
    ]
    authority_end = min(delimiter_positions, default=len(remainder))
    authority = remainder[:authority_end]
    suffix = remainder[authority_end:]
    if (not referer and suffix) or (referer and "#" in suffix):
        return None
    parsed_authority = _parsed_authority(authority)
    if parsed_authority is None:
        return None
    host, explicit_port = parsed_authority
    port = explicit_port if explicit_port is not None else (80 if scheme == "http" else 443)
    return scheme, host, port


def _configured_origin_policy(values: Sequence[str]) -> OriginPolicy:
    if not values:
        raise ValueError("at least one explicit UI origin is required")
    parsed: set[Origin] = set()
    schemes: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError("UI origins must be strings")
        origin = _parsed_origin(value, referer=False)
        if origin is None:
            raise ValueError(f"invalid UI origin: {value!r}")
        parsed.add(origin)
        schemes.add(origin[0])
    if len(schemes) != 1:
        raise ValueError("UI allowed origins must use a single scheme")
    return OriginPolicy(frozenset(parsed), schemes.pop())


def validate_allowed_origins(values: Sequence[str]) -> tuple[str, ...]:
    """Eagerly validate exact origins and their unambiguous cookie scheme."""

    _configured_origin_policy(values)
    return tuple(values)


def ui_authorities_are_loopback(
    allowed_hosts: Sequence[str],
    allowed_origins: Sequence[str],
) -> bool:
    """Return true only when every exact configured authority is local loopback."""

    configured_hosts = validate_allowed_hosts(allowed_hosts)
    origin_policy = _configured_origin_policy(allowed_origins)
    return all(is_ui_loopback_host(host) for host in configured_hosts) and all(
        is_ui_loopback_host(host) for _scheme, host, _port in origin_policy.origins
    )


def _request_origin(scope: Scope, headers: Headers) -> Origin | None:
    host_values = headers.getlist("host")
    if len(host_values) != 1:
        return None
    scheme = str(scope.get("scheme") or "").lower()
    return _parsed_origin(f"{scheme}://{host_values[0]}", referer=False)


def _has_strict_same_origin(
    scope: Scope,
    headers: Headers,
    allowed_origins: frozenset[Origin],
) -> bool:
    if any(headers.getlist(name) for name in _PROXY_ORIGIN_HEADERS):
        return False
    request_origin = _request_origin(scope, headers)
    if request_origin is None or request_origin not in allowed_origins:
        return False
    origin_values = headers.getlist("origin")
    referer_values = headers.getlist("referer")
    if not origin_values and not referer_values:
        return False
    if len(origin_values) > 1 or len(referer_values) > 1:
        return False
    referer_origin = _parsed_origin(referer_values[0], referer=True) if referer_values else None
    if referer_values and referer_origin != request_origin:
        return False
    if not origin_values:
        return referer_origin == request_origin
    origin_value = origin_values[0]
    if origin_value == "null":
        if referer_values:
            return False
        return all(
            headers.getlist(name) == [expected]
            for name, expected in _NATIVE_FORM_FETCH_METADATA.items()
        )
    return _parsed_origin(origin_value, referer=False) == request_origin


def _session_signature(session_id: str, secret: bytes) -> str:
    digest = hmac.new(secret, b"session:" + session_id.encode("ascii"), hashlib.sha256).digest()
    return _encoded_digest(digest)


def _csrf_for_session(session_id: str, secret: bytes) -> str:
    digest = hmac.new(secret, b"csrf:" + session_id.encode("ascii"), hashlib.sha256).digest()
    return _encoded_digest(digest)


def _session_cookie(headers: Headers, secret: bytes) -> str | None:
    cookie_values = headers.getlist("cookie")
    if len(cookie_values) != 1:
        return None
    cookie = SimpleCookie()
    try:
        cookie.load(cookie_values[0])
    except CookieError:
        return None
    morsel = cookie.get(SESSION_COOKIE_NAME)
    if morsel is None:
        return None
    pieces = morsel.value.split(".")
    if len(pieces) != 2:
        return None
    session_id, provided_signature = pieces
    if not TOKEN_PATTERN.fullmatch(session_id) or not TOKEN_PATTERN.fullmatch(provided_signature):
        return None
    expected_signature = _session_signature(session_id, secret)
    if not hmac.compare_digest(provided_signature, expected_signature):
        return None
    return session_id


def _new_session(secret: bytes) -> tuple[str, str]:
    session_id = secrets.token_urlsafe(32)
    return session_id, f"{session_id}.{_session_signature(session_id, secret)}"


def _set_cookie_header(value: str, *, secure: bool, max_age: int) -> str:
    cookie = SimpleCookie()
    cookie[SESSION_COOKIE_NAME] = value
    morsel = cookie[SESSION_COOKIE_NAME]
    morsel["httponly"] = True
    morsel["max-age"] = max_age
    morsel["path"] = "/"
    morsel["samesite"] = "strict"
    if secure:
        morsel["secure"] = True
    return morsel.OutputString()


def _body_receiver(body: bytes) -> Receive:
    delivered = False

    async def receive() -> Message:
        nonlocal delivered
        if delivered:
            return {"type": "http.request", "body": b"", "more_body": False}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


async def _read_bounded_body(receive: Receive, max_body_bytes: int) -> tuple[bytes | None, int]:
    chunks: list[bytes] = []
    total = 0
    while True:
        message = await receive()
        if message["type"] != "http.request":
            return None, 400
        chunk = message.get("body", b"")
        total += len(chunk)
        if total > max_body_bytes:
            return None, 413
        chunks.append(chunk)
        if not message.get("more_body", False):
            return b"".join(chunks), 200


def _declared_body_too_large(headers: Headers, max_body_bytes: int) -> bool | None:
    values = headers.getlist("content-length")
    if not values:
        return False
    if len(values) != 1 or not values[0].isascii() or not values[0].isdigit():
        return None
    return int(values[0]) > max_body_bytes


async def _submitted_form_tokens(
    scope: Scope,
    body: bytes,
    max_body_bytes: int,
) -> list[str] | None:
    request = Request(scope, receive=_body_receiver(body))
    try:
        async with request.form(
            max_files=1,
            max_fields=10_000,
            max_part_size=max_body_bytes,
        ) as form:
            values = form.getlist(CSRF_FIELD_NAME)
            tokens: list[str] = []
            for value in values:
                if not isinstance(value, str):
                    return None
                tokens.append(value)
            return tokens
    except Exception:
        return None


class ExactTrustedHostMiddleware:
    """Reject malformed authorities and hosts outside one exact allowlist."""

    def __init__(self, app: ASGIApp, *, allowed_hosts: Sequence[str]) -> None:
        self.app = app
        self.allowed_hosts = frozenset(validate_allowed_hosts(allowed_hosts))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        host_values = Headers(scope=scope).getlist("host")
        authority = _parsed_authority(host_values[0]) if len(host_values) == 1 else None
        if authority is None or authority[0] not in self.allowed_hosts:
            await PlainTextResponse("Invalid host header", status_code=400)(scope, receive, send)
            return
        await self.app(scope, receive, send)


class CSRFMiddleware:
    """Create signed sessions and reject non-same-origin mutations before body reads."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        secret: bytes,
        allowed_origins: Sequence[str],
        cookie_max_age: int = 8 * 60 * 60,
        max_body_bytes: int = DEFAULT_MUTATION_BODY_MAX_BYTES,
    ) -> None:
        if len(secret) < 32:
            raise ValueError("the application session secret must contain at least 256 bits")
        if cookie_max_age <= 0:
            raise ValueError("the session cookie lifetime must be positive")
        if max_body_bytes <= 0:
            raise ValueError("the mutation body limit must be positive")
        self.app = app
        self.secret = secret
        origin_policy = _configured_origin_policy(allowed_origins)
        self.allowed_origins = origin_policy.origins
        self.cookie_secure = origin_policy.cookie_secure
        self.cookie_max_age = cookie_max_age
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        method = str(scope.get("method") or "").upper()
        scope[COOKIE_SECURE_SCOPE_KEY] = self.cookie_secure
        if method not in SAFE_METHODS and not _has_strict_same_origin(
            scope, headers, self.allowed_origins
        ):
            await PlainTextResponse("Forbidden", status_code=403)(scope, receive, send)
            return

        session_id = _session_cookie(headers, self.secret)
        set_cookie: str | None = None
        if session_id is None:
            if method not in SAFE_METHODS:
                await PlainTextResponse("Forbidden", status_code=403)(scope, receive, send)
                return
            session_id, set_cookie = _new_session(self.secret)
        scope[CSRF_SCOPE_KEY] = _csrf_for_session(session_id, self.secret)

        downstream_receive = receive
        if method not in SAFE_METHODS:
            declared_too_large = _declared_body_too_large(headers, self.max_body_bytes)
            if declared_too_large is None:
                await PlainTextResponse("Bad Request", status_code=400)(scope, receive, send)
                return
            if declared_too_large:
                await PlainTextResponse("Content Too Large", status_code=413)(scope, receive, send)
                return
            body, body_status = await _read_bounded_body(receive, self.max_body_bytes)
            if body is None:
                await PlainTextResponse(
                    "Content Too Large" if body_status == 413 else "Bad Request",
                    status_code=body_status,
                )(scope, _body_receiver(b""), send)
                return
            form_tokens = await _submitted_form_tokens(scope, body, self.max_body_bytes)
            if form_tokens is None:
                await PlainTextResponse("Forbidden", status_code=403)(
                    scope, _body_receiver(b""), send
                )
                return
            submitted_tokens = [*headers.getlist(CSRF_HEADER_NAME), *form_tokens]
            expected_token = csrf_token(Request(scope))
            if (
                len(submitted_tokens) != 1
                or not TOKEN_PATTERN.fullmatch(submitted_tokens[0])
                or not hmac.compare_digest(submitted_tokens[0], expected_token)
            ):
                await PlainTextResponse("Forbidden", status_code=403)(
                    scope, _body_receiver(b""), send
                )
                return
            downstream_receive = _body_receiver(body)

        async def send_with_session(message: Message) -> None:
            if set_cookie is not None and message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                response_headers.append(
                    "set-cookie",
                    _set_cookie_header(
                        set_cookie,
                        secure=self.cookie_secure,
                        max_age=self.cookie_max_age,
                    ),
                )
            await send(message)

        await self.app(scope, downstream_receive, send_with_session)


class SecurityHeadersMiddleware:
    """Attach the console's defensive headers to every inner HTTP response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    response_headers[name] = value
            await send(message)

        await self.app(scope, receive, send_with_security_headers)


def csrf_token(request: Request) -> str:
    """Return the middleware-provided token for the current signed session."""

    value: Any = request.scope.get(CSRF_SCOPE_KEY)
    if not isinstance(value, str) or not TOKEN_PATTERN.fullmatch(value):
        raise RuntimeError("CSRF session context is unavailable")
    return value


def secure_cookie(request: Request) -> bool:
    """Return the scheme-derived cookie policy installed by CSRF middleware."""

    value: Any = request.scope.get(COOKIE_SECURE_SCOPE_KEY)
    if not isinstance(value, bool):
        raise RuntimeError("cookie security context is unavailable")
    return value
