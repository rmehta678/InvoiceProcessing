"""Audit-safe event recording with recursive credential redaction."""

from __future__ import annotations

import json
import logging
import math
import re
import traceback
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from invoice_agents.db.core import connect_database
from invoice_agents.models import CaseResult

SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "auth_token",
        "access_token",
        "client_secret",
        "cookie",
        "id_token",
        "proxy_authorization",
        "refresh_token",
        "secret",
        "session_token",
        "set_cookie",
        "token",
        "xai_api_key",
        "xai_apikey",
    }
)
CREDENTIAL_TEXT_KEY = (
    r"(?:xai[-_]?api[-_]?key|api[-_]?key|authorization|proxy[-_]?authorization|"
    r"client[-_]?secret|secret|set[-_]?cookie|cookie|access[-_]?token|"
    r"refresh[-_]?token|session[-_]?token|auth[-_]?token|id[-_]?token|token)"
)
BEARER_VALUE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
AUTHORIZATION_HEADER_VALUE = re.compile(
    r"(?im)(?P<prefix>(?<![\w-])(?:proxy-)?authorization[ \t]*:[ \t]*)"
    r"[^\r\n]*(?:(?:\r(?!\n)[^\r\n]*)|(?:\r?\n(?:[ \t]+|"
    r"(?:bearer|basic|digest|token)\b)[^\r\n]*))*"
)
PROVIDER_CREDENTIAL_VALUE = re.compile(r"(?i)\b(?:xai|sk)(?:-proj)?-[A-Za-z0-9_-]{8,}\b")
KEYED_CREDENTIAL_VALUE = re.compile(
    rf"(?i)(?P<prefix>(?<![\w-]){CREDENTIAL_TEXT_KEY}"
    r"[\"']?\s*[:=]\s*[\"']?)"
    r"(?P<value>\[REDACTED\]|[^\s\"',}\]]+)(?P<suffix>[\"']?)"
)
COOKIE_HEADER_VALUE = re.compile(
    r"(?im)(?P<prefix>(?<![\w-])(?:set-cookie|cookie)[ \t]*:[ \t]*)"
    r"[^\r\n]*(?:\r?\n[ \t]+(?![A-Za-z0-9-]+[ \t]*:)[^\r\n]*)*"
)
COOKIE_ASSIGNMENT_VALUE = re.compile(
    r"(?im)(?P<prefix>(?<![\w-])cookie[\"']?[ \t]*=[ \t]*)[^\r\n]*"
)
DOUBLE_QUOTED_COOKIE_FIELD_VALUE = re.compile(
    r'(?i)(?P<prefix>"(?:set-cookie|cookie)"\s*:\s*")'
    r'(?P<value>(?:\\.|[^"\\\r\n])*)(?P<suffix>"?)'
)
SINGLE_QUOTED_COOKIE_FIELD_VALUE = re.compile(
    r"(?i)(?P<prefix>'(?:set-cookie|cookie)'\s*:\s*')"
    r"(?P<value>(?:\\.|[^'\\\r\n])*)(?P<suffix>'?)"
)
SPLIT_PROVIDER_CREDENTIAL = re.compile(
    r"(?i)\b(?:x[\t\r\n]*a[\t\r\n]*i|s[\t\r\n]*k)"
    r"(?:[\t\r\n]*-[\t\r\n]*p[\t\r\n]*r[\t\r\n]*o[\t\r\n]*j)?"
    r"[\t\r\n]*-[\t\r\n]*(?:[A-Za-z0-9_-][\t\r\n]*){8,}"
)
ANSI_ESCAPE = re.compile(
    r"(?:\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-_])|"
    r"\x9b[0-?]*[ -/]*[@-~])"
)
SAFE_PROVIDER_REQUEST_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
SANITIZED_TEXT_MAX_CHARS = 4096
SANITIZED_TEXT_INPUT_MAX_CHARS = 16_384
TRUNCATION_MARKER = "…[TRUNCATED]"
JSON_TEXT_MAX_CHARS = 65_536
JSON_MAX_DEPTH = 20
JSON_MAX_NODES = 1_000
JSON_MAX_INTEGER_BITS = 512
JSON_MAX_NUMERIC_DIGITS = 128
JSON_MAX_TOTAL_NUMERIC_BITS = 8_192
JSON_MAX_TOTAL_NUMERIC_DIGITS = 2_048
JSON_PAYLOAD_REJECTED = "JSON_PAYLOAD_REJECTED"
EVENT_PAYLOAD_REJECTED = "EVENT_PAYLOAD_REJECTED"
VALUE_REJECTED = "[VALUE_REJECTED]"
_CURRENT_AUDIT: ContextVar[AuditRecorder | None] = ContextVar(
    "invoice_agents_current_audit", default=None
)

_DETECTION_JOINER = "\ufff0"
_DETECTION_JOINER_PATTERN = rf"[\t\r\n{_DETECTION_JOINER}]"
_DETECTION_GAP = rf"(?:[ ]|{_DETECTION_JOINER_PATTERN})*"
_DETECTION_NON_ASCII_LETTER = "\ufff1"
_ASCII_WHITESPACE = frozenset(" \t\r\n\f\v")
_KNOWN_CONFUSABLE_ASCII = {
    "\u0430": "a",
    "\u0435": "e",
    "\u0456": "i",
    "\u043a": "k",
    "\u043e": "o",
    "\u0440": "p",
    "\u0441": "c",
    "\u0455": "s",
    "\u0445": "x",
}
_JSON_NUMBER_TOKEN = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


def _flexible_identifier(value: str) -> str:
    components = [r"[-_]" if character == "_" else re.escape(character) for character in value]
    return f"{_DETECTION_JOINER_PATTERN}*".join(components)


_FLEXIBLE_CREDENTIAL_KEY = (
    "(?:"
    + "|".join(_flexible_identifier(key) for key in sorted(SENSITIVE_KEYS, key=len, reverse=True))
    + ")"
)
_FLEXIBLE_AUTHORIZATION_KEY = (
    "(?:"
    + "|".join(_flexible_identifier(key) for key in ("proxy_authorization", "authorization"))
    + ")"
)
_FLEXIBLE_COOKIE_KEY = (
    "(?:" + "|".join(_flexible_identifier(key) for key in ("set_cookie", "cookie")) + ")"
)
FLEXIBLE_DOUBLE_QUOTED_CREDENTIAL = re.compile(
    rf"(?i)(?P<prefix>(?<![A-Za-z0-9-]){_FLEXIBLE_CREDENTIAL_KEY}"
    rf'["\']?{_DETECTION_GAP}[:=]{_DETECTION_GAP}")'
    r'(?P<value>(?:\\.|[^"\\\r\n])*)(?:"|$|(?=\r?\n))'
)
FLEXIBLE_SINGLE_QUOTED_CREDENTIAL = re.compile(
    rf"(?i)(?P<prefix>(?<![A-Za-z0-9-]){_FLEXIBLE_CREDENTIAL_KEY}"
    rf"[\"']?{_DETECTION_GAP}[:=]{_DETECTION_GAP}')"
    r"(?P<value>(?:\\.|[^'\\\r\n])*)(?:'|$|(?=\r?\n))"
)
FLEXIBLE_UNQUOTED_CREDENTIAL = re.compile(
    rf"(?i)(?P<prefix>(?<![A-Za-z0-9-]){_FLEXIBLE_CREDENTIAL_KEY}"
    rf"[\"']?{_DETECTION_GAP}[:=]{_DETECTION_GAP})(?![\"'])"
    r"(?P<value>\[REDACTED\]|[^\s,;}\]]+)"
)
FLEXIBLE_AUTHORIZATION_HEADER = re.compile(
    rf"(?i)(?P<prefix>(?<![A-Za-z0-9-]){_FLEXIBLE_AUTHORIZATION_KEY}"
    rf"{_DETECTION_GAP}:{_DETECTION_GAP})"
    r"(?P<value>[^\r\n]*(?:(?:\r(?!\n)[^\r\n]*)|(?:\r?\n(?:[ \t]+|"
    r"(?:bearer|basic|digest|token)\b)[^\r\n]*))*)"
)
FLEXIBLE_COOKIE_HEADER = re.compile(
    rf"(?i)(?P<prefix>(?<![A-Za-z0-9-]){_FLEXIBLE_COOKIE_KEY}"
    rf"{_DETECTION_GAP}:{_DETECTION_GAP})"
    r"(?P<value>[^\r\n]*(?:\r?\n[ \t]+(?![A-Za-z0-9-]+[ \t]*:)[^\r\n]*)*)"
)
FLEXIBLE_BEARER_VALUE = re.compile(
    rf"(?i)(?P<prefix>(?<![A-Za-z0-9-]){_flexible_identifier('bearer')}"
    rf"(?:[ ]|{_DETECTION_JOINER_PATTERN})+)"
    r"(?P<value>[A-Za-z0-9._~+/=-]+)"
)
FLEXIBLE_PROVIDER_CREDENTIAL = re.compile(
    rf"(?i)(?<![A-Za-z0-9-])(?:{_flexible_identifier('xai')}|"
    rf"{_flexible_identifier('sk')})(?:{_DETECTION_JOINER_PATTERN}*[-]"
    rf"{_DETECTION_JOINER_PATTERN}*{_flexible_identifier('proj')})?"
    rf"{_DETECTION_JOINER_PATTERN}*[-]{_DETECTION_JOINER_PATTERN}*"
    rf"[A-Za-z0-9_-](?:{_DETECTION_JOINER_PATTERN}*[A-Za-z0-9_-]){{7,}}"
)


def _credential_detection_projection(value: str) -> tuple[str, list[int]]:
    """Normalize only the detection view while retaining source-span indexes."""

    projected: list[str] = []
    source_indexes: list[int] = []
    for source_index, character in enumerate(value):
        normalized = unicodedata.normalize("NFKD", character).casefold()
        for normalized_character in normalized:
            if unicodedata.category(normalized_character) in {"Mn", "Mc", "Me"}:
                continue
            known = _KNOWN_CONFUSABLE_ASCII.get(normalized_character)
            if known is not None:
                projected.append(known)
            elif not normalized_character.isascii() and unicodedata.category(
                normalized_character
            ).startswith("L"):
                projected.append(_DETECTION_NON_ASCII_LETTER)
            else:
                projected.append(normalized_character)
            source_indexes.append(source_index)
    return "".join(projected), source_indexes


def _skip_detection_whitespace(value: str, position: int) -> int:
    while position < len(value) and value[position] in _ASCII_WHITESPACE:
        position += 1
    return position


def _is_detection_boundary_word_character(value: str) -> bool:
    return value in "abcdefghijklmnopqrstuvwxyz0123456789-" or value == _DETECTION_NON_ASCII_LETTER


def _is_provider_token_character(value: str) -> bool:
    return value == "_" or _is_detection_boundary_word_character(value)


def _match_detection_identifier(value: str, position: int, identifier: str) -> int | None:
    matched_ascii_letter = False
    for offset, expected in enumerate(identifier):
        if offset:
            position = _skip_detection_whitespace(value, position)
        if position >= len(value):
            return None
        actual = value[position]
        if expected == "_":
            if actual not in {"_", "-"}:
                return None
        elif actual == _DETECTION_NON_ASCII_LETTER:
            if not expected.isascii() or not expected.isalpha():
                return None
        elif actual != expected:
            return None
        else:
            matched_ascii_letter = True
        position += 1
    return position if matched_ascii_letter else None


def _quoted_value_end(value: str, start: int, quote: str) -> int:
    escaped = False
    for position in range(start, len(value)):
        character = value[position]
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == quote:
            return position
    return len(value)


def _assignment_credential_spans(
    value: str,
    projected: str,
    source_indexes: list[int],
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    sensitive_keys = sorted(SENSITIVE_KEYS, key=len, reverse=True)
    for position in range(len(projected)):
        if position and _is_detection_boundary_word_character(projected[position - 1]):
            continue
        for sensitive_key in sensitive_keys:
            identifier_end = _match_detection_identifier(projected, position, sensitive_key)
            if identifier_end is None:
                continue
            syntax_position = _skip_detection_whitespace(projected, identifier_end)
            if syntax_position < len(projected) and projected[syntax_position] in {'"', "'"}:
                syntax_position = _skip_detection_whitespace(projected, syntax_position + 1)
            if syntax_position >= len(projected) or projected[syntax_position] not in {":", "="}:
                continue
            value_position = _skip_detection_whitespace(projected, syntax_position + 1)
            if value_position >= len(projected):
                break
            source_start = source_indexes[value_position]
            opening = projected[value_position]
            if opening in {'"', "'"}:
                source_start += 1
                source_end = _quoted_value_end(value, source_start, opening)
            elif value.startswith("[REDACTED]", source_start):
                source_end = source_start + len("[REDACTED]")
            elif sensitive_key in {
                "authorization",
                "proxy_authorization",
                "cookie",
                "set_cookie",
            }:
                source_end = source_start
                while source_end < len(value) and value[source_end] not in {"\r", "\n"}:
                    source_end += 1
            else:
                source_end = source_start
                while (
                    source_end < len(value)
                    and value[source_end] not in _ASCII_WHITESPACE
                    and value[source_end] not in {",", ";", "}", "]"}
                ):
                    source_end += 1
            if source_end > source_start:
                spans.append((source_start, source_end))
            break
    return spans


def _provider_token_end(projected: str, position: int) -> tuple[int, int] | None:
    token_characters = 0
    last_token_position = position
    ordinary_space_used = False
    has_separator = False
    while position < len(projected):
        character = projected[position]
        if _is_provider_token_character(character):
            token_characters += 1
            has_separator = has_separator or character in "_-"
            last_token_position = position + 1
            position += 1
            continue
        if character in {"\t", "\r", "\n", "\f", "\v"}:
            position += 1
            continue
        if character == " " and not ordinary_space_used:
            continuation_start = _skip_detection_whitespace(projected, position)
            continuation_end = continuation_start
            while continuation_end < len(projected) and _is_provider_token_character(
                projected[continuation_end]
            ):
                continuation_end += 1
            continuation = projected[continuation_start:continuation_end]
            if len(continuation) < 8 or not any(item in continuation for item in "_-"):
                break
            ordinary_space_used = True
            position = continuation_start
            continue
        break
    if token_characters < 8 or (ordinary_space_used and not has_separator):
        return None
    return last_token_position, token_characters


def _provider_credential_spans(
    projected: str,
    source_indexes: list[int],
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for position in range(len(projected)):
        if position and _is_detection_boundary_word_character(projected[position - 1]):
            continue
        prefix_end = _match_detection_identifier(projected, position, "sk")
        if prefix_end is None:
            prefix_end = _match_detection_identifier(projected, position, "xai")
        if prefix_end is None:
            continue
        hyphen_position = _skip_detection_whitespace(projected, prefix_end)
        if hyphen_position >= len(projected) or projected[hyphen_position] != "-":
            continue
        token_position = _skip_detection_whitespace(projected, hyphen_position + 1)
        project_end = _match_detection_identifier(projected, token_position, "proj")
        if project_end is not None:
            second_hyphen = _skip_detection_whitespace(projected, project_end)
            if second_hyphen < len(projected) and projected[second_hyphen] == "-":
                token_position = _skip_detection_whitespace(projected, second_hyphen + 1)
        token_result = _provider_token_end(projected, token_position)
        if token_result is None:
            continue
        token_end, _token_characters = token_result
        spans.append((source_indexes[position], source_indexes[token_end - 1] + 1))
    return spans


def _redact_lexical_credentials(value: str) -> str:
    projected, source_indexes = _credential_detection_projection(value)
    if not source_indexes:
        return value
    spans = _assignment_credential_spans(value, projected, source_indexes)
    spans.extend(_provider_credential_spans(projected, source_indexes))
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    for start, end in reversed(merged):
        value = f"{value[:start]}[REDACTED]{value[end:]}"
    return value


def _redact_projected_spans(
    value: str,
    pattern: re.Pattern[str],
    *,
    group: str | int = 0,
) -> str:
    projected, source_indexes = _credential_detection_projection(value)
    spans: list[tuple[int, int]] = []
    for match in pattern.finditer(projected):
        projected_start, projected_end = match.span(group)
        if projected_start == projected_end:
            continue
        spans.append(
            (
                source_indexes[projected_start],
                source_indexes[projected_end - 1] + 1,
            )
        )
    for source_start, source_end in reversed(spans):
        value = f"{value[:source_start]}[REDACTED]{value[source_end:]}"
    return value


def _redact_text_patterns(value: str) -> str:
    span_safe = _redact_lexical_credentials(value)
    for pattern in (
        FLEXIBLE_DOUBLE_QUOTED_CREDENTIAL,
        FLEXIBLE_SINGLE_QUOTED_CREDENTIAL,
        FLEXIBLE_UNQUOTED_CREDENTIAL,
        FLEXIBLE_AUTHORIZATION_HEADER,
        FLEXIBLE_COOKIE_HEADER,
        FLEXIBLE_BEARER_VALUE,
    ):
        span_safe = _redact_projected_spans(span_safe, pattern, group="value")
    span_safe = _redact_projected_spans(span_safe, FLEXIBLE_PROVIDER_CREDENTIAL)
    double_cookie_safe = DOUBLE_QUOTED_COOKIE_FIELD_VALUE.sub(
        lambda match: f"{match.group('prefix')}[REDACTED]{match.group('suffix')}",
        span_safe,
    )
    quoted_cookie_safe = SINGLE_QUOTED_COOKIE_FIELD_VALUE.sub(
        lambda match: f"{match.group('prefix')}[REDACTED]{match.group('suffix')}",
        double_cookie_safe,
    )
    cookie_safe = COOKIE_HEADER_VALUE.sub(
        lambda match: f"{match.group('prefix')}[REDACTED]",
        quoted_cookie_safe,
    )
    authorization_header_safe = AUTHORIZATION_HEADER_VALUE.sub(
        lambda match: f"{match.group('prefix')}[REDACTED]",
        cookie_safe,
    )
    provider_safe = SPLIT_PROVIDER_CREDENTIAL.sub("[REDACTED]", authorization_header_safe)
    bearer_safe = BEARER_VALUE.sub("Bearer [REDACTED]", provider_safe)
    keyed_safe = KEYED_CREDENTIAL_VALUE.sub(
        lambda match: f"{match.group('prefix')}[REDACTED]{match.group('suffix')}",
        PROVIDER_CREDENTIAL_VALUE.sub("[REDACTED]", bearer_safe),
    )
    return COOKIE_ASSIGNMENT_VALUE.sub(
        lambda match: f"{match.group('prefix')}[REDACTED]",
        keyed_safe,
    )


def _is_default_ignorable(codepoint: int) -> bool:
    return (
        codepoint in {0x00AD, 0x034F, 0x061C, 0x3164, 0xFEFF, 0xFFA0}
        or 0x115F <= codepoint <= 0x1160
        or 0x17B4 <= codepoint <= 0x17B5
        or 0x180B <= codepoint <= 0x180F
        or 0x200B <= codepoint <= 0x200F
        or 0x202A <= codepoint <= 0x202E
        or 0x2060 <= codepoint <= 0x206F
        or 0xFE00 <= codepoint <= 0xFE0F
        or 0xFFF0 <= codepoint <= 0xFFF8
        or 0x1BCA0 <= codepoint <= 0x1BCA3
        or 0x1D173 <= codepoint <= 0x1D17A
        or 0xE0000 <= codepoint <= 0xE0FFF
    )


def _strip_unsafe_controls(value: str) -> str:
    without_ansi = ANSI_ESCAPE.sub("", value)
    cleaned: list[str] = []
    for character in without_ansi:
        codepoint = ord(character)
        if character in {"\t", "\n", "\r"}:
            cleaned.append(character)
        elif character in {"\u2028", "\u2029"}:
            cleaned.append("\n")
        elif _is_default_ignorable(codepoint) or unicodedata.category(character) in {"Cc", "Cs"}:
            continue
        else:
            cleaned.append(character)
    return "".join(cleaned)


def _normalize_lines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def sanitize_text(value: str) -> str:
    """Redact credential-like free text and apply one deterministic size ceiling."""

    input_truncated = len(value) > SANITIZED_TEXT_INPUT_MAX_CHARS
    bounded = value[:SANITIZED_TEXT_INPUT_MAX_CHARS]
    sanitized = _normalize_lines(_redact_text_patterns(_strip_unsafe_controls(bounded)))
    if not input_truncated and len(sanitized) <= SANITIZED_TEXT_MAX_CHARS:
        return sanitized
    visible_chars = SANITIZED_TEXT_MAX_CHARS - len(TRUNCATION_MARKER)
    return f"{sanitized[:visible_chars]}{TRUNCATION_MARKER}"


def safe_provider_request_id(value: object) -> str | None:
    if (
        not isinstance(value, str)
        or SAFE_PROVIDER_REQUEST_ID.fullmatch(value) is None
        or sanitize_text(value) != value
    ):
        return None
    return value


def _normalized_key(value: str) -> str:
    safe = _normalize_lines(_strip_unsafe_controls(value))
    projected, _source_indexes = _credential_detection_projection(safe)
    return "".join(
        character for character in projected if character not in _ASCII_WHITESPACE
    ).replace("-", "_")


def safe_tool_call_id(value: object) -> str | None:
    """Return one unchanged, bounded, non-credential tool correlation ID."""

    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or value != value.strip()
        or not value.isprintable()
        or sanitize_text(value) != value
    ):
        return None
    return value


def _is_sensitive_key(value: str) -> bool:
    projected = _normalized_key(value)
    return any(
        (identifier_end := _match_detection_identifier(projected, 0, sensitive_key)) is not None
        and identifier_end == len(projected)
        for sensitive_key in SENSITIVE_KEYS
    )


def _redact(value: Any, *, depth: int, budget: list[int]) -> Any:
    budget[0] += 1
    if depth > JSON_MAX_DEPTH or budget[0] > JSON_MAX_NODES:
        return VALUE_REJECTED
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            raw_key = str(key)
            safe_key = sanitize_text(raw_key)
            cleaned[safe_key] = (
                "[REDACTED]"
                if _is_sensitive_key(raw_key) or _is_sensitive_key(safe_key)
                else _redact(item, depth=depth + 1, budget=budget)
            )
        return cleaned
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return sanitize_text(bytes(value).decode("utf-8", errors="replace"))
    if isinstance(value, Sequence):
        return [_redact(item, depth=depth + 1, budget=budget) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return VALUE_REJECTED
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize_text(str(value))


def redact(value: Any) -> Any:
    """Redact exact secret keys and sanitize every serializable text boundary."""

    if not _json_value_within_limits(value, allow_nonfinite=True):
        return VALUE_REJECTED
    try:
        cleaned = _redact(value, depth=0, budget=[0])
        if not _json_value_within_limits(cleaned):
            return VALUE_REJECTED
        json.dumps(cleaned, allow_nan=False, ensure_ascii=False)
    except (OverflowError, RuntimeError, TypeError, ValueError):
        return VALUE_REJECTED
    return cleaned


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate JSON key")
        payload[key] = value
    return payload


def _reject_json_constant(_value: str) -> object:
    raise ValueError("non-finite JSON number")


def _canonical_json_int(value: str) -> int:
    parsed = int(value)
    if json.dumps(parsed, allow_nan=False) != value:
        raise ValueError("non-canonical JSON integer")
    return parsed


def _canonical_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or json.dumps(parsed, allow_nan=False) != value:
        raise ValueError("non-canonical JSON float")
    return parsed


def _validate_json_lexemes(value: str) -> None:
    depth = 0
    in_string = False
    total_numeric_digits = 0
    position = 0
    while position < len(value):
        character = value[position]
        if in_string:
            codepoint = ord(character)
            if 0xD800 <= codepoint <= 0xDFFF:
                raise ValueError("JSON string contains an unpaired surrogate")
            if character == "\\":
                if position + 1 >= len(value):
                    raise ValueError("JSON string ends in an escape")
                if value[position + 1] == "u":
                    escape = value[position + 2 : position + 6]
                    if len(escape) != 4 or any(
                        item not in "0123456789abcdefABCDEF" for item in escape
                    ):
                        raise ValueError("JSON string contains an invalid Unicode escape")
                    escaped_codepoint = int(escape, 16)
                    if 0xD800 <= escaped_codepoint <= 0xDBFF:
                        low_prefix = value[position + 6 : position + 8]
                        low_escape = value[position + 8 : position + 12]
                        if low_prefix != "\\u" or len(low_escape) != 4:
                            raise ValueError("JSON string contains an unpaired high surrogate")
                        low_codepoint = int(low_escape, 16)
                        if not 0xDC00 <= low_codepoint <= 0xDFFF:
                            raise ValueError("JSON string contains an unpaired high surrogate")
                        position += 12
                        continue
                    if 0xDC00 <= escaped_codepoint <= 0xDFFF:
                        raise ValueError("JSON string contains an unpaired low surrogate")
                    position += 6
                    continue
                position += 2
                continue
            if character == '"':
                in_string = False
            elif ord(character) < 0x20:
                raise ValueError("JSON string contains an unescaped control")
            position += 1
            continue
        if character == '"':
            in_string = True
        elif character in "{[":
            depth += 1
            if depth > JSON_MAX_DEPTH + 1:
                raise ValueError("JSON nesting exceeds the audit boundary")
        elif character in "}]":
            depth -= 1
        elif character == "-" or character.isdigit():
            number = _JSON_NUMBER_TOKEN.match(value, position)
            if number is not None:
                numeric_lexeme = number.group(0)
                numeric_digits = sum(item.isdigit() for item in numeric_lexeme)
                total_numeric_digits += numeric_digits
                if (
                    numeric_digits > JSON_MAX_NUMERIC_DIGITS
                    or total_numeric_digits > JSON_MAX_TOTAL_NUMERIC_DIGITS
                ):
                    raise ValueError("JSON numeric budget exceeded")
                position = number.end()
                continue
        position += 1


def _bounded_json_mapping(value: str) -> dict[str, object]:
    if len(value) > JSON_TEXT_MAX_CHARS or value != value.strip():
        raise ValueError("JSON text exceeds the audit boundary")
    _validate_json_lexemes(value)
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_canonical_json_float,
            parse_int=_canonical_json_int,
        )
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds the parser boundary") from exc
    if type(parsed) is not dict:
        raise ValueError("JSON audit payload is not an object")
    if not _json_value_within_limits(parsed):
        raise ValueError("JSON audit payload exceeds structural limits")
    return parsed


def _contains_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def _json_value_within_limits(value: object, *, allow_nonfinite: bool = False) -> bool:
    node_budget = 0
    character_budget = 0
    numeric_bit_budget = 0
    active_containers: set[int] = set()
    stack: list[tuple[bool, object, int]] = [(False, value, 0)]
    try:
        while stack:
            exiting, item, depth = stack.pop()
            if exiting:
                active_containers.remove(id(item))
                continue
            node_budget += 1
            if depth > JSON_MAX_DEPTH or node_budget > JSON_MAX_NODES:
                return False
            if isinstance(item, Mapping):
                identity = id(item)
                if identity in active_containers or len(item) > JSON_MAX_NODES:
                    return False
                active_containers.add(identity)
                stack.append((True, item, depth))
                safe_keys: set[str] = set()
                children: list[object] = []
                for key, nested in item.items():
                    raw_key = str(key)
                    character_budget += len(raw_key)
                    safe_key = sanitize_text(raw_key)
                    if (
                        _contains_surrogate(raw_key)
                        or safe_key in safe_keys
                        or character_budget > JSON_TEXT_MAX_CHARS
                    ):
                        return False
                    safe_keys.add(safe_key)
                    children.append(nested)
                stack.extend((False, child, depth + 1) for child in reversed(children))
                continue
            if isinstance(item, str):
                character_budget += len(item)
                if _contains_surrogate(item) or character_budget > JSON_TEXT_MAX_CHARS:
                    return False
                continue
            if isinstance(item, (bytes, bytearray, memoryview)):
                character_budget += len(item)
                if character_budget > JSON_TEXT_MAX_CHARS:
                    return False
                continue
            if isinstance(item, Sequence):
                identity = id(item)
                if identity in active_containers or len(item) > JSON_MAX_NODES:
                    return False
                active_containers.add(identity)
                stack.append((True, item, depth))
                stack.extend((False, child, depth + 1) for child in reversed(item))
                continue
            if item is None or type(item) is bool:
                continue
            if type(item) is int:
                integer_bits = abs(item).bit_length()
                numeric_bit_budget += integer_bits
                if (
                    integer_bits > JSON_MAX_INTEGER_BITS
                    or numeric_bit_budget > JSON_MAX_TOTAL_NUMERIC_BITS
                ):
                    return False
                continue
            if type(item) is float:
                numeric_bit_budget += 64
                if (
                    not allow_nonfinite and not math.isfinite(item)
                ) or numeric_bit_budget > JSON_MAX_TOTAL_NUMERIC_BITS:
                    return False
                continue
            text = str(item)
            character_budget += len(text)
            if _contains_surrogate(text) or character_budget > JSON_TEXT_MAX_CHARS:
                return False
    except (OverflowError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _rejected_json_payload() -> str:
    return json.dumps(
        {"error": JSON_PAYLOAD_REJECTED, "original": "[REDACTED]"},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _rejected_event_payload(event_type: str) -> dict[str, str]:
    return {
        "error": EVENT_PAYLOAD_REJECTED,
        "type": sanitize_text(event_type.removeprefix("autogen.")),
    }


def sanitize_json_text(value: str) -> str:
    """Redact one JSON object or emit an explicit opaque rejection marker."""

    try:
        parsed = _bounded_json_mapping(value)
    except (TypeError, ValueError):
        return _rejected_json_payload()
    cleaned = redact(parsed)
    if not isinstance(cleaned, Mapping):
        return _rejected_json_payload()
    try:
        return json.dumps(
            cleaned,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (OverflowError, RuntimeError, TypeError, ValueError):
        return _rejected_json_payload()


def _safe_usage(value: object) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    usage: dict[str, int] = {}
    for field in ("prompt_tokens", "completion_tokens"):
        count = value.get(field)
        if type(count) is int and count >= 0:
            usage[field] = count
    return usage or None


def _safe_text_field(value: object) -> str | None:
    return sanitize_text(value) if isinstance(value, str) else None


def _normalized_tool_items(event_name: str, value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    request = event_name == "ToolCallRequestEvent"
    for raw_item in value:
        if not isinstance(raw_item, Mapping):
            continue
        item: dict[str, Any] = {}
        id_field = "id" if request else "call_id"
        identifier = _safe_text_field(raw_item.get(id_field))
        name = _safe_text_field(raw_item.get("name"))
        if identifier is not None:
            item[id_field] = identifier
        if name is not None:
            item["name"] = name
        json_field = "arguments" if request else "content"
        json_value = raw_item.get(json_field)
        if isinstance(json_value, str):
            item[json_field] = (
                sanitize_json_text(json_value)
                if request or json_value.lstrip().startswith(("{", "[", '"'))
                else sanitize_text(json_value)
            )
        elif json_value is not None:
            item[json_field] = redact(json_value)
        if not request and type(raw_item.get("is_error")) is bool:
            item["is_error"] = raw_item["is_error"]
        normalized.append(item)
    return normalized


def normalize_autogen_event_payload(
    event_type: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Select only the semantic AutoGen fields allowed in an audit payload."""

    event_name = event_type.removeprefix("autogen.")
    normalized: dict[str, Any] = {"type": event_name}
    source = _safe_text_field(payload.get("source"))
    if source is not None:
        normalized["source"] = source
    usage = _safe_usage(payload.get("models_usage"))
    if usage is not None:
        normalized["models_usage"] = usage
    if event_name in {"ToolCallRequestEvent", "ToolCallExecutionEvent"}:
        normalized["content"] = _normalized_tool_items(event_name, payload.get("content"))
    elif "content" in payload:
        normalized["content"] = redact(payload["content"])
    if event_name == "HandoffMessage":
        target = _safe_text_field(payload.get("target"))
        if target is not None:
            normalized["target"] = target
    return normalized


def sanitize_stored_event_payload(event_type: str, value: str) -> str:
    """Present current or legacy event JSON without exposing historical secrets."""

    try:
        parsed = _bounded_json_mapping(value)
    except (TypeError, ValueError):
        safe_payload: Any = _rejected_event_payload(event_type)
    else:
        safe_payload = (
            normalize_autogen_event_payload(event_type, parsed)
            if event_type.startswith("autogen.")
            else redact(parsed)
        )
    try:
        return json.dumps(
            safe_payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (OverflowError, RuntimeError, TypeError, ValueError):
        return json.dumps(
            _rejected_event_payload(event_type),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


def sanitize_case_result(result: CaseResult) -> CaseResult:
    """Normalize only terminal free text before storage or artifact publication."""

    payment = result.payment
    if payment is not None and payment.error is not None:
        payment = payment.model_copy(
            update={"error": sanitize_text(payment.error)},
            deep=True,
        )
    errors = [
        error.model_copy(
            update={
                "category": sanitize_text(error.category),
                "message": sanitize_text(error.message),
                "stop_reason": (
                    sanitize_text(error.stop_reason) if error.stop_reason is not None else None
                ),
                "provider_request_id": safe_provider_request_id(error.provider_request_id),
                "details": cast(dict[str, Any], redact(error.details)),
            },
            deep=True,
        )
        for error in result.errors
    ]
    return result.model_copy(
        update={"payment": payment, "errors": errors},
        deep=True,
    )


class RedactingFilter(logging.Filter):
    """Prevent credentials from reaching console or local logging handlers."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = sanitize_text(record.getMessage())
        record.args = ()
        if record.exc_info is not None:
            record.exc_text = sanitize_text("".join(traceback.format_exception(*record.exc_info)))
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = sanitize_text(record.exc_text)
        if record.stack_info:
            record.stack_info = sanitize_text(record.stack_info)
        return True


class ProviderRetryAuditHandler(logging.Handler):
    """Persist OpenAI SDK retry attempts against the current async case context."""

    def emit(self, record: logging.LogRecord) -> None:
        message = sanitize_text(record.getMessage())
        if "retry" not in message.casefold():
            return
        recorder = _CURRENT_AUDIT.get()
        if recorder is not None:
            recorder.record(
                "provider.retry",
                {"logger": record.name, "level": record.levelname, "message": message},
            )


def configure_logging(level: str = "INFO") -> None:
    """Configure concise local logging and an application-wide redaction filter."""

    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[handler],
        force=True,
    )
    # Structured AutoGen events are persisted from run_stream. Avoid duplicating
    # full prompts/responses to the human console at ordinary INFO verbosity.
    logging.getLogger("autogen_core").setLevel(logging.WARNING)
    logging.getLogger("autogen_core.events").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    openai_logger = logging.getLogger("openai")
    openai_logger.setLevel(logging.INFO)
    if not any(isinstance(item, ProviderRetryAuditHandler) for item in openai_logger.handlers):
        openai_logger.addHandler(ProviderRetryAuditHandler())
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        trace.set_tracer_provider(TracerProvider())


class AuditRecorder:
    """Append correlated, redacted events to the mutable workflow database."""

    def __init__(self, workflow_db: Path, case_id: str | None = None) -> None:
        self.workflow_db = workflow_db
        self.case_id = case_id
        self.tracer = trace.get_tracer("invoice_agents")

    def record(
        self,
        event_type: str,
        payload: Any,
        *,
        source_id: str | None = None,
        agent_name: str | None = None,
        tool_call_id: str | None = None,
        db_evidence_id: str | None = None,
        review_id: str | None = None,
        payment_id: str | None = None,
        provider_request_id: str | None = None,
    ) -> str:
        """Persist one event and mirror its timing/correlation in a local span."""

        sanitized_tool_call_id = safe_tool_call_id(tool_call_id)
        if tool_call_id is not None and sanitized_tool_call_id is None:
            raise ValueError("invalid tool call correlation ID")
        event_id = f"evt_{uuid4().hex}"
        safe_payload: Any
        if event_type.startswith("autogen."):
            safe_payload = (
                normalize_autogen_event_payload(event_type, payload)
                if isinstance(payload, Mapping) and _json_value_within_limits(payload)
                else _rejected_event_payload(event_type)
            )
        else:
            safe_payload = redact(payload)
        try:
            encoded = json.dumps(
                safe_payload,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
            )
        except (OverflowError, RuntimeError, TypeError, ValueError):
            safe_payload = (
                _rejected_event_payload(event_type)
                if event_type.startswith("autogen.")
                else VALUE_REJECTED
            )
            encoded = json.dumps(safe_payload, ensure_ascii=False, sort_keys=True)
        sanitized_provider_request_id = safe_provider_request_id(provider_request_id)
        created_at = datetime.now(UTC).isoformat()
        with self.tracer.start_as_current_span(event_type) as span:
            if self.case_id:
                span.set_attribute("invoice.case_id", self.case_id)
            if agent_name:
                span.set_attribute("invoice.agent", agent_name)
            with connect_database(self.workflow_db) as connection:
                connection.execute(
                    "INSERT INTO events("
                    "event_id, case_id, source_id, event_type, agent_name, tool_call_id, "
                    "db_evidence_id, review_id, payment_id, provider_request_id, payload_json, "
                    "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event_id,
                        self.case_id,
                        source_id,
                        event_type,
                        agent_name,
                        sanitized_tool_call_id,
                        db_evidence_id,
                        review_id,
                        payment_id,
                        sanitized_provider_request_id,
                        encoded,
                        created_at,
                    ),
                )
                connection.commit()
        return event_id


@contextmanager
def bind_audit_recorder(recorder: AuditRecorder) -> Iterator[None]:
    """Bind SDK retry logging to one async case without global case state."""

    token = _CURRENT_AUDIT.set(recorder)
    try:
        yield
    finally:
        _CURRENT_AUDIT.reset(token)
