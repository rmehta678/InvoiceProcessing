"""Fail-closed CLI validation, sanitization, and exit-code contracts."""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest
from click.testing import Result
from typer.testing import CliRunner

from invoice_agents import cli
from invoice_agents.db import cli as database_cli
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.models import CaseStatus

runner = CliRunner()

SECRET = "sk-proj-cli-boundary-secret"
RAW_PATH = "/private/operator/invoices/secret-customer.txt"
TEST_FILE_PATH = str(Path(__file__).resolve())
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
WORKTREE_PATH = str(WORKTREE_ROOT)
WORKSPACE_PATH = str(
    WORKTREE_ROOT.parent.parent if WORKTREE_ROOT.parent.name == ".worktrees" else WORKTREE_ROOT
)
DEBUG_STACK_MAX_FRAMES = 8
DEBUG_STACK_MAX_CHARACTERS = 512
OPERATIONAL_MESSAGE_MAX_CHARACTERS = 512
DEBUG_STACK_SEPARATOR = " -> "
DEBUG_STACK_TRUNCATION_MARKER = "…[TRUNCATED]"


def _assert_concise_failure(
    result: Result,
    *,
    category: str | None = None,
    stop_reason: str | None = None,
    debug: bool = False,
) -> None:
    exit_code = result.exit_code
    output = result.output
    assert exit_code == 1, output
    assert result.stdout == "", output
    assert result.stderr == output
    lines = output.splitlines()
    assert len(lines) == (4 if debug else 1), output
    line = lines[0]
    match = re.fullmatch(
        r"category=([A-Z_]+) stop_reason=([A-Z0-9_]+) message=(.+)",
        line,
    )
    assert match is not None, output
    rendered_category, rendered_stop_reason, rendered_message = match.groups()
    assert re.search(r"\b[A-Za-z_][A-Za-z0-9_-]*=", rendered_message) is None, output
    if category is not None:
        assert rendered_category == category
    if stop_reason is not None:
        assert rendered_stop_reason == stop_reason
    assert SECRET not in output
    assert RAW_PATH not in output
    assert TEST_FILE_PATH not in output
    assert WORKTREE_PATH not in output
    assert WORKSPACE_PATH not in output
    assert "Traceback (most recent call last)" not in output
    assert "InvoiceAgentsError(" not in output
    assert "OperationalError(" not in output
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
        assert forbidden_field not in line, output
    if debug:
        assert lines[1] == "debug_exception_type=ValueError", output
        assert lines[2].startswith("debug_exception_message="), output
        assert re.fullmatch(r"debug_stack=[^=]+", lines[3]) is not None, output


def _raise_expected_operation_error() -> NoReturn:
    raise InvoiceAgentsError(
        ErrorCategory.PROVIDER,
        f"provider rejected api_key={SECRET} at {RAW_PATH}",
        stop_reason="PROVIDER_REJECTED",
    )


def _raise_through_named_frames(frame_names: list[str], message: str) -> NoReturn:
    """Build an exact named traceback chain without adding a terminal helper frame."""

    namespace: dict[str, object] = {"_message": message}
    definitions: list[str] = []
    next_frame: str | None = None
    for frame_name in reversed(frame_names):
        body = (
            "raise ValueError(_message)" if next_frame is None else f"return {next_frame}()"
        )
        definitions.append(f"def {frame_name}():\n    {body}\n")
        next_frame = frame_name
    exec("\n".join(definitions), namespace)
    first_frame = namespace[frame_names[0]]
    assert callable(first_frame)
    first_frame()
    raise AssertionError("named debug frame chain returned")


def _command_arguments(tmp_path: Path, command: str) -> list[str]:
    source = tmp_path / "invoice.txt"
    source.write_text("invoice", encoding="utf-8")
    source_dir = tmp_path / "invoices"
    source_dir.mkdir(exist_ok=True)
    (source_dir / "invoice.txt").write_text("invoice", encoding="utf-8")
    arguments = {
        "root-process": ["--invoice-path", str(source)],
        "process": ["process", "--invoice-path", str(source)],
        "batch": ["batch", "--invoice-dir", str(source_dir)],
        "case-status": ["case", "status", "case_expected_boundary"],
        "review-list": ["review", "list"],
        "review-show": ["review", "show", "review_expected_boundary"],
        "review-decide": [
            "review",
            "decide",
            "review_expected_boundary",
            "--reviewer",
            "reviewer@example.test",
            "--decision",
            "REJECT",
            "--reason",
            "boundary test",
        ],
        "review-resume": ["review", "resume", "case_expected_boundary"],
        "ui": ["ui", "--no-init-db"],
        "contract": ["contract", "--live"],
    }
    return arguments[command]


def _install_downstream_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
    calls: list[str],
) -> list[str]:
    """Fail the command-specific operation after settings and source admission."""

    def settings() -> SimpleNamespace:
        calls.append("settings")
        return SimpleNamespace(
            workflow_db=tmp_path / "workflow.db",
            inventory_db=tmp_path / "inventory.db",
        )

    def fail() -> NoReturn:
        calls.append(command)
        _raise_expected_operation_error()

    async def fail_async(*_args: object, **_kwargs: object) -> NoReturn:
        fail()

    monkeypatch.setattr(cli, "_settings", settings)
    if command in {"root-process", "process"}:
        monkeypatch.setattr(cli, "process_invoice", fail_async)
    elif command == "batch":
        monkeypatch.setattr(cli, "process_batch", fail_async)
    elif command in {"case-status", "review-list", "review-show"}:

        class FailingStore:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            def load_result(self, _case_id: str) -> NoReturn:
                fail()

            def list_reviews(self, *, pending_only: bool) -> NoReturn:
                del pending_only
                fail()

            def load_review(self, _review_id: str) -> NoReturn:
                fail()

        monkeypatch.setattr(cli, "WorkflowStore", FailingStore)
    elif command == "review-decide":
        monkeypatch.setattr(cli, "WorkflowStore", lambda *_args, **_kwargs: object())
        monkeypatch.setattr(cli, "record_human_decision", lambda *_args, **_kwargs: fail())
    elif command == "review-resume":
        claim = object()

        def claim_case(*_args: object, **_kwargs: object) -> object:
            calls.append("claim")
            return claim

        monkeypatch.setattr(cli, "claim_resumable_case", claim_case)
        monkeypatch.setattr(cli, "resume_case", fail_async)
        return ["settings", "claim", command]
    elif command == "ui":
        monkeypatch.setitem(
            sys.modules,
            "uvicorn",
            SimpleNamespace(run=lambda *_args, **_kwargs: fail()),
        )
        monkeypatch.setitem(
            sys.modules,
            "invoice_agents.ui.server",
            SimpleNamespace(create_app=lambda _settings: object()),
        )
    elif command == "contract":
        monkeypatch.setattr(cli, "run_live_contracts", fail_async)
    else:  # pragma: no cover - the parameter list is the closed command inventory.
        raise AssertionError(f"unsupported command fixture: {command}")
    return ["settings", command]


@pytest.mark.parametrize(
    "command",
    [
        "root-process",
        "process",
        "batch",
        "case-status",
        "review-list",
        "review-show",
        "review-decide",
        "review-resume",
        "ui",
        "contract",
    ],
)
def test_every_application_command_uses_one_sanitized_operational_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    calls: list[str] = []
    expected_calls = _install_downstream_failure(monkeypatch, tmp_path, command, calls)

    result = runner.invoke(cli.app, _command_arguments(tmp_path, command))

    _assert_concise_failure(
        result,
        category=ErrorCategory.PROVIDER,
        stop_reason="PROVIDER_REJECTED",
    )
    assert "[REDACTED]" in result.output
    assert calls == expected_calls


@pytest.mark.parametrize("state", ["missing", "empty", "unsupported-only"])
def test_batch_rejects_non_sources_before_settings_admission_or_model_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    directory = tmp_path / state
    expected_reason = "SOURCE_DIRECTORY_MISSING"
    if state != "missing":
        directory.mkdir()
        expected_reason = "SOURCE_DIRECTORY_EMPTY"
    if state == "unsupported-only":
        (directory / "notes.md").write_text("not an invoice", encoding="utf-8")
        (directory / "nested").mkdir()
    calls: list[str] = []

    def forbidden_settings() -> NoReturn:
        calls.append("settings")
        raise AssertionError("settings/admission started before source validation")

    async def forbidden_batch(*_args: object, **_kwargs: object) -> NoReturn:
        calls.append("batch")
        raise AssertionError("batch/model work started before source validation")

    monkeypatch.setattr(cli, "_settings", forbidden_settings)
    monkeypatch.setattr(cli, "process_batch", forbidden_batch)

    result = runner.invoke(cli.app, ["batch", "--invoice-dir", str(directory)])

    _assert_concise_failure(result, category=ErrorCategory.SOURCE, stop_reason=expected_reason)
    assert calls == []
    assert "total=0" not in result.output


def test_supported_invoice_paths_are_sorted_and_reject_missing_or_empty(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(InvoiceAgentsError) as missing_error:
        cli._supported_invoice_paths(missing)
    assert missing_error.value.stop_reason == "SOURCE_DIRECTORY_MISSING"

    directory = tmp_path / "sources"
    directory.mkdir()
    (directory / "ignored.md").write_text("ignored", encoding="utf-8")
    with pytest.raises(InvoiceAgentsError) as empty_error:
        cli._supported_invoice_paths(directory)
    assert empty_error.value.stop_reason == "SOURCE_DIRECTORY_EMPTY"

    expected = [directory / "a.PDF", directory / "b.txt"]
    for path in reversed(expected):
        path.write_bytes(b"invoice")
    assert cli._supported_invoice_paths(directory) == expected


def test_missing_single_source_is_rejected_before_processing_mutates_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing.txt"
    calls: list[Path] = []

    async def forbidden_process(path: Path, _settings: object) -> NoReturn:
        calls.append(path)
        raise AssertionError("admission started for a missing source")

    monkeypatch.setattr(cli, "process_invoice", forbidden_process)

    result = runner.invoke(cli.app, ["process", "--invoice-path", str(missing)])

    _assert_concise_failure(result, category=ErrorCategory.SOURCE, stop_reason="SOURCE_NOT_FOUND")
    assert calls == []


class _MissingStore:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def load_result(self, _case_id: str) -> None:
        return None

    def load_review(self, _review_id: str) -> None:
        return None


@pytest.mark.parametrize(
    ("arguments", "stop_reason"),
    [
        (["case", "status", "case_missing"], "CASE_NOT_FOUND"),
        (["review", "show", "review_missing"], "REVIEW_NOT_FOUND"),
    ],
)
def test_missing_persisted_ids_are_concise_operational_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    stop_reason: str,
) -> None:
    monkeypatch.setattr(cli, "_settings", lambda: SimpleNamespace(workflow_db=tmp_path / "db"))
    monkeypatch.setattr(cli, "WorkflowStore", _MissingStore)

    result = runner.invoke(cli.app, arguments)

    _assert_concise_failure(result, stop_reason=stop_reason)
    assert "NoneType" not in result.output


@pytest.mark.parametrize(
    ("failure", "category", "stop_reason"),
    [
        (sqlite3.OperationalError(f"api_key={SECRET}"), "DATABASE", "DATABASE_OPERATION_FAILED"),
        (PermissionError(f"api_key={SECRET}"), "SOURCE", "FILESYSTEM_OPERATION_FAILED"),
    ],
)
def test_sqlite_and_expected_filesystem_failures_are_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    category: str,
    stop_reason: str,
) -> None:
    source = tmp_path / "invoice.txt"
    source.write_text("invoice", encoding="utf-8")

    def fail_settings() -> NoReturn:
        raise failure

    monkeypatch.setattr(cli, "_settings", fail_settings)

    result = runner.invoke(cli.app, ["process", "--invoice-path", str(source)])

    _assert_concise_failure(result, category=category, stop_reason=stop_reason)


def test_operational_messages_are_single_line_and_have_a_strict_character_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "invoice.txt"
    source.write_text("invoice", encoding="utf-8")

    def fail_settings() -> NoReturn:
        raise InvoiceAgentsError(
            ErrorCategory.PROVIDER,
            f"provider failed\napi_key={SECRET} at {RAW_PATH} {'x' * 10_000}",
            stop_reason="PROVIDER_REJECTED",
        )

    monkeypatch.setattr(cli, "_settings", fail_settings)

    result = runner.invoke(cli.app, ["process", "--invoice-path", str(source)])

    _assert_concise_failure(
        result,
        category=ErrorCategory.PROVIDER,
        stop_reason="PROVIDER_REJECTED",
    )
    message = result.output.split(" message=", 1)[1].rstrip("\n")
    assert len(message) == OPERATIONAL_MESSAGE_MAX_CHARACTERS
    assert message.endswith(DEBUG_STACK_TRUNCATION_MARKER)


@pytest.mark.parametrize(
    "error",
    [
        InvoiceAgentsError(
            ErrorCategory.PROVIDER,
            f"api_key={SECRET}",
            stop_reason=None,
        ),
        InvoiceAgentsError(
            ErrorCategory.PROVIDER,
            "provider failed",
            stop_reason=f"PROVIDER_FAILED api_key={SECRET}",
        ),
    ],
)
def test_invalid_application_error_contract_fails_closed_without_raw_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: InvoiceAgentsError,
) -> None:
    source = tmp_path / "invoice.txt"
    source.write_text("invoice", encoding="utf-8")
    monkeypatch.setattr(cli, "_settings", lambda: (_ for _ in ()).throw(error))

    result = runner.invoke(cli.app, ["process", "--invoice-path", str(source)])

    _assert_concise_failure(
        result,
        category=ErrorCategory.ORCHESTRATION,
        stop_reason="OPERATIONAL_ERROR_CONTRACT_INVALID",
    )


@pytest.mark.parametrize(
    "error",
    [
        InvoiceAgentsError(
            ErrorCategory.PROVIDER,
            "provider failed",
            stop_reason="PROVIDER_FAILED",
        ),
        ValueError("unexpected failure"),
    ],
)
def test_sanitizer_contract_failure_has_no_unsanitized_fallback(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    failure = RuntimeError("sanitizer contract unavailable")
    monkeypatch.setattr(cli, "sanitize_text", lambda _value: (_ for _ in ()).throw(failure))

    with pytest.raises(RuntimeError, match="sanitizer contract unavailable"):
        cli._print_operational_error(error)


def test_batch_rejects_missing_downstream_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "invoices"
    source_dir.mkdir()
    (source_dir / "invoice.txt").write_text("invoice", encoding="utf-8")
    monkeypatch.setattr(cli, "_settings", lambda: object())

    async def no_results(*_args: object, **_kwargs: object) -> list[object]:
        return []

    monkeypatch.setattr(cli, "process_batch", no_results)

    result = runner.invoke(cli.app, ["batch", "--invoice-dir", str(source_dir)])

    _assert_concise_failure(
        result,
        category=ErrorCategory.ORCHESTRATION,
        stop_reason="BATCH_RESULT_CARDINALITY_INVALID",
    )


def test_live_contract_requires_nonempty_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_settings", lambda: object())

    async def no_checks(*_args: object, **_kwargs: object) -> list[object]:
        return []

    monkeypatch.setattr(cli, "run_live_contracts", no_checks)

    result = runner.invoke(cli.app, ["contract", "--live"])

    _assert_concise_failure(
        result,
        category=ErrorCategory.ORCHESTRATION,
        stop_reason="CONTRACT_EVIDENCE_MISSING",
    )


def _database_arguments(tmp_path: Path, operation: str) -> list[str]:
    target = tmp_path / f"{operation}.db"
    arguments = ["db", operation, "--db", str(target)]
    if operation in {"migrate", "verify"}:
        arguments.extend(["--kind", "inventory"])
    elif operation == "reconcile-legacy-authorization":
        arguments.extend(
            [
                "--reviewer",
                "operator@example.test",
                "--reason",
                "permanent disposition",
                "--disposition",
                "PERMANENTLY_QUARANTINED_NON_AUTHORIZING",
                "--confirm",
            ]
        )
    return arguments


@pytest.mark.parametrize(
    ("operation", "dependency"),
    [
        ("migrate", "migrate_database"),
        ("seed", "seed_inventory"),
        ("verify", "verify_database"),
        ("reconcile-legacy-authorization", "reconcile_legacy_authorization"),
    ],
)
def test_nested_database_commands_share_the_sanitized_sqlite_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    dependency: str,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> NoReturn:
        raise sqlite3.OperationalError(f"api_key={SECRET}")

    monkeypatch.setattr(database_cli, dependency, fail)

    result = runner.invoke(cli.app, _database_arguments(tmp_path, operation))

    _assert_concise_failure(
        result,
        category=ErrorCategory.DATABASE,
        stop_reason="DATABASE_OPERATION_FAILED",
    )


def test_standalone_database_cli_rejects_an_unsafe_error_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> NoReturn:
        raise InvoiceAgentsError(
            ErrorCategory.DATABASE,
            "database failed",
            stop_reason=f"DATABASE_FAILED api_key={SECRET}",
        )

    monkeypatch.setattr(database_cli, "migrate_database", fail)
    result = runner.invoke(
        database_cli.app,
        ["migrate", "--db", str(tmp_path / "unsafe.db"), "--kind", "inventory"],
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == (
        "database_operation=migrate status=FAILED "
        "error_code=DATABASE_ERROR_CONTRACT_INVALID\n"
    )
    assert SECRET not in result.output


def test_standalone_database_cli_has_no_sanitizer_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("sanitizer contract unavailable")
    monkeypatch.setattr(
        database_cli,
        "sanitize_text",
        lambda _value: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(RuntimeError, match="sanitizer contract unavailable"):
        database_cli._run_database_operation(
            "verify",
            lambda: (_ for _ in ()).throw(sqlite3.OperationalError("failed")),
        )


def _raise_unexpected() -> NoReturn:
    local_secret = SECRET
    local_path = RAW_PATH
    raise ValueError(f"unexpected api_key={local_secret} at {local_path}")


def test_debug_is_explicit_invocation_local_and_prints_only_sanitized_stack_structure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "invoice.txt"
    source.write_text("invoice", encoding="utf-8")
    absolute_source = str(source.resolve())
    absolute_temporary_directory = str(tmp_path.resolve())

    def raise_unexpected_with_source() -> NoReturn:
        local_secret = SECRET
        local_path = RAW_PATH
        local_source = absolute_source
        raise ValueError(
            f"unexpected api_key={local_secret} at {local_path} from {local_source} "
            f"in {TEST_FILE_PATH} under {WORKTREE_PATH}"
        )

    monkeypatch.setattr(cli, "_settings", raise_unexpected_with_source)

    debug = runner.invoke(cli.app, ["--debug", "process", "--invoice-path", str(source)])
    ordinary_after_debug = runner.invoke(cli.app, ["process", "--invoice-path", str(source)])

    for result, is_debug in ((debug, True), (ordinary_after_debug, False)):
        _assert_concise_failure(
            result,
            category=ErrorCategory.ORCHESTRATION,
            stop_reason="UNEXPECTED_ERROR",
            debug=is_debug,
        )
        for forbidden_path in (
            RAW_PATH,
            TEST_FILE_PATH,
            WORKTREE_PATH,
            absolute_source,
            absolute_temporary_directory,
        ):
            assert forbidden_path not in result.output
        assert "local_secret" not in result.output
        assert "local_path" not in result.output
        assert "local_source" not in result.output
        assert "raise ValueError" not in result.output
    assert "debug_exception_type=ValueError" in debug.output
    assert "debug_stack=" in debug.output
    assert "raise_unexpected_with_source" in debug.output
    debug_message = debug.output.splitlines()[2]
    debug_stack = debug.output.splitlines()[3]
    assert "unexpected" in debug_message
    assert "/" not in debug_stack
    assert "\\" not in debug_stack
    assert "debug_stack=" not in ordinary_after_debug.output
    assert "debug_exception_type=" not in ordinary_after_debug.output


def test_debug_stack_has_exact_frame_and_character_ceilings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single sanitized line is still unsafe when its stack can grow without bound."""

    source = tmp_path / "deep-stack-invoice.txt"
    source.write_text("invoice", encoding="utf-8")
    error_message = (
        f"deep api_key={SECRET} at {RAW_PATH} in {WORKSPACE_PATH} from {source.resolve()}"
    )
    named_frames = [f"_debug_frame_{index:02d}" for index in range(1, 13)]
    monkeypatch.setattr(
        cli,
        "_settings",
        lambda: _raise_through_named_frames(named_frames, error_message),
    )
    frame_limited = runner.invoke(
        cli.app,
        ["--debug", "process", "--invoice-path", str(source)],
    )
    _assert_concise_failure(
        frame_limited,
        category=ErrorCategory.ORCHESTRATION,
        stop_reason="UNEXPECTED_ERROR",
        debug=True,
    )
    frame_payload = frame_limited.output.splitlines()[3].removeprefix("debug_stack=")
    expected_deepest_frames = named_frames[-DEBUG_STACK_MAX_FRAMES:]
    assert frame_payload.split(DEBUG_STACK_SEPARATOR) == expected_deepest_frames
    assert frame_payload == DEBUG_STACK_SEPARATOR.join(expected_deepest_frames)
    assert len(frame_payload) <= DEBUG_STACK_MAX_CHARACTERS

    long_frame_names = [
        f"_debug_character_budget_frame_{'x' * 64}_{index:02d}"
        for index in range(1, DEBUG_STACK_MAX_FRAMES + 1)
    ]
    unbounded_character_payload = DEBUG_STACK_SEPARATOR.join(long_frame_names)
    assert len(unbounded_character_payload) > DEBUG_STACK_MAX_CHARACTERS
    monkeypatch.setattr(
        cli,
        "_settings",
        lambda: _raise_through_named_frames(
            long_frame_names,
            error_message,
        ),
    )
    character_limited = runner.invoke(
        cli.app,
        ["--debug", "process", "--invoice-path", str(source)],
    )
    _assert_concise_failure(
        character_limited,
        category=ErrorCategory.ORCHESTRATION,
        stop_reason="UNEXPECTED_ERROR",
        debug=True,
    )
    character_payload = character_limited.output.splitlines()[3].removeprefix("debug_stack=")
    expected_character_payload = (
        unbounded_character_payload[
            : DEBUG_STACK_MAX_CHARACTERS - len(DEBUG_STACK_TRUNCATION_MARKER)
        ]
        + DEBUG_STACK_TRUNCATION_MARKER
    )
    assert len(character_payload) == DEBUG_STACK_MAX_CHARACTERS
    assert character_payload == expected_character_payload
    assert character_payload.endswith(DEBUG_STACK_TRUNCATION_MARKER)


def test_nested_database_unexpected_error_respects_the_same_invocation_local_debug_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        database_cli, "migrate_database", lambda *_args, **_kwargs: _raise_unexpected()
    )
    arguments = _database_arguments(tmp_path, "migrate")

    debug = runner.invoke(cli.app, ["--debug", *arguments])
    ordinary = runner.invoke(cli.app, arguments)

    for result, is_debug in ((debug, True), (ordinary, False)):
        _assert_concise_failure(
            result,
            category=ErrorCategory.ORCHESTRATION,
            stop_reason="UNEXPECTED_ERROR",
            debug=is_debug,
        )
        assert RAW_PATH not in result.output
        assert TEST_FILE_PATH not in result.output
        assert WORKTREE_PATH not in result.output
    assert "debug_stack=" in debug.output
    debug_stack = debug.output.splitlines()[3]
    assert "/" not in debug_stack
    assert "\\" not in debug_stack
    assert "debug_stack=" not in ordinary.output


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), SystemExit(17)])
def test_process_boundary_never_catches_or_remaps_base_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: BaseException,
) -> None:
    source = tmp_path / "invoice.txt"
    source.write_text("invoice", encoding="utf-8")
    monkeypatch.setattr(cli, "_settings", lambda: object())

    async def interrupt(*_args: object, **_kwargs: object) -> NoReturn:
        raise interruption

    monkeypatch.setattr(cli, "process_invoice", interrupt)

    with pytest.raises(type(interruption)) as caught:
        cli.process_command(source)
    if isinstance(interruption, SystemExit):
        assert caught.value.code == 17


def test_needs_human_remains_exit_two_and_is_not_remapped_by_error_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "invoice.txt"
    source.write_text("invoice", encoding="utf-8")
    result_needing_human = SimpleNamespace(status=CaseStatus.NEEDS_HUMAN)
    monkeypatch.setattr(cli, "_settings", lambda: object())
    monkeypatch.setattr(cli, "_print_result", lambda _result: None)

    async def process(*_args: object, **_kwargs: object) -> object:
        return result_needing_human

    monkeypatch.setattr(cli, "process_invoice", process)

    result = runner.invoke(cli.app, ["process", "--invoice-path", str(source)])

    assert result.exit_code == 2
    assert "category=" not in result.output


def test_contract_not_run_remains_an_explicit_exit_two() -> None:
    result = runner.invoke(cli.app, ["contract"])

    assert result.exit_code == 2
    assert "live contracts NOT RUN" in result.output
