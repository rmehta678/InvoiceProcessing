"""Result downloads are validated and sanitized, never raw file serving."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from factories import make_succeeded_case
from fastapi.testclient import TestClient

from invoice_agents.config import Settings
from invoice_agents.db.core import connect_database
from invoice_agents.db.store import WorkflowStore
from invoice_agents.models import ErrorRecord
from invoice_agents.ui import routes


def _legacy_result(settings: Settings, case_id: str, marker: str) -> str:
    store = WorkflowStore(settings)
    result = store.load_result(case_id)
    assert result is not None and result.payment is not None
    payment = result.payment.model_copy(
        update={"error": f"cookie=session={marker}; preference={marker}-continuation"},
        deep=True,
    )
    error = ErrorRecord(
        category="PROVIDER",
        message=f"provider rejected sk-abcd\u2061efgh_{marker}",
        case_id=case_id,
        stop_reason="PROVIDER_REQUEST_FAILED",
    )
    legacy = result.model_copy(update={"payment": payment, "errors": [error]}, deep=True)
    encoded = legacy.model_dump_json(indent=2)
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET result_json = ? WHERE case_id = ?",
            (encoded, case_id),
        )
        connection.commit()
    return encoded


def test_result_artifact_is_bounded_bound_to_database_and_newly_sanitized(
    client: TestClient,
    settings: Settings,
    ui_workdir: Path,
) -> None:
    case_id = make_succeeded_case(settings)
    marker = "round2-result-marker"
    raw = _legacy_result(settings, case_id, marker)
    artifact_dir = ui_workdir / "artifacts" / "results"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / f"{case_id}.json").write_text(raw, encoding="utf-8")

    response = client.get(f"/cases/{case_id}/result.json")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert marker not in response.text
    payload = response.json()
    assert payload["case_id"] == case_id
    assert payload["payment"]["error"] == "cookie=[REDACTED]"
    assert payload["errors"][0]["message"] == "provider rejected [REDACTED]"
    assert response.text == WorkflowStore(settings).load_result(case_id).model_dump_json()


@pytest.mark.parametrize(
    "mutation",
    [
        "malformed",
        "duplicate",
        "mismatched_case",
        "mismatched_authority",
        "oversize",
        "noncanonical_datetime",
    ],
)
def test_result_artifact_rejects_invalid_or_unbound_content_without_echoing_it(
    client: TestClient,
    settings: Settings,
    ui_workdir: Path,
    mutation: str,
) -> None:
    case_id = make_succeeded_case(settings)
    marker = f"round2-{mutation}-marker"
    raw = _legacy_result(settings, case_id, marker)
    artifact_dir = ui_workdir / "artifacts" / "results"
    artifact_dir.mkdir(parents=True)
    target = artifact_dir / f"{case_id}.json"
    if mutation == "malformed":
        candidate = f'{{"case_id":"{case_id}","marker":"{marker}"'
    elif mutation == "duplicate":
        candidate = f'{{"case_id":"{case_id}","marker":"{marker}",' + raw.lstrip()[1:]
    elif mutation == "mismatched_case":
        payload = json.loads(raw)
        payload["case_id"] = "case_other"
        candidate = json.dumps(payload)
    elif mutation == "mismatched_authority":
        payload = json.loads(raw)
        payload["stop_reason"] = marker
        candidate = json.dumps(payload)
    elif mutation == "oversize":
        candidate = raw + (" " * 1_048_577) + marker
    else:
        payload = json.loads(raw)
        payload["started_at"] = payload["started_at"].replace("Z", "+00:00")
        payload["errors"][0]["message"] = marker
        candidate = json.dumps(payload)
    target.write_text(candidate, encoding="utf-8")

    response = client.get(f"/cases/{case_id}/result.json")

    assert response.status_code == 409
    assert marker not in response.text
    assert "RESULT_ARTIFACT_INVALID" in response.text


@pytest.mark.parametrize("node_kind", ["dangling_symlink", "directory", "fifo"])
def test_result_artifact_rejects_hostile_filesystem_nodes(
    client: TestClient,
    settings: Settings,
    ui_workdir: Path,
    node_kind: str,
) -> None:
    case_id = make_succeeded_case(settings)
    artifact_dir = ui_workdir / "artifacts" / "results"
    artifact_dir.mkdir(parents=True)
    target = artifact_dir / f"{case_id}.json"
    if node_kind == "dangling_symlink":
        target.symlink_to(artifact_dir / "does-not-exist.json")
    elif node_kind == "directory":
        target.mkdir()
    else:
        os.mkfifo(target)

    response = client.get(f"/cases/{case_id}/result.json")

    assert response.status_code == 409
    assert "RESULT_ARTIFACT_INVALID" in response.text


def test_result_artifact_rejects_excessive_nesting_without_parser_crash(
    client: TestClient,
    settings: Settings,
    ui_workdir: Path,
) -> None:
    case_id = make_succeeded_case(settings)
    marker = "round3-deep-artifact-canary"
    artifact_dir = ui_workdir / "artifacts" / "results"
    artifact_dir.mkdir(parents=True)
    target = artifact_dir / f"{case_id}.json"
    target.write_text(
        '{"case_id":"'
        + case_id
        + '","nested":'
        + ("[" * 16_000)
        + '"'
        + marker
        + '"'
        + ("]" * 16_000)
        + "}",
        encoding="utf-8",
    )

    response = client.get(f"/cases/{case_id}/result.json")

    assert response.status_code == 409
    assert marker not in response.text
    assert "RESULT_ARTIFACT_INVALID" in response.text


def test_result_artifact_open_is_descriptor_first_and_nonblocking(
    client: TestClient,
    settings: Settings,
    ui_workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = make_succeeded_case(settings)
    raw = _legacy_result(settings, case_id, "round3-descriptor-canary")
    artifact_dir = ui_workdir / "artifacts" / "results"
    artifact_dir.mkdir(parents=True)
    target = (artifact_dir / f"{case_id}.json").absolute()
    target.write_text(raw, encoding="utf-8")
    real_open = routes.os.open
    real_is_file = Path.is_file

    def checked_open(path: os.PathLike[str] | str, flags: int, *args: object) -> int:
        if Path(path) == target:
            assert flags & os.O_NOFOLLOW
            assert flags & os.O_NONBLOCK
        return real_open(path, flags, *args)

    def forbidden_is_file(path: Path) -> bool:
        if path.absolute() == target:
            raise AssertionError("result route performed a path precheck")
        return real_is_file(path)

    monkeypatch.setattr(routes.os, "open", checked_open)
    monkeypatch.setattr(Path, "is_file", forbidden_is_file)

    response = client.get(f"/cases/{case_id}/result.json")

    assert response.status_code == 200
