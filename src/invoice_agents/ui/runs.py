"""Background run registry over the existing orchestration services.

Runs are keyed by case ID and reuse the same bounded-concurrency semantics as
``process_batch``. The registry only knows whether a task is in flight; case
status and execution authority always come from the workflow database claim
acquired by the orchestration entrypoint used by both CLI and UI callers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from invoice_agents.config import Settings
from invoice_agents.db.store import ExecutionClaim, WorkflowStore
from invoice_agents.models import CaseResult, CaseStatus
from invoice_agents.orchestration import (
    EXECUTION_LEASE_SECONDS,
    prepare_claimed_invoice,
    resume_case,
    run_prepared_case,
)


@dataclass(slots=True)
class RunHandle:
    """One in-flight or completed background run for a case."""

    case_id: str
    kind: str  # "process" | "resume" | "batch"
    state: str  # "queued" | "running" | "done"
    started_at: datetime
    task: asyncio.Task[CaseResult] | None = None
    claim: ExecutionClaim | None = None
    error: str | None = None


@dataclass(slots=True)
class BatchEntry:
    """One submitted file in a batch; prepare-time failures keep their result."""

    path: Path
    case_id: str
    prepared_started_at: datetime | None
    claim: ExecutionClaim | None = None
    result: CaseResult | None = None

    @property
    def prepared(self) -> bool:
        return self.prepared_started_at is not None


@dataclass(slots=True)
class BatchState:
    batch_id: str
    created_at: datetime
    concurrency: int
    entries: list[BatchEntry] = field(default_factory=list)
    task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()


def _now() -> datetime:
    return datetime.now(UTC)


class RunRegistry:
    """Track in-flight asyncio runs; storage remains the source of truth."""

    def __init__(self) -> None:
        self._runs: dict[str, RunHandle] = {}
        self._sources: dict[str, str] = {}
        self._batches: dict[str, BatchState] = {}

    def handle(self, case_id: str) -> RunHandle | None:
        return self._runs.get(case_id)

    def is_running(self, case_id: str) -> bool:
        handle = self._runs.get(case_id)
        return handle is not None and handle.state != "done"

    def run_state(self, case_id: str) -> str | None:
        handle = self._runs.get(case_id)
        return handle.state if handle else None

    def run_error(self, case_id: str) -> str | None:
        handle = self._runs.get(case_id)
        return handle.error if handle else None

    def running_case_for_source(self, path: Path) -> str | None:
        """The case currently running for this source file, if any."""

        case_id = self._sources.get(str(path.resolve()))
        if case_id is not None and self.is_running(case_id):
            return case_id
        return None

    def batch(self, batch_id: str) -> BatchState | None:
        return self._batches.get(batch_id)

    def _finish(self, handle: RunHandle, source_key: str | None) -> None:
        handle.state = "done"
        if source_key is not None and self._sources.get(source_key) == handle.case_id:
            del self._sources[source_key]
        task = handle.task
        if task is not None and task.done() and not task.cancelled():
            exc = task.exception()
            if exc is not None:
                handle.error = str(exc)

    async def start_process(self, path: Path, settings: Settings) -> str | CaseResult:
        """Prepare and launch one case; terminal prepare failures are returned as-is."""

        prepared = await asyncio.to_thread(prepare_claimed_invoice, path, settings)
        if isinstance(prepared, CaseResult):
            return prepared
        case_id, started_at, claim = prepared
        self._launch(
            case_id,
            "process",
            run_prepared_case(case_id, started_at, settings, claim=claim),
            claim=claim,
            source_path=path,
        )
        return case_id

    def start_resume(self, case_id: str, settings: Settings) -> RunHandle:
        """Claim resume authority before scheduling so storage proves the handoff."""

        existing = self._runs.get(case_id)
        if existing is not None and existing.state != "done":
            return existing
        claim = WorkflowStore(settings).claim_case_execution(
            case_id,
            frozenset({CaseStatus.NEEDS_HUMAN}),
            EXECUTION_LEASE_SECONDS,
        )
        return self._launch(
            case_id,
            "resume",
            resume_case(case_id, settings, claim=claim),
            claim=claim,
            source_path=None,
        )

    def _launch(
        self,
        case_id: str,
        kind: str,
        run: Coroutine[Any, Any, CaseResult],
        *,
        claim: ExecutionClaim,
        source_path: Path | None,
    ) -> RunHandle:
        handle = RunHandle(
            case_id=case_id,
            kind=kind,
            state="running",
            started_at=_now(),
            claim=claim,
        )
        source_key = str(source_path.resolve()) if source_path is not None else None
        if source_key is not None:
            self._sources[source_key] = case_id
        handle.task = asyncio.create_task(run, name=f"invoice-ui-{kind}-{case_id}")
        handle.task.add_done_callback(lambda _task: self._finish(handle, source_key))
        self._runs[case_id] = handle
        return handle

    async def start_batch(
        self, paths: list[Path], settings: Settings, concurrency: int | None
    ) -> BatchState:
        """Prepare all sources sequentially, then run them with bounded concurrency.

        This mirrors ``process_batch``: sequential preparation gives every later
        identity agent visibility of all submitted representations; runs share one
        semaphore sized by the requested or configured concurrency.
        """

        batch = BatchState(
            batch_id=f"batch_{uuid4().hex[:12]}",
            created_at=_now(),
            concurrency=concurrency or settings.case_concurrency,
        )
        self._batches[batch.batch_id] = batch
        prepared_entries: list[BatchEntry] = []
        for path in paths:
            prepared = await asyncio.to_thread(prepare_claimed_invoice, path, settings)
            if isinstance(prepared, CaseResult):
                entry = BatchEntry(
                    path=path,
                    case_id=prepared.case_id,
                    prepared_started_at=None,
                    result=prepared,
                )
            else:
                entry = BatchEntry(
                    path=path,
                    case_id=prepared[0],
                    prepared_started_at=prepared[1],
                    claim=prepared[2],
                )
                prepared_entries.append(entry)
                self._runs[entry.case_id] = RunHandle(
                    case_id=entry.case_id,
                    kind="batch",
                    state="queued",
                    started_at=_now(),
                    claim=entry.claim,
                )
            batch.entries.append(entry)
        batch.task = asyncio.create_task(
            self._run_batch(batch, prepared_entries, settings),
            name=f"invoice-ui-{batch.batch_id}",
        )
        return batch

    async def _run_batch(
        self, batch: BatchState, entries: list[BatchEntry], settings: Settings
    ) -> None:
        semaphore = asyncio.Semaphore(batch.concurrency)

        async def bounded(entry: BatchEntry) -> None:
            handle = self._runs[entry.case_id]
            async with semaphore:
                handle.state = "running"
                started_at = entry.prepared_started_at
                assert started_at is not None  # only prepared entries reach here
                claim = entry.claim
                assert claim is not None  # retained preparation authority is mandatory
                try:
                    await run_prepared_case(
                        entry.case_id,
                        started_at,
                        settings,
                        claim=claim,
                    )
                except BaseException as exc:
                    # Surfaced verbatim in the matrix; run_prepared_case already
                    # persists its own terminal result for expected failures.
                    handle.error = str(exc)
                finally:
                    handle.state = "done"

        await asyncio.gather(*(bounded(entry) for entry in entries))
