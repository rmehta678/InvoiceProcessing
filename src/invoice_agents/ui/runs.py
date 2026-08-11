"""Background run registry over the existing orchestration services.

Runs are keyed by case ID and reuse the same bounded-concurrency semantics as
``process_batch``. The registry only knows whether a task is in flight; case
status and execution authority always come from the workflow database claim
acquired by the orchestration entrypoint used by both CLI and UI callers.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

from invoice_agents.config import Settings
from invoice_agents.db.store import (
    AdmittedCase,
    ExecutionClaim,
    StagedInvoiceAdmission,
    SubmissionAdmission,
    WorkflowStore,
)
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.models import CaseResult, SourceArtifact
from invoice_agents.orchestration import (
    _DURABILITY_PRECEDENCE_STOPS,
    DURABILITY_DEADLINE_SECONDS,
    TERMINAL_WORKER_CLEANUP_GRACE_SECONDS,
    _await_task_despite_cancellation,
    _durably_cancel_unstarted_claim,
    _durably_cancel_unstarted_claims,
    _inspect_claim_durability,
    _select_drained_failure,
    claim_resumable_case,
    resume_case,
    run_prepared_case,
    stage_claimed_invoice_async,
    validate_case_concurrency,
)
from invoice_agents.source_store import snapshot_source


@dataclass(slots=True)
class RunHandle:
    """One in-flight or completed background run for a case."""

    case_id: str
    kind: str  # "process" | "resume" | "batch"
    state: str  # "queued" | "running" | "done" | "unresolved"
    started_at: datetime
    task: asyncio.Task[CaseResult] | None = None
    error: str | None = None


@dataclass(slots=True)
class BatchEntry:
    """One submitted file in a batch; prepare-time failures keep their result."""

    path: Path
    case_id: str
    prepared_started_at: datetime | None
    result: CaseResult | None = None
    state: str = "queued"

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
    state: str = "queued"

    @property
    def running(self) -> bool:
        return self.state in {"queued", "running"}


@dataclass(slots=True, repr=False)
class _LifecycleOwner:
    """Private execution authority retained only while durability is unresolved."""

    claim: ExecutionClaim = field(repr=False)
    started_at: datetime


def _now() -> datetime:
    return datetime.now(UTC)


def _stable_run_error(exc: BaseException) -> str:
    """Map arbitrary child failures to one parent-owned, non-secret registry code."""

    if isinstance(exc, InvoiceAgentsError):
        return f"BACKGROUND_{exc.category.value}_ERROR"
    if isinstance(exc, asyncio.CancelledError):
        return "BACKGROUND_CANCELLED"
    return "UNEXPECTED_RUNTIME_ERROR"


def _select_admission_drained_failure(
    primary: BaseException,
    outcomes: list[object],
    durability: dict[str, BaseException | None],
    admission_failures: list[BaseException],
) -> BaseException:
    """Order execution durability, admission durability, then ordinary failures."""

    for outcome in outcomes:
        if (
            isinstance(outcome, InvoiceAgentsError)
            and outcome.stop_reason in _DURABILITY_PRECEDENCE_STOPS
        ):
            return outcome
    for durability_failure in durability.values():
        if durability_failure is not None:
            return durability_failure
    if admission_failures:
        return admission_failures[0]
    return _select_drained_failure(primary, outcomes, durability)


async def _prepare_claimed_for_launch(
    path: Path,
    settings: Settings,
) -> StagedInvoiceAdmission | CaseResult:
    """Stage with Task 9's fail-closed source primitives without workflow mutation."""

    staged = await stage_claimed_invoice_async(path, settings)
    if type(staged) not in {StagedInvoiceAdmission, CaseResult}:
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "pre-admission staging returned an invalid payload",
            stop_reason="PREPARATION_WORKER_PROTOCOL_INVALID",
        ) from None
    return staged


async def _snapshot_admission_identity(path: Path, settings: Settings) -> SourceArtifact:
    """Resolve content identity before deciding whether isolated staging is needed."""

    return await asyncio.to_thread(
        snapshot_source,
        path,
        settings.source_archive_dir,
        settings.source_max_bytes,
        pdf_policy=settings.pdf_policy(),
    )


class RunRegistry:
    """Track in-flight asyncio runs; storage remains the source of truth."""

    def __init__(self, *, global_limit: int) -> None:
        if type(global_limit) is not int or not 1 <= global_limit <= 8:
            raise ValueError("global model concurrency must be between 1 and 8")
        self._model_slots = asyncio.Semaphore(global_limit)
        self._runs: dict[str, RunHandle] = {}
        self._sources: dict[str, str] = {}
        self._batches: dict[str, BatchState] = {}
        self._lifecycle_owners: dict[str, _LifecycleOwner] = {}
        self._workflow_db: Path | None = None

    def _bind_settings(self, settings: Settings) -> None:
        workflow_db = settings.workflow_db.resolve()
        if self._workflow_db is None:
            self._workflow_db = workflow_db
        elif self._workflow_db != workflow_db:
            raise InvoiceAgentsError(
                ErrorCategory.CONFIGURATION,
                "run registry cannot span different workflow databases",
                stop_reason="DATABASE_AUTHORIZATION_CONTEXT_MISMATCH",
            ) from None

    def _install_lifecycle_owner(
        self,
        case_id: str,
        started_at: datetime,
        claim: ExecutionClaim,
    ) -> _LifecycleOwner:
        existing = self._lifecycle_owners.get(case_id)
        if existing is not None:
            exact_existing = existing.claim == claim and existing.started_at == started_at
            raise InvoiceAgentsError(
                category=ErrorCategory.ORCHESTRATION,
                message=(
                    "case already has the exact private lifecycle owner"
                    if exact_existing
                    else "case has a mismatched private lifecycle owner"
                ),
                case_id=case_id,
                stop_reason=(
                    "CASE_ALREADY_CLAIMED" if exact_existing else "CASE_LIFECYCLE_OWNER_MISMATCH"
                ),
            )
        owner = _LifecycleOwner(claim=claim, started_at=started_at)
        self._lifecycle_owners[case_id] = owner
        return owner

    def _retain_exact_lifecycle_owner(
        self,
        case_id: str,
        started_at: datetime,
        claim: ExecutionClaim,
    ) -> _LifecycleOwner:
        """Retain exact post-commit authority or fail closed on a conflicting owner."""

        existing = self._lifecycle_owners.get(case_id)
        if existing is None:
            existing = _LifecycleOwner(claim=claim, started_at=started_at)
            self._lifecycle_owners[case_id] = existing
            return existing
        if existing.claim != claim or existing.started_at != started_at:
            raise InvoiceAgentsError(
                category=ErrorCategory.ORCHESTRATION,
                message="post-commit repair found a mismatched private lifecycle owner",
                case_id=case_id,
                stop_reason="CASE_LIFECYCLE_OWNER_MISMATCH",
            ) from None
        return existing

    def _drop_lifecycle_owner(self, case_id: str, owner: _LifecycleOwner) -> None:
        if self._lifecycle_owners.get(case_id) is owner:
            del self._lifecycle_owners[case_id]

    def handle(self, case_id: str) -> RunHandle | None:
        return self._runs.get(case_id)

    def is_running(self, case_id: str) -> bool:
        handle = self._runs.get(case_id)
        return handle is not None and handle.state in {"queued", "running"}

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
        memory_batch = self._batches.get(batch_id)
        if self._workflow_db is None:
            return memory_batch
        stored = WorkflowStore(self._workflow_db).load_batch(batch_id)
        if stored is None:
            return None
        if memory_batch is None:
            memory_batch = BatchState(
                batch_id=stored.batch_id,
                created_at=stored.created_at,
                concurrency=stored.concurrency,
                state=stored.state,
            )
            memory_batch.entries = [
                BatchEntry(
                    path=item.source_path,
                    case_id=item.case_id,
                    prepared_started_at=item.started_at,
                    result=item.result,
                    state=item.state,
                )
                for item in stored.entries
            ]
            self._batches[batch_id] = memory_batch
            return memory_batch
        memory_batch.state = stored.state
        by_case = {entry.case_id: entry for entry in memory_batch.entries}
        for stored_entry in stored.entries:
            entry = by_case.get(stored_entry.case_id)
            if entry is not None:
                entry.state = stored_entry.state
                entry.result = stored_entry.result
        return memory_batch

    def _finish(self, handle: RunHandle, source_key: str | None) -> None:
        # The lifecycle owner, not this callback, decides whether terminal
        # durability was proved.  A callback can run after same-tick
        # cancellation, so treating task completion itself as success would
        # orphan the already-issued execution claim.
        if handle.state in {"queued", "running"}:
            handle.state = "unresolved"
        if source_key is not None and self._sources.get(source_key) == handle.case_id:
            del self._sources[source_key]
        task = handle.task
        if task is not None and task.done() and not task.cancelled():
            exc = task.exception()
            if exc is not None:
                handle.error = _stable_run_error(exc)
                exc.__traceback__ = None
                exc.__cause__ = None
                exc.__context__ = None
        # Keep the completed task as the stable wait handle returned to callers.
        # Calling ``exception`` above consumes detached-task failures; retaining
        # the task does not make task completion a durability signal.

    def _finish_batch(self, batch: BatchState) -> None:
        """Retain only stable batch display state after its owner has completed."""

        task = batch.task
        if task is not None and task.done() and not task.cancelled():
            exc = task.exception()
            if exc is not None:
                exc.__traceback__ = None
                exc.__cause__ = None
                exc.__context__ = None
        if task is not None and task.done():
            batch.task = None

    async def start_process(
        self,
        path: Path,
        settings: Settings,
        *,
        submission_id: str,
        force_reprocess: bool = False,
    ) -> str | CaseResult:
        """Atomically admit and launch one source, or reuse its durable target."""

        self._bind_settings(settings)
        store = WorkflowStore(settings)
        source = await _snapshot_admission_identity(path, settings)
        existing = await asyncio.to_thread(
            store.load_submission,
            submission_id,
            "single",
            (source.source_id,),
        )
        if existing is not None:
            return existing.cases[0].case_id
        staged = await _prepare_claimed_for_launch(path, settings)
        if isinstance(staged, CaseResult):
            return staged
        if staged.source != source or staged.submitted_path != path.resolve():
            raise InvoiceAgentsError(
                ErrorCategory.ORCHESTRATION,
                "staged source identity changed during admission",
                case_id=staged.case_id,
                stop_reason="SOURCE_HASH_MISMATCH",
            ) from None
        # Keep issuance in this task: cancellation must never detach an unknown
        # commit containing a fresh execution claim.
        admission = store.claim_submission(
            submission_id,
            "single",
            (staged,),
            force_reprocess=force_reprocess,
        )
        admitted = admission.cases[0]
        if admitted.claim is None:
            return admitted.case_id
        await self._launch(
            admitted.case_id,
            "process",
            run_prepared_case(
                admitted.case_id,
                admitted.started_at,
                settings,
                claim=admitted.claim,
            ),
            claim=admitted.claim,
            source_path=path,
            settings=settings,
            claimed_started_at=admitted.started_at,
        )
        return admitted.case_id

    async def start_resume(self, case_id: str, settings: Settings) -> RunHandle:
        """Claim resume authority before scheduling so storage proves the handoff."""

        self._bind_settings(settings)
        existing = self._runs.get(case_id)
        if existing is not None and existing.state != "done":
            return existing
        previous = WorkflowStore(settings).load_result(case_id)
        claim = claim_resumable_case(case_id, settings)
        return await self._launch(
            case_id,
            "resume",
            resume_case(case_id, settings, claim=claim),
            claim=claim,
            source_path=None,
            settings=settings,
            claimed_started_at=previous.started_at if previous is not None else _now(),
        )

    async def _launch(
        self,
        case_id: str,
        kind: str,
        run: Coroutine[Any, Any, CaseResult],
        *,
        claim: ExecutionClaim,
        source_path: Path | None,
        settings: Settings,
        claimed_started_at: datetime,
    ) -> RunHandle:
        """Install a durability owner before exposing a single-run task."""

        handle = RunHandle(
            case_id=case_id,
            kind=kind,
            state="running",
            started_at=_now(),
        )
        owner = self._install_lifecycle_owner(case_id, claimed_started_at, claim)
        source_key = str(source_path.resolve()) if source_path is not None else None
        if source_key is not None:
            self._sources[source_key] = case_id
        ownership_installed = asyncio.Event()
        launch = asyncio.Event()

        async def run_at_model_boundary() -> CaseResult:
            async with self._model_slots:
                if kind == "process":
                    await asyncio.to_thread(
                        WorkflowStore(settings).mark_admission_running,
                        case_id,
                    )
                return await run

        async def own_lifecycle() -> CaseResult:
            child: asyncio.Task[CaseResult] | None = None
            ownership_installed.set()
            try:
                await launch.wait()
                child = asyncio.create_task(
                    run_at_model_boundary(),
                    name=f"invoice-ui-{kind}-child-{case_id}",
                )
                result = await asyncio.shield(child)
                durability = await _inspect_claim_durability(
                    [(case_id, owner.started_at, owner.claim)],
                    settings,
                )
                durability_failure = durability[case_id]
                if durability_failure is not None:
                    raise durability_failure
                if kind == "process":
                    await asyncio.to_thread(
                        WorkflowStore(settings).mark_admission_terminal,
                        case_id,
                    )
                handle.state = "done"
                self._drop_lifecycle_owner(case_id, owner)
                return result
            except BaseException as primary_failure:
                outcomes: list[object] = []
                cancellation_admission_failures: list[BaseException] = []
                if child is None:
                    run.close()
                elif not child.done():
                    child.cancel()
                    try:
                        await _await_task_despite_cancellation(
                            child,
                            deadline=monotonic() + DURABILITY_DEADLINE_SECONDS,
                            case_id=case_id,
                        )
                    except BaseException as drain_failure:
                        outcomes.append(drain_failure)

                durability = await _inspect_claim_durability(
                    [(case_id, owner.started_at, owner.claim)],
                    settings,
                )
                if durability[case_id] is not None:
                    try:
                        await _durably_cancel_unstarted_claim(
                            case_id,
                            owner.started_at,
                            settings,
                            owner.claim,
                        )
                    except BaseException as terminal_failure:
                        outcomes.append(terminal_failure)
                    durability = await _inspect_claim_durability(
                        [(case_id, owner.started_at, owner.claim)],
                        settings,
                    )

                admission_terminal = durability[case_id] is None
                if kind == "process" and admission_terminal:
                    try:
                        await asyncio.to_thread(
                            WorkflowStore(settings).mark_admission_terminal,
                            case_id,
                        )
                    except BaseException as admission_failure:
                        outcomes.append(admission_failure)
                        cancellation_admission_failures.append(admission_failure)
                        admission_terminal = False
                handle.state = "done" if admission_terminal else "unresolved"
                if admission_terminal:
                    self._drop_lifecycle_owner(case_id, owner)
                selected_failure = _select_admission_drained_failure(
                    primary_failure,
                    outcomes,
                    durability,
                    cancellation_admission_failures,
                )
                selected_failure.__cause__ = None
                selected_failure.__context__ = None
                raise selected_failure from None

        handle.task = asyncio.create_task(
            own_lifecycle(),
            name=f"invoice-ui-{kind}-{case_id}",
        )
        handle.task.add_done_callback(lambda _task: self._finish(handle, source_key))
        self._runs[case_id] = handle
        try:
            await asyncio.shield(ownership_installed.wait())
        except asyncio.CancelledError as cancellation:
            handle.task.cancel()
            try:
                await _await_task_despite_cancellation(
                    handle.task,
                    deadline=(
                        monotonic()
                        + (3 * DURABILITY_DEADLINE_SECONDS)
                        + TERMINAL_WORKER_CLEANUP_GRACE_SECONDS
                    ),
                    case_id=case_id,
                )
            except asyncio.CancelledError:
                raise cancellation from None
            except BaseException as durability_failure:
                raise durability_failure from None
            raise cancellation
        launch.set()
        return handle

    async def start_batch(
        self,
        paths: list[Path],
        settings: Settings,
        concurrency: int | None,
        *,
        submission_id: str,
    ) -> BatchState:
        """Atomically admit a durable batch, then run only newly claimed cases."""

        self._bind_settings(settings)
        selected_concurrency = validate_case_concurrency(
            concurrency,
            settings.case_concurrency,
        )
        store = WorkflowStore(settings)
        source_identities = []
        for path in paths:
            source_identities.append(await _snapshot_admission_identity(path, settings))
        source_ids = tuple(source.source_id for source in source_identities)
        existing = store.load_submission(submission_id, "batch", source_ids)
        if existing is not None:
            assert existing.batch_id is not None
            loaded = self.batch(existing.batch_id)
            if loaded is None:
                raise InvoiceAgentsError(
                    ErrorCategory.DATABASE,
                    "durable submission references a missing batch",
                    stop_reason="PERSISTED_SUBMISSION_INVALID",
                ) from None
            return loaded
        staged_sources: list[StagedInvoiceAdmission] = []
        for path, source_identity in zip(paths, source_identities, strict=True):
            staged = await _prepare_claimed_for_launch(path, settings)
            if isinstance(staged, CaseResult):
                if not staged.errors:
                    raise InvoiceAgentsError(
                        ErrorCategory.ORCHESTRATION,
                        "batch staging returned a failure without exact error evidence",
                        case_id=staged.case_id,
                        stop_reason="PREPARATION_WORKER_PROTOCOL_INVALID",
                    ) from None
                error = staged.errors[0]
                try:
                    category = ErrorCategory(error.category)
                except ValueError:
                    raise InvoiceAgentsError(
                        ErrorCategory.ORCHESTRATION,
                        "batch staging returned an unknown error category",
                        case_id=staged.case_id,
                        stop_reason="PREPARATION_WORKER_PROTOCOL_INVALID",
                    ) from None
                raise InvoiceAgentsError(
                    category,
                    error.message,
                    case_id=staged.case_id,
                    stop_reason=staged.stop_reason,
                ) from None
            if staged.source != source_identity or staged.submitted_path != path.resolve():
                raise InvoiceAgentsError(
                    ErrorCategory.ORCHESTRATION,
                    "staged source identity changed during batch admission",
                    case_id=staged.case_id,
                    stop_reason="SOURCE_HASH_MISMATCH",
                ) from None
            staged_sources.append(staged)
        # This transaction runs synchronously so task cancellation cannot detach an
        # unknown commit and orphan newly issued execution claims.
        admission: SubmissionAdmission | None = None
        batch: BatchState | None = None
        prepared_entries: list[BatchEntry] = []
        try:
            admission = store.claim_submission(
                submission_id,
                "batch",
                tuple(staged_sources),
                concurrency=selected_concurrency,
            )
            assert admission.batch_id is not None
            if all(item.claim is None for item in admission.cases):
                loaded = self.batch(admission.batch_id)
                if loaded is None:
                    raise InvoiceAgentsError(
                        ErrorCategory.DATABASE,
                        "durable admission references a missing batch",
                        stop_reason="PERSISTED_SUBMISSION_INVALID",
                    ) from None
                return loaded
            batch = BatchState(
                batch_id=admission.batch_id,
                created_at=admission.created_at,
                concurrency=selected_concurrency,
                state=(
                    "running"
                    if any(item.state in {"queued", "running"} for item in admission.cases)
                    else "failed"
                    if any(item.state == "failed" for item in admission.cases)
                    else "done"
                ),
                entries=[
                    BatchEntry(
                        path=item.source_path,
                        case_id=item.case_id,
                        prepared_started_at=item.started_at,
                        state=item.state,
                    )
                    for item in admission.cases
                ],
            )
            self._batches[batch.batch_id] = batch
            admitted_by_case: dict[str, AdmittedCase] = {
                item.case_id: item for item in admission.cases
            }
            for entry in batch.entries:
                admitted = admitted_by_case[entry.case_id]
                if admitted.claim is None:
                    continue
                owner = self._install_lifecycle_owner(
                    admitted.case_id,
                    admitted.started_at,
                    admitted.claim,
                )
                prepared_entries.append(entry)
                self._runs[entry.case_id] = RunHandle(
                    case_id=entry.case_id,
                    kind="batch",
                    state="queued",
                    started_at=_now(),
                )
                assert self._lifecycle_owners[entry.case_id] is owner
            if not prepared_entries:
                return batch
            ownership_installed = asyncio.Event()
            batch.task = asyncio.create_task(
                self._run_batch(batch, prepared_entries, settings, ownership_installed),
                name=f"invoice-ui-{batch.batch_id}",
            )
            batch.task.add_done_callback(lambda _task: self._finish_batch(batch))
        except BaseException as primary_failure:
            if batch is not None and batch.task is not None:
                batch.task.cancel()
                await asyncio.gather(batch.task, return_exceptions=True)
            if admission is None:
                raise
            fresh_claims: list[tuple[str, datetime, ExecutionClaim]] = []
            for admitted in admission.cases:
                if admitted.claim is not None:
                    fresh_claims.append((admitted.case_id, admitted.started_at, admitted.claim))
            if not fresh_claims:
                raise
            outcomes = await _durably_cancel_unstarted_claims(fresh_claims, settings)
            durability = await _inspect_claim_durability(fresh_claims, settings)
            admission_failures: list[BaseException] = []
            for admitted in admission.cases:
                if admitted.claim is None:
                    continue
                handle = self._runs.get(admitted.case_id)
                if durability.get(admitted.case_id) is None:
                    try:
                        # Do not detach an unknown admission-state commit while
                        # repairing a post-commit handoff failure.
                        store.mark_admission_terminal(admitted.case_id)
                    except BaseException as admission_failure:
                        outcomes.append(admission_failure)
                        admission_failures.append(admission_failure)
                        if handle is not None:
                            handle.state = "unresolved"
                        try:
                            self._retain_exact_lifecycle_owner(
                                admitted.case_id,
                                admitted.started_at,
                                admitted.claim,
                            )
                        except BaseException as owner_failure:
                            outcomes.append(owner_failure)
                            admission_failures.append(owner_failure)
                    else:
                        try:
                            installed_owner = self._retain_exact_lifecycle_owner(
                                admitted.case_id,
                                admitted.started_at,
                                admitted.claim,
                            )
                        except BaseException as owner_failure:
                            outcomes.append(owner_failure)
                            admission_failures.append(owner_failure)
                            if handle is not None:
                                handle.state = "unresolved"
                        else:
                            if handle is not None:
                                handle.state = "done"
                            self._drop_lifecycle_owner(admitted.case_id, installed_owner)
                else:
                    if handle is not None:
                        handle.state = "unresolved"
                    try:
                        self._retain_exact_lifecycle_owner(
                            admitted.case_id,
                            admitted.started_at,
                            admitted.claim,
                        )
                    except BaseException as owner_failure:
                        outcomes.append(owner_failure)
                        admission_failures.append(owner_failure)
            selected_failure = _select_admission_drained_failure(
                primary_failure,
                list(outcomes),
                durability,
                admission_failures,
            )
            selected_failure.__cause__ = None
            selected_failure.__context__ = None
            raise selected_failure from None
        assert batch is not None and batch.task is not None
        try:
            await asyncio.shield(ownership_installed.wait())
        except asyncio.CancelledError as cancellation:
            batch.task.cancel()
            if ownership_installed.is_set():
                with contextlib.suppress(asyncio.CancelledError):
                    await _await_task_despite_cancellation(
                        batch.task,
                        deadline=(
                            monotonic()
                            + (3 * DURABILITY_DEADLINE_SECONDS)
                            + TERMINAL_WORKER_CLEANUP_GRACE_SECONDS
                        ),
                    )
            else:
                await asyncio.gather(batch.task, return_exceptions=True)
                claims = [
                    (
                        entry.case_id,
                        self._lifecycle_owners[entry.case_id].started_at,
                        self._lifecycle_owners[entry.case_id].claim,
                    )
                    for entry in prepared_entries
                ]
                outcomes = await _durably_cancel_unstarted_claims(claims, settings)
                durability = await _inspect_claim_durability(claims, settings)
                cancellation_admission_failures: list[BaseException] = []
                store = WorkflowStore(settings)
                for entry in prepared_entries:
                    handle = self._runs[entry.case_id]
                    owner = self._lifecycle_owners[entry.case_id]
                    if durability.get(entry.case_id) is not None:
                        handle.state = "unresolved"
                        continue
                    try:
                        store.mark_admission_terminal(entry.case_id)
                    except BaseException as admission_failure:
                        outcomes.append(admission_failure)
                        cancellation_admission_failures.append(admission_failure)
                        handle.state = "unresolved"
                    else:
                        handle.state = "done"
                        self._drop_lifecycle_owner(entry.case_id, owner)
                selected_failure = _select_admission_drained_failure(
                    cancellation,
                    list(outcomes),
                    durability,
                    cancellation_admission_failures,
                )
                if selected_failure is not cancellation:
                    selected_failure.__cause__ = None
                    selected_failure.__context__ = None
                    raise selected_failure from None
            raise cancellation
        return batch

    async def _run_batch(
        self,
        batch: BatchState,
        entries: list[BatchEntry],
        settings: Settings,
        ownership_installed: asyncio.Event,
    ) -> None:
        launch = asyncio.Event()
        store = WorkflowStore(settings)

        async def bounded(entry: BatchEntry) -> None:
            handle = self._runs[entry.case_id]
            owner = self._lifecycle_owners[entry.case_id]
            try:
                await launch.wait()
                async with self._model_slots:
                    await asyncio.to_thread(store.mark_admission_running, entry.case_id)
                    handle.state = "running"
                    entry.state = "running"
                    await run_prepared_case(
                        entry.case_id,
                        owner.started_at,
                        settings,
                        claim=owner.claim,
                    )
                durability = await _inspect_claim_durability(
                    [(entry.case_id, owner.started_at, owner.claim)],
                    settings,
                )
                durability_failure = durability[entry.case_id]
                if durability_failure is not None:
                    raise durability_failure
                await asyncio.to_thread(store.mark_admission_terminal, entry.case_id)
                handle.state = "done"
                entry.state = "done"
            except asyncio.CancelledError:

                async def ensure_durability() -> None:
                    durability = await _inspect_claim_durability(
                        [(entry.case_id, owner.started_at, owner.claim)], settings
                    )
                    if durability[entry.case_id] is not None:
                        await _durably_cancel_unstarted_claim(
                            entry.case_id,
                            owner.started_at,
                            settings,
                            owner.claim,
                        )

                durability_task = asyncio.create_task(
                    ensure_durability(),
                    name=f"invoice-ui-batch-durability-{entry.case_id}",
                )
                await _await_task_despite_cancellation(
                    durability_task,
                    deadline=(
                        monotonic()
                        + (2 * DURABILITY_DEADLINE_SECONDS)
                        + TERMINAL_WORKER_CLEANUP_GRACE_SECONDS
                    ),
                    case_id=entry.case_id,
                )
                raise
            except BaseException as exc:
                handle.error = _stable_run_error(exc)
                raise

        next_entry_index = 0

        async def worker() -> None:
            nonlocal next_entry_index
            while next_entry_index < len(entries):
                entry = entries[next_entry_index]
                next_entry_index += 1
                await bounded(entry)

        tasks = [
            asyncio.create_task(
                worker(),
                name=f"invoice-ui-{batch.batch_id}-worker-{worker_index}",
            )
            for worker_index in range(min(batch.concurrency, len(entries)))
        ]
        ownership_installed.set()
        launch.set()
        selected_failure: BaseException | None = None
        try:
            await asyncio.gather(*tasks)
        except BaseException as primary_failure:
            for task in tasks:
                if not task.done():
                    task.cancel()

            async def drain() -> list[BaseException | None]:
                return await asyncio.gather(*tasks, return_exceptions=True)

            drain_task = asyncio.create_task(
                drain(), name=f"invoice-ui-{batch.batch_id}-cancellation-drain"
            )
            outcomes = await _await_task_despite_cancellation(
                drain_task,
                deadline=(
                    monotonic()
                    + (3 * DURABILITY_DEADLINE_SECONDS)
                    + TERMINAL_WORKER_CLEANUP_GRACE_SECONDS
                ),
            )
            claims = [
                (
                    entry.case_id,
                    self._lifecycle_owners[entry.case_id].started_at,
                    self._lifecycle_owners[entry.case_id].claim,
                )
                for entry in entries
            ]
            durability = await _inspect_claim_durability(claims, settings)
            nonterminal_claims = [claim for claim in claims if durability.get(claim[0]) is not None]
            if nonterminal_claims:
                outcomes.extend(
                    await _durably_cancel_unstarted_claims(nonterminal_claims, settings)
                )
                durability = await _inspect_claim_durability(claims, settings)
            indeterminate_case_ids = {
                outcome.case_id
                for outcome in outcomes
                if isinstance(outcome, InvoiceAgentsError)
                and outcome.stop_reason in _DURABILITY_PRECEDENCE_STOPS
                and outcome.case_id is not None
            }
            admission_failures: list[BaseException] = []
            for entry in entries:
                handle = self._runs[entry.case_id]
                handle.state = (
                    "done"
                    if durability.get(entry.case_id) is None
                    and entry.case_id not in indeterminate_case_ids
                    else "unresolved"
                )
                if handle.state == "done":
                    try:
                        await asyncio.to_thread(store.mark_admission_terminal, entry.case_id)
                    except BaseException as admission_failure:
                        outcomes.append(admission_failure)
                        admission_failures.append(admission_failure)
                        handle.state = "unresolved"
            if not admission_failures and all(
                self._runs[entry.case_id].state == "done" for entry in entries
            ):
                try:
                    terminal_batch = await asyncio.to_thread(store.load_batch, batch.batch_id)
                    if terminal_batch is None:
                        raise InvoiceAgentsError(
                            ErrorCategory.DATABASE,
                            "terminalized batch disappeared from durable storage",
                            stop_reason="PERSISTED_SUBMISSION_INVALID",
                        ) from None
                    terminal_by_case = {item.case_id: item for item in terminal_batch.entries}
                    if any(entry.case_id not in terminal_by_case for entry in entries):
                        raise InvoiceAgentsError(
                            ErrorCategory.DATABASE,
                            "terminalized batch lost a newly admitted entry",
                            stop_reason="PERSISTED_SUBMISSION_INVALID",
                        ) from None
                except BaseException as admission_failure:
                    outcomes.append(admission_failure)
                    admission_failures.append(admission_failure)
                    for entry in entries:
                        self._runs[entry.case_id].state = "unresolved"
                else:
                    batch.state = terminal_batch.state
                    for entry in entries:
                        stored_entry = terminal_by_case[entry.case_id]
                        entry.state = stored_entry.state
                        entry.result = stored_entry.result
                        owner = self._lifecycle_owners[entry.case_id]
                        self._drop_lifecycle_owner(entry.case_id, owner)
            selected_failure = _select_admission_drained_failure(
                primary_failure,
                list(outcomes),
                durability,
                admission_failures,
            )
        if selected_failure is not None:
            selected_failure.__cause__ = None
            selected_failure.__context__ = None
            raise selected_failure from None
        try:
            persisted = await asyncio.to_thread(store.load_batch, batch.batch_id)
        except BaseException:
            for entry in entries:
                self._runs[entry.case_id].state = "unresolved"
            raise
        if persisted is None:
            for entry in entries:
                self._runs[entry.case_id].state = "unresolved"
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "completed batch disappeared from durable storage",
                stop_reason="PERSISTED_SUBMISSION_INVALID",
            ) from None
        batch.state = persisted.state
        persisted_by_case = {item.case_id: item for item in persisted.entries}
        for entry in entries:
            persisted_entry = persisted_by_case.get(entry.case_id)
            if persisted_entry is None:
                self._runs[entry.case_id].state = "unresolved"
                raise InvoiceAgentsError(
                    ErrorCategory.DATABASE,
                    "completed batch lost a durable entry",
                    case_id=entry.case_id,
                    stop_reason="PERSISTED_SUBMISSION_INVALID",
                ) from None
            entry.state = persisted_entry.state
            entry.result = persisted_entry.result
        for entry in entries:
            owner = self._lifecycle_owners[entry.case_id]
            self._drop_lifecycle_owner(entry.case_id, owner)
