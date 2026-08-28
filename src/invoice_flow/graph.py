"""LangGraph orchestration of the four-stage pipeline.

    load
      |
    extract  <--------------+
      |                     |
    verify_extraction ------+  repair cycle (bounded)
      |
    validate
      |
    vp_decide  <------------+
      |                     |
    audit_critique ---------+  reflection cycle (bounded)
      |
    finalise_approval
      |
    settle

Both self-correction loops are genuine cycles in the graph rather than `for`
statements inside a node. That costs a little state plumbing and buys two
things: every iteration is a checkpointable transition that shows up in the
trace individually, and the loop bounds are visible in the topology instead of
being an implementation detail of an agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import END, START, StateGraph

from .agents import approval as approval_agent
from .agents import ingestion as ingestion_agent
from .agents.payment import run_payment
from .agents.validation import run_validation
from .config import Settings
from .llm.base import LLMClient
from .llm.router import AllProvidersUnavailable
from .models import ExtractionAttempt, ReflectionRound, RunState
from .observability.trace import Tracer
from .tools.inventory import InventoryRepository
from .tools.loaders import load_document


@dataclass
class Dependencies:
    """Everything the nodes need, injected once at graph construction."""

    settings: Settings
    client: LLMClient
    repo: InventoryRepository
    tracer: Tracer


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


def build_graph(deps: Dependencies) -> Any:
    """Compile the pipeline graph with `deps` bound into every node."""

    tracer = deps.tracer

    # -- stage 1: ingestion ------------------------------------------------

    def load(state: RunState) -> dict[str, Any]:
        with tracer.span("load") as span:
            document = load_document(state.invoice_path)
            span["file_format"] = document.file_format
            span["characters"] = len(document.text)
        return {"document": document, "started_at": state.started_at or datetime.now(timezone.utc)}

    def extract(state: RunState) -> dict[str, Any]:
        assert state.document is not None
        round_index = len(state.extraction_attempts)
        with tracer.span("extract", round=round_index):
            draft = ingestion_agent.extract_once(
                state.document, deps.client, state.extraction_attempts
            )
        return {"pending_draft": draft}

    def verify_extraction(state: RunState) -> dict[str, Any]:
        assert state.pending_draft is not None
        attempt = ingestion_agent.assess_attempt(
            state.pending_draft, state.extraction_attempts, deps.settings
        )
        tracer.emit(
            "ingestion.attempt",
            round=attempt.round_index,
            issue_count=len(attempt.issues),
            issues=attempt.issues,
            line_items=len(attempt.draft.line_items),
            invoice_number=attempt.draft.invoice_number,
            accepted=attempt.accepted,
        )

        attempts: list[ExtractionAttempt] = [*state.extraction_attempts, attempt]
        update: dict[str, Any] = {"extraction_attempts": attempts, "pending_draft": None}

        if attempt.accepted:
            draft = ingestion_agent.finalise_confidence(attempt)
            update["draft"] = draft
            if attempt.issues:
                tracer.emit(
                    "ingestion.unresolved",
                    round=attempt.round_index,
                    issues=attempt.issues,
                    note=(
                        "Discrepancies persisted across attempts and are attributed to "
                        "the source document rather than the extractor."
                    ),
                )
        return update

    def route_after_verify(state: RunState) -> str:
        return "accept" if state.draft is not None else "repair"

    # -- stage 2: validation -----------------------------------------------

    def validate(state: RunState) -> dict[str, Any]:
        assert state.draft is not None
        with tracer.span("validate") as span:
            report = run_validation(
                state.draft,
                deps.repo,
                deps.client,
                deps.settings,
                document=state.document,
                tracer=tracer,
            )
            span["critical"] = len(report.critical)
            span["warnings"] = len(report.warnings)
            span["codes"] = sorted(code.value for code in report.codes())
        return {"validation": report}

    # -- stage 3: approval -------------------------------------------------

    def vp_decide(state: RunState) -> dict[str, Any]:
        assert state.draft is not None and state.validation is not None
        verdict = approval_agent.evaluate_policy(state.draft, state.validation)

        notes = list(state.approval_notes)
        if not notes:
            notes = list(verdict.reasons)
            tracer.emit(
                "approval.policy",
                scrutiny_level=verdict.scrutiny_level,
                allowed=[d.value for d in sorted(verdict.allowed, key=lambda x: x.value)],
                approval_blocked=verdict.approval_blocked,
                fallback=verdict.fallback.value,
                reasons=verdict.reasons,
            )

        with tracer.span("vp_decide", round=state.approval_round_index):
            proposal, new_notes, overridden = approval_agent.propose_decision(
                state.draft,
                state.validation,
                verdict,
                deps.client,
                previous=state.approval_draft,
                critique=state.last_critique,
                tracer=tracer,
            )

        rounds = list(state.approval_rounds)
        if state.approval_draft is not None and state.last_critique is not None:
            rounds.append(
                ReflectionRound(
                    round_index=state.approval_round_index,
                    draft=state.approval_draft,
                    critique=state.last_critique,
                    revised=proposal.decision != state.approval_draft.decision,
                    overridden_from=state.approval_overridden_from,
                )
            )

        return {
            "approval_draft": proposal,
            "approval_overridden_from": overridden,
            "approval_notes": notes + new_notes,
            "approval_rounds": rounds,
            "last_critique": None,
            "approval_round_index": state.approval_round_index
            + (1 if state.approval_draft is not None else 0),
        }

    def audit_critique(state: RunState) -> dict[str, Any]:
        assert state.draft is not None and state.validation is not None
        assert state.approval_draft is not None
        with tracer.span("audit_critique", round=state.approval_round_index):
            critique = approval_agent.critique_decision(
                state.draft,
                state.validation,
                state.approval_draft,
                deps.client,
                round_index=state.approval_round_index,
                tracer=tracer,
            )
        return {"last_critique": critique}

    def route_after_critique(state: RunState) -> str:
        assert state.approval_draft is not None
        if approval_agent.critic_is_satisfied(state.last_critique, state.approval_draft):
            return "settled"
        if state.approval_round_index >= deps.settings.max_approval_reflections - 1:
            return "budget_exhausted"
        return "revise"

    def finalise_approval(state: RunState) -> dict[str, Any]:
        assert state.draft is not None and state.validation is not None
        assert state.approval_draft is not None

        verdict = approval_agent.evaluate_policy(state.draft, state.validation)
        rounds = list(state.approval_rounds)
        rounds.append(
            ReflectionRound(
                round_index=state.approval_round_index,
                draft=state.approval_draft,
                critique=state.last_critique,
                overridden_from=state.approval_overridden_from,
            )
        )

        notes = list(state.approval_notes)
        if (
            state.last_critique is not None
            and not approval_agent.critic_is_satisfied(state.last_critique, state.approval_draft)
        ):
            notes.append(
                f"Reflection budget of {deps.settings.max_approval_reflections} rounds "
                "exhausted with audit still objecting; final VP decision stands."
            )

        decision = approval_agent.finalise(
            state.approval_draft,
            verdict,
            rounds,
            notes,
            overridden_from=state.approval_overridden_from,
        )
        tracer.emit(
            "approval.final",
            decision=decision.decision.value,
            scrutiny_level=decision.scrutiny_level,
            rounds=len(decision.rounds),
            rationale=decision.rationale,
        )
        return {"approval": decision}

    # -- stage 4: payment --------------------------------------------------

    def settle(state: RunState) -> dict[str, Any]:
        assert state.draft is not None and state.approval is not None
        with tracer.span("settle") as span:
            receipt = run_payment(
                state.draft,
                state.approval,
                deps.repo,
                run_id=state.run_id,
                source_path=state.invoice_path,
                tracer=tracer,
            )
            span["status"] = receipt.status
        return {"payment": receipt, "finished_at": datetime.now(timezone.utc)}

    # -- wiring ------------------------------------------------------------

    graph = StateGraph(RunState)
    graph.add_node("load", load)
    graph.add_node("extract", extract)
    graph.add_node("verify_extraction", verify_extraction)
    graph.add_node("validate", validate)
    graph.add_node("vp_decide", vp_decide)
    graph.add_node("audit_critique", audit_critique)
    graph.add_node("finalise_approval", finalise_approval)
    graph.add_node("settle", settle)

    graph.add_edge(START, "load")
    graph.add_edge("load", "extract")
    graph.add_edge("extract", "verify_extraction")
    graph.add_conditional_edges(
        "verify_extraction",
        route_after_verify,
        {"repair": "extract", "accept": "validate"},
    )
    graph.add_edge("validate", "vp_decide")
    graph.add_edge("vp_decide", "audit_critique")
    graph.add_conditional_edges(
        "audit_critique",
        route_after_critique,
        {
            "revise": "vp_decide",
            "settled": "finalise_approval",
            "budget_exhausted": "finalise_approval",
        },
    )
    graph.add_edge("finalise_approval", "settle")
    graph.add_edge("settle", END)

    return graph.compile()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def process_invoice(invoice_path: str, deps: Dependencies) -> RunState:
    """Run one invoice end to end and return the final state.

    Failures are captured onto the state rather than raised, so a batch run
    reports a bad document as one failed invoice instead of dying.
    """
    graph = build_graph(deps)
    initial = RunState(
        run_id=deps.tracer.run_id,
        invoice_path=invoice_path,
        started_at=datetime.now(timezone.utc),
    )

    deps.tracer.emit("run.start", invoice_path=invoice_path, model=deps.settings.model)

    try:
        # The recursion limit bounds total node transitions; the two cycles are
        # individually bounded by their routers.
        result = graph.invoke(initial, {"recursion_limit": 50})
    except AllProvidersUnavailable:
        # A misconfigured environment is the operator's problem, not the
        # invoice's. Let it surface as a setup error rather than a rejection.
        raise
    except Exception as exc:  # noqa: BLE001 - surface any failure as run state
        deps.tracer.emit("run.error", error_type=type(exc).__name__, error=str(exc))
        initial.error = f"{type(exc).__name__}: {exc}"
        initial.finished_at = datetime.now(timezone.utc)
        return initial

    state = RunState.model_validate(result) if isinstance(result, dict) else result
    if state.finished_at is None:
        state.finished_at = datetime.now(timezone.utc)

    deps.tracer.emit(
        "run.end",
        decision=state.approval.decision.value if state.approval else None,
        payment_status=state.payment.status if state.payment else None,
        duration_seconds=state.duration_seconds,
        usage=deps.tracer.usage.as_dict(),
    )
    return state
