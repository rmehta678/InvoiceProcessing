"""Construct a fresh least-privilege AutoGen Swarm for one invoice case."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import (
    HandoffTermination,
    MaxMessageTermination,
    TextMentionTermination,
)
from autogen_agentchat.teams import Swarm
from autogen_ext.models.openai import OpenAIChatCompletionClient

from invoice_agents.agents.decision_rules import (
    assert_new_review_cycle_permitted,
    blocking_evidence,
    unaddressed_blockers,
    validate_final_decision,
)
from invoice_agents.config import XAI_BASE_URL, XAI_MODEL, Settings
from invoice_agents.db.store import ExecutionClaim, WorkflowStore
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.hitl.service import create_review_request
from invoice_agents.models import (
    Critique,
    DecisionKind,
    ExtractedInvoice,
    FinalDecision,
    PaymentResult,
    RiskAssessment,
    ToolStatus,
)
from invoice_agents.observability.audit import AuditRecorder
from invoice_agents.payment.service import mock_payment
from invoice_agents.source_store import verified_source_path
from invoice_agents.tools.comparison import (
    InventoryReader,
    apply_mapping_evidence,
    build_risk_assessment,
    compare_inventory_evidence,
    compute_invoice_totals,
    find_prior_invoice_candidates,
    recompute_line_extension,
)
from invoice_agents.tools.evidence import extract_invoice_evidence


@dataclass(slots=True)
class AgentCaseContext:
    """Per-case mutable context captured only by this case's tool closures."""

    case_id: str
    settings: Settings
    store: WorkflowStore
    audit: AuditRecorder
    claim: ExecutionClaim
    payment_result: PaymentResult | None = None
    tool_failures: list[str] = field(default_factory=list)

    def invoice(self) -> ExtractedInvoice:
        return self.store.load_current_extraction(self.claim)

    def risk(self) -> RiskAssessment:
        payload = self.store.load_current_comparison(self.claim, "risk")
        if payload is None:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "risk assessment has not been recorded",
                case_id=self.case_id,
                stop_reason="RISK_ASSESSMENT_MISSING",
            )
        return RiskAssessment.model_validate(payload)


def create_model_client(settings: Settings) -> OpenAIChatCompletionClient:
    """Create the sole permitted model client with exact provider/model settings."""

    return OpenAIChatCompletionClient(
        model=XAI_MODEL,
        base_url=XAI_BASE_URL,
        api_key=settings.provider_key(),
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        },
        parallel_tool_calls=False,
        reasoning_effort="high",
        timeout=settings.model_timeout_seconds,
        max_retries=settings.transient_retries,
        # The compatibility suite proves this xAI-safe name representation. It is a
        # fixed choice, not a runtime fallback selected after a rejected request.
        include_name_in_message=False,
        add_name_prefixes=True,
    )


def build_team(
    context: AgentCaseContext,
    model_client: OpenAIChatCompletionClient,
) -> Swarm:
    """Create a fresh Swarm; AutoGen agents and histories are never shared across cases."""

    async def get_case_metadata() -> dict[str, Any]:
        """Return the case/source identity without exposing credentials."""

        invoice = context.invoice()
        result = {
            "case_id": context.case_id,
            "source": invoice.source.model_dump(mode="json"),
            "required_stages": [
                "document_evidence",
                "identity",
                "inventory",
                "financial_risk",
                "independent_critique",
                "approval",
                "payment_if_eligible",
            ],
        }
        context.audit.record("tool.case_metadata", result, agent_name="case_coordinator")
        return result

    async def extract_and_record_invoice() -> dict[str, Any]:
        """Read format-specific source evidence and persist a strict extraction."""

        # Preparation may already have extracted the case so batch identity checks can
        # see all submissions. Re-reading uses the same parser and hash, never a canned value.
        source = context.invoice().source
        verified_source_path(source)
        invoice = extract_invoice_evidence(source)
        context.store.save_extraction(context.case_id, invoice, context.claim)
        payload = invoice.model_dump(mode="json")
        context.audit.record(
            "tool.invoice_extracted",
            payload,
            source_id=invoice.source.source_id,
            agent_name="document_evidence_agent",
        )
        return payload

    async def find_and_record_identity_candidates() -> list[dict[str, Any]]:
        """Find prior hashes, representations, revisions, and identity conflicts."""

        invoice = context.invoice()
        candidates = find_prior_invoice_candidates(context.case_id, invoice, context.store)
        payload = [candidate.model_dump(mode="json") for candidate in candidates]
        record_id = context.store.save_identity(context.case_id, payload, context.claim)
        context.audit.record(
            "tool.identity_candidates",
            payload,
            source_id=invoice.source.source_id,
            agent_name="identity_provenance_agent",
            db_evidence_id=record_id,
        )
        return payload

    async def compare_and_record_inventory() -> dict[str, Any]:
        """Resolve exact/approved aliases, aggregate quantities, and query exact stock rows."""

        invoice = context.invoice()
        reader = InventoryReader(context.settings.inventory_db)
        mappings, comparisons, unresolved = compare_inventory_evidence(invoice, reader)
        errors = [
            result.error
            for result in unresolved.values()
            if result.status is ToolStatus.ERROR and result.error
        ]
        if errors:
            context.tool_failures.extend(errors)
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "; ".join(errors),
                case_id=context.case_id,
                stop_reason="INVENTORY_QUERY_FAILED",
            )
        payload = {
            "comparisons": [comparison.model_dump(mode="json") for comparison in comparisons],
            "unresolved_candidates": {
                item: result.model_dump(mode="json") for item, result in unresolved.items()
            },
        }
        record_id = context.store.save_comparison(
            context.case_id, "inventory", payload, context.claim
        )
        # Mapping outcomes become a new extraction version (§7): canonical_sku only from
        # explicit bases, candidates only from unresolved lookups. v1 stays immutable.
        enriched = apply_mapping_evidence(invoice, mappings, unresolved)
        extraction_id = context.store.save_extraction(context.case_id, enriched, context.claim)
        context.audit.record(
            "tool.inventory_comparison",
            payload,
            source_id=invoice.source.source_id,
            agent_name="inventory_comparison_agent",
            db_evidence_id=record_id,
        )
        context.audit.record(
            "tool.mapping_evidence_recorded",
            {
                "extraction_id": extraction_id,
                "lines": [
                    {
                        "line_id": line.line_id,
                        "canonical_sku": line.canonical_sku,
                        "candidate_skus": line.candidate_skus,
                    }
                    for line in enriched.lines
                ],
            },
            source_id=invoice.source.source_id,
            agent_name="inventory_comparison_agent",
            db_evidence_id=extraction_id,
        )
        return payload

    async def analyze_and_record_financial_risk() -> dict[str, Any]:
        """Recompute amounts/dates/signals and apply explicit human-review policy."""

        invoice = context.invoice()
        stored_inventory = context.store.load_current_comparison(context.claim, "inventory")
        if stored_inventory is None:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "inventory comparison must precede risk analysis",
                case_id=context.case_id,
                stop_reason="INVENTORY_COMPARISON_MISSING",
            )
        from invoice_agents.models import IdentityCandidate, InventoryComparison

        inventory = [
            InventoryComparison.model_validate(item) for item in stored_inventory["comparisons"]
        ]
        identity = [
            IdentityCandidate.model_validate(item)
            for item in context.store.load_current_identity(context.claim)
        ]
        financial = compute_invoice_totals(invoice)
        risk = build_risk_assessment(invoice, inventory, identity, financial, context.settings)
        payload = risk.model_dump(mode="json")
        record_id = context.store.save_comparison(context.case_id, "risk", payload, context.claim)
        context.audit.record(
            "tool.financial_risk_assessment",
            payload,
            source_id=invoice.source.source_id,
            agent_name="financial_risk_agent",
            db_evidence_id=record_id,
        )
        return payload

    async def get_critic_evidence() -> dict[str, Any]:
        """Return immutable specialist evidence for independent challenge."""

        invoice = context.invoice()
        risk = context.risk()
        return {
            "invoice": invoice.model_dump(mode="json"),
            "risk": risk.model_dump(mode="json"),
            "instruction": "Challenge mappings, aggregation, arithmetic, dates, identity, and evidence completeness.",
        }

    async def recheck_inventory_item(item_name: str) -> dict[str, Any]:
        """Independently re-derive one item's exact-row and approved-alias evidence."""

        reader = InventoryReader(context.settings.inventory_db)
        payload = {
            "exact": reader.lookup_inventory_exact(item_name).model_dump(mode="json"),
            "approved_alias": reader.lookup_item_alias(item_name).model_dump(mode="json"),
        }
        context.audit.record(
            "tool.critic_inventory_recheck",
            payload,
            agent_name="independent_critic_agent",
        )
        return payload

    async def recompute_line(quantity: str, unit_price: str) -> dict[str, Any]:
        """Recompute one exact Decimal line extension with no stored state."""

        payload: dict[str, Any] = dict(recompute_line_extension(quantity, unit_price))
        context.audit.record(
            "tool.critic_line_recompute",
            payload,
            agent_name="independent_critic_agent",
        )
        return payload

    async def record_critique(
        supported_findings: list[str],
        challenged_findings: list[str],
        missing_evidence: list[str],
        requested_follow_up: list[str],
        recommended_disposition: Literal["APPROVE", "REJECT", "HOLD", "FAILED"],
        rationale: list[str],
    ) -> dict[str, Any]:
        """Validate and persist the critic's independent structured result."""

        critique = Critique(
            supported_findings=supported_findings,
            challenged_findings=challenged_findings,
            missing_evidence=missing_evidence,
            requested_follow_up=requested_follow_up,
            recommended_disposition=DecisionKind(recommended_disposition),
            rationale=rationale,
        )
        record_id = context.store.save_critique(context.case_id, critique, context.claim)
        payload = critique.model_dump(mode="json")
        context.audit.record(
            "tool.critique_recorded",
            payload,
            source_id=context.invoice().source.source_id,
            agent_name="independent_critic_agent",
            db_evidence_id=record_id,
        )
        return payload

    async def get_approval_package() -> dict[str, Any]:
        """Return evidence, policy triggers, critique, and any resolved human action."""

        invoice = context.invoice()
        risk = context.risk()
        critique = context.store.load_current_critique(context.claim)
        review = context.store.load_current_review(context.claim)
        human = (
            review.human_decision if review is not None and review.status == "RESOLVED" else None
        )
        return {
            "invoice_identity": {
                "invoice_number": invoice.invoice_number.model_dump(mode="json"),
                "vendor": invoice.vendor.model_dump(mode="json"),
                "amount": str(invoice.declared_total)
                if invoice.declared_total is not None
                else None,
                "currency": invoice.currency.model_dump(mode="json"),
            },
            "risk": risk.model_dump(mode="json"),
            "critique": critique.model_dump(mode="json"),
            "review": review.model_dump(mode="json") if review else None,
            "blocking_evidence": [
                blocker.model_dump(mode="json") for blocker in blocking_evidence(risk)
            ],
            "unaddressed_blocking_evidence": [
                blocker.model_dump(mode="json") for blocker in unaddressed_blockers(risk, human)
            ],
        }

    async def persist_human_review(
        agent_recommendation: Literal["APPROVE", "REJECT", "HOLD", "FAILED"],
        agent_rationale: list[str],
    ) -> dict[str, Any]:
        """Create a persisted review package before handing off to the human queue.

        A PENDING review is returned as-is (idempotent). A RESOLVED authorizing review
        no longer blocks a new request: schema v2 supports further review cycles when
        evidence still blocks the case after an authorizing decision. A RESOLVED
        REJECT or REQUEST_CORRECTION is final and re-escalation raises instead of
        looping the queue.
        """

        existing = context.store.load_current_review(context.claim)
        if existing is not None and existing.status == "PENDING":
            return existing.model_dump(mode="json")
        assert_new_review_cycle_permitted(existing, case_id=context.case_id)
        recommendation = DecisionKind(agent_recommendation)
        critique = context.store.load_current_critique(context.claim)
        extra_reasons: list[str] = []
        if (
            recommendation is DecisionKind.APPROVE
            and critique.recommended_disposition is not DecisionKind.APPROVE
        ):
            extra_reasons.append(
                f"independent critic recommends {critique.recommended_disposition} while the "
                "approval agent recommends APPROVE; the disagreement is unresolved"
            )
        review = create_review_request(
            context.case_id,
            context.invoice(),
            context.risk(),
            critique,
            recommendation,
            agent_rationale,
            context.store,
            context.claim,
            extra_reasons=extra_reasons,
        )
        context.audit.record(
            "workflow.review_requested",
            review.model_dump(mode="json"),
            source_id=context.invoice().source.source_id,
            agent_name="approval_agent",
            review_id=review.review_id,
        )
        return review.model_dump(mode="json")

    async def submit_final_decision(
        decision: Literal["APPROVE", "REJECT", "HOLD", "FAILED"],
        reasons: list[str],
        payment_eligible: bool,
    ) -> dict[str, Any]:
        """Persist a schema-valid final decision only after critique and policy checks.

        All rules live in decision_rules.validate_final_decision; this closure only
        loads state, delegates, and persists.
        """

        selected = DecisionKind(decision)
        risk = context.risk()
        critique = context.store.load_current_critique(context.claim)
        review = context.store.load_current_review(context.claim)
        validate_final_decision(
            selected,
            payment_eligible,
            risk,
            critique,
            review,
            case_id=context.case_id,
        )
        human = review.human_decision if review else None
        final = FinalDecision(
            decision=selected,
            reasons=reasons,
            evidence=[ref for line in context.invoice().lines for ref in line.evidence[:1]],
            critic_disposition=critique.recommended_disposition,
            human_outcome=human,
            payment_eligible=payment_eligible,
        )
        context.store.save_final_decision(context.case_id, final, context.claim)
        payload = final.model_dump(mode="json")
        context.audit.record(
            "workflow.final_decision",
            payload,
            source_id=context.invoice().source.source_id,
            agent_name="approval_agent",
        )
        return payload

    async def execute_mock_payment() -> dict[str, Any]:
        """Execute ledger-backed payment only after a persisted eligible approval."""

        payment = mock_payment(
            context.case_id,
            context.invoice(),
            context.store,
            context.settings.workflow_db,
            context.claim,
        )
        context.payment_result = payment
        context.audit.record(
            "tool.mock_payment",
            payment.model_dump(mode="json"),
            source_id=context.invoice().source.source_id,
            agent_name="payment_agent",
            payment_id=payment.payment_id,
        )
        return payment.model_dump(mode="json")

    coordinator = AssistantAgent(
        "case_coordinator",
        model_client,
        description="Starts each case, verifies the required work plan, then delegates evidence extraction.",
        tools=[get_case_metadata],
        handoffs=["document_evidence_agent"],
        system_message=(
            "You coordinate one invoice case. Call get_case_metadata exactly once. Do not decide invoice "
            "facts. Then hand off to document_evidence_agent. Never claim completion yourself."
        ),
        reflect_on_tool_use=True,
        max_tool_iterations=2,
    )
    document = AssistantAgent(
        "document_evidence_agent",
        model_client,
        description="Reads the source format and records a provenance-preserving invoice extraction.",
        tools=[extract_and_record_invoice],
        handoffs=["identity_provenance_agent", "independent_critic_agent"],
        system_message=(
            "Call extract_and_record_invoice. Inspect missing fields, ambiguity, and raw/normalized evidence; "
            "do not repair or approve anything. On the initial pass hand off to identity_provenance_agent. "
            "If the critic sent a focused follow-up, perform it and hand back to independent_critic_agent."
        ),
        reflect_on_tool_use=True,
        max_tool_iterations=2,
    )
    identity_agent = AssistantAgent(
        "identity_provenance_agent",
        model_client,
        description="Classifies duplicate artifacts, representations, revisions, and identity conflicts.",
        tools=[find_and_record_identity_candidates],
        handoffs=["inventory_comparison_agent", "independent_critic_agent"],
        system_message=(
            "Call find_and_record_identity_candidates. Never choose a winning revision. Initially hand off "
            "to inventory_comparison_agent; after a critic follow-up hand back to independent_critic_agent."
        ),
        reflect_on_tool_use=True,
        max_tool_iterations=2,
    )
    inventory_agent = AssistantAgent(
        "inventory_comparison_agent",
        model_client,
        description="Maps exact/approved aliases, aggregates repeated SKUs, and compares read-only stock.",
        tools=[compare_and_record_inventory],
        handoffs=["financial_risk_agent", "independent_critic_agent"],
        system_message=(
            "Call compare_and_record_inventory. Treat fuzzy candidates as unresolved and cite exact queried "
            "rows. Initially hand off to financial_risk_agent; after a critic follow-up hand back to "
            "independent_critic_agent."
        ),
        reflect_on_tool_use=True,
        max_tool_iterations=2,
    )
    analyst = AssistantAgent(
        "financial_risk_agent",
        model_client,
        description="Recomputes Decimal totals and assesses dates, currency, terms, and suspicious signals.",
        tools=[analyze_and_record_financial_risk],
        handoffs=["independent_critic_agent"],
        system_message=(
            "Call analyze_and_record_financial_risk. Distinguish invoice evidence from unavailable database "
            "reconciliations. Hand off to independent_critic_agent."
        ),
        reflect_on_tool_use=True,
        max_tool_iterations=2,
    )
    critic = AssistantAgent(
        "independent_critic_agent",
        model_client,
        description="Independently challenges evidence and disposition before approval.",
        tools=[get_critic_evidence, recheck_inventory_item, recompute_line, record_critique],
        handoffs=[
            "document_evidence_agent",
            "identity_provenance_agent",
            "inventory_comparison_agent",
            "financial_risk_agent",
            "approval_agent",
        ],
        system_message=(
            "Independently inspect the package using get_critic_evidence, then call record_critique exactly "
            "once with concise evidence-backed findings. You can re-derive evidence yourself: "
            "recheck_inventory_item(item_name) returns the exact inventory row and approved-alias provenance, "
            "and recompute_line(quantity, unit_price) returns the exact Decimal extension. Use them to check "
            "at least one disputed mapping or amount instead of narrating. If a concrete discrepancy needs "
            "one focused re-check by a specialist, handoff to that specialist; only one follow-up cycle is "
            "allowed. Otherwise hand off to approval_agent. recommended_disposition must be exactly APPROVE, "
            "REJECT, HOLD, or FAILED. Known unavailable vendor/PO/price/tax/bank tables are disclosed "
            "prototype scope limits, not by themselves missing evidence or a reason to hold. Date-terms and "
            "mapping policies are enforced deterministically and appear in policy_review_reasons; challenge "
            "the evidence, not the configured policy. Do not approve or pay."
        ),
        reflect_on_tool_use=True,
        max_tool_iterations=6,
    )
    approval = AssistantAgent(
        "approval_agent",
        model_client,
        description="Creates a persisted human handoff or records the final structured disposition.",
        tools=[get_approval_package, persist_human_review, submit_final_decision],
        handoffs=["human_reviewer", "payment_agent"],
        system_message=(
            "Call get_approval_package. If review is absent and either policy_review_reasons is non-empty "
            "or you would APPROVE against a disagreeing critic disposition, call persist_human_review with "
            "your original recommendation and rationale (state the disagreement in the rationale), then "
            "handoff to human_reviewer. A resolved REJECT is final: call submit_final_decision with REJECT. "
            "A resolved REQUEST_CORRECTION is final: call submit_final_decision with HOLD. Those rulings "
            "already account for every listed blocker; never request another review over them. Only after "
            "an AUTHORIZING decision (APPROVE, ESTABLISH_MAPPING, SUPERSEDE_REVISION): if "
            "unaddressed_blocking_evidence is non-empty, the resolved human decision did not "
            "identify the exact current blocker-ID set - call "
            "persist_human_review again recommending HOLD, citing that evidence, then handoff to "
            "human_reviewer; if it is empty, obey the authorizing decision and call submit_final_decision. "
            "Without review triggers or critic disagreement, independently call submit_final_decision. "
            "APPROVE must be payment eligible and handed to payment_agent. For REJECT or HOLD, finish with "
            "[CASE_COMPLETE]. Never invent a default decision."
        ),
        reflect_on_tool_use=True,
        max_tool_iterations=4,
    )
    payment = AssistantAgent(
        "payment_agent",
        model_client,
        description="Executes only a persisted eligible approval through the idempotent local payment tool.",
        tools=[execute_mock_payment],
        system_message=(
            "Call execute_mock_payment exactly once. Report its exact status; never translate failure or "
            "NOT_ELIGIBLE into success. Then finish with [CASE_COMPLETE]."
        ),
        reflect_on_tool_use=True,
        max_tool_iterations=2,
    )
    termination = (
        HandoffTermination(target="human_reviewer")
        | TextMentionTermination("[CASE_COMPLETE]")
        | MaxMessageTermination(context.settings.max_messages, include_agent_event=False)
    )
    return Swarm(
        [
            coordinator,
            document,
            identity_agent,
            inventory_agent,
            analyst,
            critic,
            approval,
            payment,
        ],
        termination_condition=termination,
        emit_team_events=True,
    )
