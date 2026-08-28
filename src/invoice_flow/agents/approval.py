"""Approval agent: the VP decision, with a critic in the loop.

Three components, in order of authority:

1. **The policy engine** sets a ceiling. If a critical finding is outstanding,
   APPROVED is off the table -- not discouraged in a prompt, removed as an
   option. Prompts are guidance; this is a control.
2. **The VP agent** reasons within that ceiling and chooses REJECTED or
   ESCALATED, or approves when nothing blocks it.
3. **The critic agent** challenges the draft. If it objects with a different
   decision, the VP gets one chance to revise or defend.

Putting the hard limit in code rather than in the prompt is the difference
between a system that cannot pay a fraudulent invoice and one that merely tends
not to. Both reflection rounds are preserved for the audit trail.
"""

from __future__ import annotations

from typing import Any

from ..config import APPROVAL_THRESHOLD
from ..llm.base import LLMClient, LLMError
from ..llm.prompts import (
    APPROVAL_SYSTEM,
    CRITIC_SYSTEM,
    approval_user_prompt,
    critic_user_prompt,
)
from ..models import (
    ApprovalDecision,
    ApprovalDraft,
    Critique,
    Decision,
    FindingCode,
    InvoiceDraft,
    ReflectionRound,
    ValidationReport,
)

# Findings that make an invoice unpayable no matter how it reads.
_HARD_BLOCKS = {
    FindingCode.VENDOR_MISSING,
    FindingCode.NO_LINE_ITEMS,
    FindingCode.TOTAL_NON_POSITIVE,
    FindingCode.TOTAL_MISSING,
    FindingCode.ITEM_UNKNOWN,
    FindingCode.ITEM_OUT_OF_STOCK,
    FindingCode.STOCK_SHORTFALL,
    FindingCode.QUANTITY_INVALID,
    FindingCode.UNIT_PRICE_INVALID,
    FindingCode.SUBTOTAL_MISMATCH,
    FindingCode.TOTAL_MISMATCH,
    FindingCode.DUE_DATE_BEFORE_INVOICE_DATE,
}

# Findings that need a human but do not condemn the invoice.
_ESCALATION_TRIGGERS = {
    FindingCode.CURRENCY_NON_BASE,
    FindingCode.EXTRACTION_LOW_CONFIDENCE,
    # A revision is not a bad invoice; it needs someone to say which version is
    # real. Auto-preferring the file named `_revised` would be trusting a
    # filename, and the failure mode is a $50,000 "revision".
    FindingCode.DUPLICATE_INVOICE_CONFLICT,
}

# Signals that are individually explicable and collectively a pattern. Any one
# may be innocent -- a rushed vendor writes "URGENT", a small supplier prefers
# wire. Two together is the shape payment fraud actually takes.
_FRAUD_SIGNALS = {
    FindingCode.URGENCY_PRESSURE,
    FindingCode.WIRE_TRANSFER_REQUEST,
    FindingCode.DUE_DATE_UNPARSEABLE,
    FindingCode.AMOUNT_JUST_UNDER_THRESHOLD,
}

# How many fraud signals remove automated approval as an option.
FRAUD_SIGNAL_ESCALATION_THRESHOLD = 2


class PolicyVerdict:
    """What the rule engine permits, before any model reasons about it."""

    def __init__(
        self,
        allowed: set[Decision],
        scrutiny_level: str,
        reasons: list[str],
        fallback: Decision,
    ) -> None:
        self.allowed = allowed
        self.scrutiny_level = scrutiny_level
        self.reasons = reasons
        self.fallback = fallback

    @property
    def approval_blocked(self) -> bool:
        return Decision.APPROVED not in self.allowed


def evaluate_policy(draft: InvoiceDraft, report: ValidationReport) -> PolicyVerdict:
    """Determine which decisions policy permits for this invoice."""
    codes = report.codes()
    reasons: list[str] = []
    allowed = {Decision.APPROVED, Decision.REJECTED, Decision.ESCALATED}

    blocking = sorted(code.value for code in codes & _HARD_BLOCKS)
    if blocking:
        allowed.discard(Decision.APPROVED)
        reasons.append(
            "Approval is blocked by policy: "
            + ", ".join(blocking)
            + ". Choose REJECTED or ESCALATED."
        )

    if report.critical and not blocking:
        allowed.discard(Decision.APPROVED)
        reasons.append(
            f"{len(report.critical)} critical finding(s) outstanding; approval is not available."
        )

    escalators = sorted(code.value for code in codes & _ESCALATION_TRIGGERS)
    if escalators:
        allowed.discard(Decision.APPROVED)
        reasons.append(
            "Requires human judgement rather than automated approval: "
            + ", ".join(escalators)
            + "."
        )

    fraud_signals = sorted(code.value for code in codes & _FRAUD_SIGNALS)
    fraud_pattern = len(fraud_signals) >= FRAUD_SIGNAL_ESCALATION_THRESHOLD
    if fraud_pattern:
        allowed.discard(Decision.APPROVED)
        reasons.append(
            f"{len(fraud_signals)} fraud signals present ({', '.join(fraud_signals)}). "
            "Individually explicable, together a recognised payment-fraud pattern; "
            "automated approval is not available."
        )

    scrutiny = "standard"
    total = draft.total or 0.0
    if fraud_pattern:
        scrutiny = "heightened"
    elif total > APPROVAL_THRESHOLD:
        scrutiny = "heightened"
        reasons.append(
            f"Total of ${total:,.2f} exceeds the ${APPROVAL_THRESHOLD:,.0f} threshold; "
            "heightened scrutiny applies."
        )
    elif FindingCode.AMOUNT_JUST_UNDER_THRESHOLD in codes:
        scrutiny = "heightened"
        reasons.append(
            "Total sits just below the approval threshold; heightened scrutiny applies "
            "despite being under the limit."
        )
    elif len(report.warnings) >= 3:
        scrutiny = "heightened"
        reasons.append(
            f"{len(report.warnings)} warnings on a single invoice; heightened scrutiny applies."
        )

    # Where policy bars approval, escalation is the safer default: a wrong
    # rejection damages a supplier relationship, a wrong approval loses money,
    # and a wrong escalation only costs someone five minutes.
    if Decision.APPROVED in allowed:
        fallback = Decision.APPROVED
    elif blocking:
        fallback = Decision.REJECTED
    else:
        fallback = Decision.ESCALATED

    return PolicyVerdict(allowed, scrutiny, reasons, fallback)


def _enforce(decision: Decision, verdict: PolicyVerdict) -> tuple[Decision, str | None]:
    """Clamp a model decision to what policy permits."""
    if decision in verdict.allowed:
        return decision, None
    return (
        verdict.fallback,
        f"Model proposed {decision.value}, which policy does not permit for this invoice; "
        f"overridden to {verdict.fallback.value}.",
    )


def _rule_based_decision(
    draft: InvoiceDraft, report: ValidationReport, verdict: PolicyVerdict
) -> ApprovalDraft:
    """Decision made without the LLM, used when the model is unreachable.

    Keeps the pipeline useful during an outage instead of failing the invoice.
    """
    if verdict.fallback is Decision.APPROVED:
        rationale = (
            f"No critical findings against invoice {draft.invoice_number or '(unnumbered)'} "
            f"for ${draft.total or 0:,.2f}. All billed items exist in inventory with "
            "sufficient stock and the invoice reconciles internally."
        )
    else:
        blockers = [f.message for f in report.critical] or [f.message for f in report.warnings]
        rationale = (
            f"Decision {verdict.fallback.value} reached by rule engine "
            "(LLM reasoning unavailable). Outstanding issues: "
            + " ".join(blockers[:3])
        )
    return ApprovalDraft(
        decision=verdict.fallback,
        rationale=rationale,
        key_factors=[f.message for f in report.critical[:5]],
        conditions=[],
    )


def base_messages(
    draft: InvoiceDraft, report: ValidationReport, verdict: PolicyVerdict
) -> list[dict[str, Any]]:
    """The VP's opening brief."""
    return [
        {"role": "system", "content": APPROVAL_SYSTEM},
        {
            "role": "user",
            "content": approval_user_prompt(draft, report, verdict.reasons, verdict.scrutiny_level),
        },
    ]


def propose_decision(
    draft: InvoiceDraft,
    report: ValidationReport,
    verdict: PolicyVerdict,
    client: LLMClient,
    previous: ApprovalDraft | None = None,
    critique: Critique | None = None,
    tracer: Any = None,
) -> tuple[ApprovalDraft, list[str], Decision | None]:
    """One VP turn: an opening decision, or a revision answering the critic.

    Returns the decision clamped to what policy permits, notes about that
    clamping, and the model's original decision when it was overridden. A model
    that proposes APPROVED on a blocked invoice is overridden here rather than
    argued with.
    """
    notes: list[str] = []
    messages = base_messages(draft, report, verdict)
    round_index = 0

    if previous is not None and critique is not None:
        round_index = 1
        messages = [
            *messages,
            {"role": "assistant", "content": previous.model_dump_json()},
            {
                "role": "user",
                "content": (
                    "Internal audit has challenged your decision.\n\n"
                    "Objections:\n"
                    + "\n".join(f"  - {objection}" for objection in critique.objections)
                    + (
                        f"\n\nAudit suggests: {critique.suggested_decision.value}"
                        if critique.suggested_decision
                        else ""
                    )
                    + f"\n\nAudit reasoning: {critique.reasoning}\n\n"
                    "Revise your decision if the objection is sound, or restate it with "
                    "reasoning that answers the objection directly. Do not change your "
                    "decision merely because you were challenged."
                ),
            },
        ]

    try:
        proposal = client.complete_structured(
            messages, ApprovalDraft, agent=f"approval.vp.r{round_index}"
        )
    except LLMError as exc:
        if tracer is not None:
            tracer.emit("approval.llm_unavailable", error=str(exc), stage="draft")
        if previous is not None:
            notes.append(f"LLM unavailable during revision ({exc}); prior decision stands.")
            return previous, notes, None
        notes.append(f"LLM unavailable ({exc}); rule-based decision applied.")
        return _rule_based_decision(draft, report, verdict), notes, None

    proposed = proposal.decision
    clamped, override_note = _enforce(proposed, verdict)
    if override_note:
        notes.append(override_note)
        if tracer is not None:
            tracer.emit(
                "approval.policy_override",
                round=round_index,
                proposed=proposed.value,
                enforced=clamped.value,
            )
        proposal = proposal.model_copy(update={"decision": clamped})
        return proposal, notes, proposed
    return proposal, notes, None


def critique_decision(
    draft: InvoiceDraft,
    report: ValidationReport,
    current: ApprovalDraft,
    client: LLMClient,
    round_index: int = 0,
    tracer: Any = None,
) -> Critique | None:
    """One audit turn. `None` means the critic could not be reached."""
    try:
        critique = client.complete_structured(
            [
                {"role": "system", "content": CRITIC_SYSTEM},
                {"role": "user", "content": critic_user_prompt(draft, report, current)},
            ],
            Critique,
            agent=f"approval.critic.r{round_index}",
        )
    except LLMError as exc:
        if tracer is not None:
            tracer.emit("approval.llm_unavailable", error=str(exc), stage="critique")
        return None

    if tracer is not None:
        tracer.emit(
            "approval.critique",
            round=round_index,
            agrees=critique.agrees,
            objections=critique.objections,
            suggested_decision=(
                critique.suggested_decision.value if critique.suggested_decision else None
            ),
        )
    return critique


def critic_is_satisfied(critique: Critique | None, current: ApprovalDraft) -> bool:
    """True when the loop should stop: agreement, or no critic available."""
    if critique is None:
        return True
    return critique.agrees and (
        critique.suggested_decision is None or critique.suggested_decision == current.decision
    )


def finalise(
    current: ApprovalDraft,
    verdict: PolicyVerdict,
    rounds: list[ReflectionRound],
    policy_notes: list[str],
    overridden_from: Decision | None = None,
) -> ApprovalDecision:
    """Assemble the final decision, re-clamped to policy one last time."""
    notes = list(policy_notes)
    final, override_note = _enforce(current.decision, verdict)
    if override_note:
        notes.append(override_note)

    if not rounds:
        rounds = [ReflectionRound(round_index=0, draft=current, critique=None)]

    # A clamped decision must not inherit the rationale that argued for the
    # decision policy just overturned -- an auditor reading "REJECTED" above
    # text recommending payment learns nothing and trusts nothing. The model's
    # own words are preserved verbatim on the reflection rounds.
    # Kept to the headline: the controls that fired are already enumerated in
    # `policy_reasons`, and repeating them here only pads the one line a reader
    # sees first.
    rationale = current.rationale
    if overridden_from is not None:
        rationale = (
            f"{final.value} enforced by policy: the reviewing agent proposed "
            f"{overridden_from.value}, which is not available for this invoice. "
            "The controls that fired are listed in the policy notes, and the "
            "agent's own reasoning is preserved in the review rounds."
        )

    return ApprovalDecision(
        decision=final,
        rationale=rationale,
        key_factors=current.key_factors,
        conditions=current.conditions,
        policy_reasons=notes,
        scrutiny_level=verdict.scrutiny_level,
        rounds=rounds,
    )
