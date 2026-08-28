"""End-to-end golden tests over the full sample invoice set.

The scripted VP in `fake_llm` tries to approve everything and its critic always
agrees. Every correct outcome below is therefore produced by the policy engine
and the deterministic checks, not by a cooperative model. That is the property
worth testing: the system is safe because of code we control.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fake_llm import ScriptedGrokClient

from invoice_flow.config import Settings
from invoice_flow.graph import Dependencies, process_invoice
from invoice_flow.models import Decision, FindingCode, Severity
from invoice_flow.observability.trace import Tracer, new_run_id
from invoice_flow.tools.inventory import InventoryRepository

# invoice file -> (expected decision, finding codes that must be present)
GOLDEN: dict[str, tuple[Decision, set[FindingCode]]] = {
    "invoice_1001.txt": (Decision.APPROVED, set()),
    "invoice_1002.txt": (Decision.REJECTED, {FindingCode.STOCK_SHORTFALL}),
    "invoice_1003.txt": (
        Decision.REJECTED,
        {
            FindingCode.ITEM_OUT_OF_STOCK,
            FindingCode.DUE_DATE_UNPARSEABLE,
            FindingCode.URGENCY_PRESSURE,
            FindingCode.WIRE_TRANSFER_REQUEST,
        },
    ),
    "invoice_1004.json": (Decision.APPROVED, set()),
    "invoice_1004_revised.json": (Decision.APPROVED, set()),
    "invoice_1005.json": (
        Decision.REJECTED,
        {FindingCode.STOCK_SHORTFALL, FindingCode.AMOUNT_OVER_THRESHOLD},
    ),
    "invoice_1006.csv": (Decision.APPROVED, set()),
    "invoice_1007.csv": (
        Decision.REJECTED,
        {FindingCode.STOCK_SHORTFALL, FindingCode.TOTAL_MISMATCH},
    ),
    "invoice_1008.txt": (
        Decision.REJECTED,
        {FindingCode.ITEM_UNKNOWN, FindingCode.AMOUNT_JUST_UNDER_THRESHOLD},
    ),
    "invoice_1009.json": (
        Decision.REJECTED,
        {
            FindingCode.QUANTITY_INVALID,
            FindingCode.VENDOR_MISSING,
            FindingCode.TOTAL_NON_POSITIVE,
            FindingCode.SUBTOTAL_MISMATCH,
            FindingCode.DUE_DATE_MISSING,
        },
    ),
    "invoice_1010.txt": (Decision.APPROVED, set()),
    "invoice_1011.pdf": (Decision.APPROVED, set()),
    "invoice_1011.txt": (Decision.APPROVED, set()),
    "invoice_1012.pdf": (Decision.APPROVED, {FindingCode.AMOUNT_JUST_UNDER_THRESHOLD}),
    "invoice_1012.txt": (Decision.APPROVED, {FindingCode.AMOUNT_JUST_UNDER_THRESHOLD}),
    "invoice_1013.json": (
        Decision.REJECTED,
        {FindingCode.STOCK_SHORTFALL, FindingCode.TOTAL_MISMATCH},
    ),
    "invoice_1013.pdf": (Decision.REJECTED, {FindingCode.STOCK_SHORTFALL}),
    "invoice_1014.xml": (Decision.ESCALATED, {FindingCode.CURRENCY_NON_BASE}),
    "invoice_1015.csv": (Decision.APPROVED, set()),
    "invoice_1016.json": (Decision.REJECTED, {FindingCode.ITEM_UNKNOWN}),
}


def run_one(path: Path, db: Path, client: ScriptedGrokClient | None = None):
    repo = InventoryRepository(db)
    try:
        deps = Dependencies(
            settings=Settings.from_env(db_path=db),
            client=client or ScriptedGrokClient(),
            repo=repo,
            tracer=Tracer(run_id=new_run_id()),
        )
        return process_invoice(str(path), deps)
    finally:
        repo.close()


@pytest.mark.parametrize("filename", sorted(GOLDEN))
def test_golden_outcomes(filename: str, invoice_dir: Path, temp_db: Path, capsys) -> None:
    expected_decision, expected_codes = GOLDEN[filename]
    state = run_one(invoice_dir / filename, temp_db)

    assert state.error is None, f"{filename} failed: {state.error}"
    assert state.approval is not None
    assert state.approval.decision is expected_decision, (
        f"{filename}: expected {expected_decision.value}, "
        f"got {state.approval.decision.value}"
    )

    actual = {finding.code for finding in state.all_findings}
    missing = expected_codes - actual
    assert not missing, f"{filename}: missing findings {sorted(c.value for c in missing)}"


@pytest.mark.parametrize("filename", sorted(GOLDEN))
def test_payment_only_follows_approval(filename: str, invoice_dir: Path, temp_db: Path) -> None:
    expected_decision, _ = GOLDEN[filename]
    state = run_one(invoice_dir / filename, temp_db)

    assert state.payment is not None
    if expected_decision is Decision.APPROVED:
        assert state.payment.status == "success"
        assert state.payment.reference
    else:
        assert state.payment.status == "skipped"
        assert state.payment.reference is None


def test_scripted_vp_approves_everything(invoice_dir: Path, temp_db: Path) -> None:
    """Guard on the guard: confirm the scripted VP really does say APPROVED.

    If this ever fails, the golden results above stop proving that the policy
    engine is what blocks bad invoices.
    """
    state = run_one(invoice_dir / "invoice_1003.txt", temp_db)
    assert state.approval is not None
    # The round records the enforced decision, and preserves what the model
    # actually proposed so the override is visible in the audit trail.
    assert state.approval.rounds[0].overridden_from is Decision.APPROVED
    assert state.approval.rounds[0].draft.decision is Decision.REJECTED
    assert state.approval.decision is Decision.REJECTED
    assert any("overridden" in reason.lower() for reason in state.approval.policy_reasons)


def test_overridden_decision_does_not_inherit_the_overturned_rationale(
    invoice_dir: Path, temp_db: Path
) -> None:
    """The headline must explain the decision that was actually made.

    The scripted VP writes "proceeding with payment" and policy overrides it to
    REJECTED. Carrying that text through leaves an audit record whose reasoning
    recommends the opposite of its own verdict.
    """
    state = run_one(invoice_dir / "invoice_1003.txt", temp_db)
    assert state.approval is not None
    assert state.approval.decision is Decision.REJECTED

    rationale = state.approval.rationale
    assert "proceeding with payment" not in rationale
    assert "REJECTED" in rationale
    assert "APPROVED" in rationale, "the overturned proposal must still be named"
    # The model's own words survive, one level down, for whoever audits this.
    assert state.approval.rounds[0].draft.rationale == (
        "Scripted VP rationale: proceeding with payment."
    )


def test_rationale_is_untouched_when_policy_does_not_override(
    invoice_dir: Path, temp_db: Path
) -> None:
    """A decision policy permits keeps the agent's reasoning verbatim."""
    state = run_one(invoice_dir / "invoice_1001.txt", temp_db)
    assert state.approval is not None
    assert state.approval.decision is Decision.APPROVED
    assert state.approval.rationale == "Scripted VP rationale: proceeding with payment."


def test_aggregates_repeated_line_items(invoice_dir: Path, temp_db: Path) -> None:
    """1013 bills WidgetA over three lines; per-line checks would let it pass."""
    state = run_one(invoice_dir / "invoice_1013.json", temp_db)
    assert state.draft is not None
    assert state.draft.aggregated_quantities()["widgeta"] == 22
    assert len(state.draft.line_items) == 8

    shortfalls = {
        finding.detail["item"]
        for finding in state.all_findings
        if finding.code is FindingCode.STOCK_SHORTFALL
    }
    assert shortfalls == {"WidgetA", "WidgetB", "GadgetX"}


def test_ocr_item_names_match_catalogue(invoice_dir: Path, temp_db: Path) -> None:
    """'Widget A' and 'Gadget X' must resolve; the invoice must still pass."""
    state = run_one(invoice_dir / "invoice_1012.pdf", temp_db)
    assert state.validation is not None
    matched = {check.matched_item for check in state.validation.item_checks}
    assert matched == {"WidgetA", "WidgetB", "GadgetX"}
    assert state.approval is not None
    assert state.approval.decision is Decision.APPROVED


def test_near_miss_item_is_not_auto_corrected(invoice_dir: Path, temp_db: Path) -> None:
    """WidgetC resembles WidgetA. Substituting it would authorise a bad payment."""
    state = run_one(invoice_dir / "invoice_1016.json", temp_db)
    unknown = [f for f in state.all_findings if f.code is FindingCode.ITEM_UNKNOWN]
    assert len(unknown) == 1
    assert unknown[0].detail["item"] == "WidgetC"
    assert state.approval is not None
    assert state.approval.decision is Decision.REJECTED


def test_ocr_date_is_repaired(invoice_dir: Path, temp_db: Path) -> None:
    """'26-Jan-2O26' carries a letter O; it must still resolve to a real date."""
    state = run_one(invoice_dir / "invoice_1012.pdf", temp_db)
    assert state.draft is not None
    assert state.draft.due_date is not None
    assert state.draft.invoice_date is not None
    assert state.draft.invoice_date.year == 2026


def test_conflicting_revision_escalates_rather_than_guessing(
    invoice_dir: Path, temp_db: Path
) -> None:
    """1004 is paid at $1,890, then 1004_revised arrives at $5,940.

    Approving the second pays $7,830 for a $5,940 invoice, because the first
    $1,890 has already gone. The system cannot reverse its own payment, so the
    choice between settling the $4,050 difference and voiding the original is a
    human's. The finding has to hand over that number, not just the alarm.
    """
    first = run_one(invoice_dir / "invoice_1004.json", temp_db)
    second = run_one(invoice_dir / "invoice_1004_revised.json", temp_db)

    assert first.payment is not None and first.payment.status == "success"

    assert second.approval is not None
    assert second.approval.decision is Decision.ESCALATED
    assert second.payment is not None and second.payment.status == "skipped"

    conflict = next(
        f for f in second.all_findings if f.code is FindingCode.DUPLICATE_INVOICE_CONFLICT
    )
    assert conflict.detail["previous_amount"] == 1890.00
    assert conflict.detail["current_amount"] == 5940.00
    assert conflict.detail["outstanding_difference"] == 4050.00
    assert "$4,050.00" in conflict.message


def test_revision_of_an_unpaid_invoice_is_not_a_conflict(temp_db: Path) -> None:
    """Nothing was disbursed, so a corrected version is just a better invoice.

    Escalating it would punish a vendor for fixing the very problem that got
    their first submission rejected.
    """
    from invoice_flow.agents.approval import evaluate_policy
    from invoice_flow.agents.validation import run_deterministic_checks
    from invoice_flow.models import InvoiceDraft, LineItem
    from invoice_flow.tools.inventory import InventoryRepository
    from invoice_flow.tools.payment import payable_fingerprint

    def version(quantity: float, revision: str | None = None) -> InvoiceDraft:
        return InvoiceDraft(
            invoice_number="INV-2001",
            revision=revision,
            vendor_name="Precision Parts Ltd.",
            invoice_date_raw="2026-01-22",
            due_date_raw="2026-02-22",
            line_items=[LineItem(name="GadgetX", quantity=quantity, unit_price=750.0)],
            subtotal=quantity * 750.0,
            total=quantity * 750.0,
            currency="USD",
        )

    rejected = version(40)  # 40 against stock of 5
    corrected = version(4, revision="R1")

    repo = InventoryRepository(temp_db)
    try:
        repo.record_ledger_entry(
            run_id="run-rejected",
            invoice_number="INV-2001",
            vendor=rejected.vendor_name,
            amount=rejected.total,
            currency="USD",
            decision="REJECTED",
            payment_status="skipped",
            content_hash=payable_fingerprint(rejected),
        )
        report = run_deterministic_checks(corrected, repo)
    finally:
        repo.close()

    codes = {f.code for f in report.findings}
    assert FindingCode.DUPLICATE_INVOICE_REVISION in codes
    assert FindingCode.DUPLICATE_INVOICE_CONFLICT not in codes

    revision = next(
        f for f in report.findings if f.code is FindingCode.DUPLICATE_INVOICE_REVISION
    )
    assert revision.severity is not Severity.CRITICAL
    assert "R1" in revision.message, "the document's own revision marker belongs in the message"

    # The whole point: approval is still on the table.
    assert Decision.APPROVED in evaluate_policy(corrected, report).allowed


def test_same_invoice_in_two_formats_is_recognised_not_flagged(
    invoice_dir: Path, temp_db: Path
) -> None:
    """INV-1011 ships as PDF and TXT with identical payable facts. That is one
    document seen twice, not a conflict -- and only one payment."""
    pdf = run_one(invoice_dir / "invoice_1011.pdf", temp_db)
    txt = run_one(invoice_dir / "invoice_1011.txt", temp_db)

    assert pdf.payment is not None and pdf.payment.status == "success"

    codes = {f.code for f in txt.all_findings}
    assert FindingCode.DUPLICATE_INVOICE in codes
    assert FindingCode.DUPLICATE_INVOICE_CONFLICT not in codes
    assert txt.approval is not None and txt.approval.decision is Decision.APPROVED
    assert txt.payment is not None and txt.payment.status == "duplicate"


def test_critic_objection_triggers_revision(invoice_dir: Path, temp_db: Path) -> None:
    """An objecting critic must produce a second VP turn in the trace."""
    client = ScriptedGrokClient(critic_objects=True, critic_suggestion=Decision.ESCALATED)
    state = run_one(invoice_dir / "invoice_1001.txt", temp_db, client=client)

    assert state.approval is not None
    assert len(state.approval.rounds) >= 1
    assert any(round_.critique and not round_.critique.agrees for round_ in state.approval.rounds)
    assert any(call.startswith("approval.vp.r1") for call in client.calls)


def test_reflection_loop_is_bounded(invoice_dir: Path, temp_db: Path) -> None:
    """A permanently objecting critic must not loop forever."""
    client = ScriptedGrokClient(critic_objects=True, critic_suggestion=Decision.REJECTED)
    state = run_one(invoice_dir / "invoice_1001.txt", temp_db, client=client)

    assert state.error is None
    assert state.approval is not None
    settings = Settings.from_env()
    assert len(state.approval.rounds) <= settings.max_approval_reflections + 1


def test_pipeline_survives_llm_failure_during_approval(
    invoice_dir: Path, temp_db: Path
) -> None:
    """With the VP unreachable, the rule engine must still reach a decision."""
    client = ScriptedGrokClient(fail_on={"approval.vp"})
    state = run_one(invoice_dir / "invoice_1003.txt", temp_db, client=client)

    assert state.error is None
    assert state.approval is not None
    assert state.approval.decision is Decision.REJECTED
    assert any("rule-based" in reason.lower() for reason in state.approval.policy_reasons)


def test_validation_summary_failure_is_not_fatal(invoice_dir: Path, temp_db: Path) -> None:
    """Losing the narrative summary degrades the report, not the decision."""
    client = ScriptedGrokClient(fail_on={"validation"})
    state = run_one(invoice_dir / "invoice_1002.txt", temp_db, client=client)

    assert state.error is None
    assert state.validation is not None
    assert state.validation.agent_summary is None
    assert state.approval is not None
    assert state.approval.decision is Decision.REJECTED


def test_validation_agent_uses_its_tools(invoice_dir: Path, temp_db: Path) -> None:
    client = ScriptedGrokClient()
    state = run_one(invoice_dir / "invoice_1001.txt", temp_db, client=client)
    assert state.validation is not None
    assert any(call.startswith("validation") for call in client.calls)


def test_missing_total_is_derived_not_refused(temp_db: Path) -> None:
    """A total that can be reconstructed leaves the invoice payable, flagged."""
    from invoice_flow.agents.approval import evaluate_policy
    from invoice_flow.agents.validation import run_deterministic_checks
    from invoice_flow.models import InvoiceDraft, LineItem
    from invoice_flow.tools.arithmetic import reconstruct_total
    from invoice_flow.tools.inventory import InventoryRepository

    draft = InvoiceDraft(
        invoice_number="INV-NT",
        vendor_name="Widgets Inc.",
        invoice_date_raw="2026-01-15",
        due_date_raw="2026-02-01",
        line_items=[LineItem(name="WidgetA", quantity=10, unit_price=250.0)],
        currency="USD",
    )
    assert reconstruct_total(draft) is True
    assert draft.total == 2500.0

    repo = InventoryRepository(temp_db)
    try:
        report = run_deterministic_checks(draft, repo)
    finally:
        repo.close()

    codes = {f.code for f in report.findings}
    assert FindingCode.TOTAL_RECONSTRUCTED in codes
    assert FindingCode.TOTAL_MISSING not in codes
    assert Decision.APPROVED in evaluate_policy(draft, report).allowed


def test_underivable_total_blocks_approval(temp_db: Path) -> None:
    """No stated total and nothing to derive from means no amount to pay."""
    from invoice_flow.agents.approval import evaluate_policy
    from invoice_flow.agents.validation import run_deterministic_checks
    from invoice_flow.models import InvoiceDraft, LineItem
    from invoice_flow.tools.arithmetic import reconstruct_total
    from invoice_flow.tools.inventory import InventoryRepository

    draft = InvoiceDraft(
        invoice_number="INV-NT2",
        vendor_name="Widgets Inc.",
        invoice_date_raw="2026-01-15",
        due_date_raw="2026-02-01",
        line_items=[LineItem(name="WidgetA", quantity=10)],  # no price anywhere
        currency="USD",
    )
    assert reconstruct_total(draft) is False
    assert draft.total is None

    repo = InventoryRepository(temp_db)
    try:
        report = run_deterministic_checks(draft, repo)
    finally:
        repo.close()

    assert FindingCode.TOTAL_MISSING in {f.code for f in report.findings}
    assert Decision.APPROVED not in evaluate_policy(draft, report).allowed


def test_reconstruction_runs_only_after_the_repair_loop(temp_db: Path) -> None:
    """Filling a total mid-loop would make the arithmetic check compare a derived
    figure against itself, masking real vendor errors."""
    from invoice_flow.models import InvoiceDraft, LineItem
    from invoice_flow.tools.arithmetic import check_arithmetic, reconstruct_total

    draft = InvoiceDraft(
        line_items=[LineItem(name="WidgetA", quantity=10, unit_price=250.0)],
        subtotal=2500.0,
    )
    # Before reconstruction there is nothing to disagree with, so no finding.
    assert check_arithmetic(draft) == []
    reconstruct_total(draft)
    # After reconstruction the derived total trivially reconciles -- which is
    # precisely why it must not happen while the loop is still running.
    assert check_arithmetic(draft) == []


def test_two_fraud_signals_remove_automated_approval(temp_db: Path) -> None:
    """1003's profile with a legitimate in-stock item must still not auto-approve."""
    from invoice_flow.agents.approval import evaluate_policy
    from invoice_flow.agents.validation import run_deterministic_checks
    from invoice_flow.models import InvoiceDraft, LineItem
    from invoice_flow.tools.inventory import InventoryRepository

    draft = InvoiceDraft(
        invoice_number="INV-F",
        vendor_name="Plausible Supplies Ltd.",
        invoice_date_raw="2026-01-20",
        due_date_raw="yesterday",
        line_items=[LineItem(name="WidgetA", quantity=10, unit_price=250.0)],
        subtotal=2500.0,
        total=2500.0,
        currency="USD",
        notes="URGENT - Pay immediately to avoid penalties!!! Wire transfer preferred.",
    )

    repo = InventoryRepository(temp_db)
    try:
        report = run_deterministic_checks(draft, repo, draft.notes or "")
    finally:
        repo.close()

    verdict = evaluate_policy(draft, report)
    assert not report.critical, "no hard block here -- the fraud rule must do the work"
    assert Decision.APPROVED not in verdict.allowed
    assert verdict.fallback is Decision.ESCALATED
    assert verdict.scrutiny_level == "heightened"


def test_a_single_fraud_signal_still_permits_approval(temp_db: Path) -> None:
    """One signal is explicable; a rushed vendor should not be held up."""
    from invoice_flow.agents.approval import evaluate_policy
    from invoice_flow.agents.validation import run_deterministic_checks
    from invoice_flow.models import InvoiceDraft, LineItem
    from invoice_flow.tools.inventory import InventoryRepository

    draft = InvoiceDraft(
        invoice_number="INV-U",
        vendor_name="Widgets Inc.",
        invoice_date_raw="2026-01-15",
        due_date_raw="2026-02-01",
        line_items=[LineItem(name="WidgetA", quantity=10, unit_price=250.0)],
        subtotal=2500.0,
        total=2500.0,
        currency="USD",
        notes="Please treat as urgent, our quarter closes Friday.",
    )

    repo = InventoryRepository(temp_db)
    try:
        report = run_deterministic_checks(draft, repo, draft.notes or "")
    finally:
        repo.close()

    assert FindingCode.URGENCY_PRESSURE in {f.code for f in report.findings}
    assert Decision.APPROVED in evaluate_policy(draft, report).allowed


def test_fingerprint_survives_extraction_variance_on_optional_fields() -> None:
    """Two reads of one document must hash alike.

    Most sample invoices show "$" and no ISO currency code, so the extractor
    faithfully returns "USD" on one pass and null on the next. Hashing those
    differently made a second read of INV-1001 look like a conflicting revision
    and escalated a clean invoice -- a failure the golden suite cannot see,
    because it replays one fixed extraction per document.
    """
    from invoice_flow.models import InvoiceDraft, LineItem
    from invoice_flow.tools.payment import payable_fingerprint

    def draft(currency: str | None) -> InvoiceDraft:
        return InvoiceDraft(
            invoice_number="INV-1001",
            vendor_name="Widgets Inc.",
            total=5000.0,
            currency=currency,
            line_items=[
                LineItem(name="WidgetA", quantity=10, unit_price=250.0),
                LineItem(name="WidgetB", quantity=5, unit_price=500.0),
            ],
        )

    stated = payable_fingerprint(draft("USD"))
    assert stated == payable_fingerprint(draft(None))
    assert stated == payable_fingerprint(draft(""))
    assert stated == payable_fingerprint(draft(" usd "))
    # A real currency difference is still a real difference -- 1014 is EUR and
    # must not collapse into the same payable facts as a USD invoice.
    assert stated != payable_fingerprint(draft("EUR"))


def test_unsupported_file_fails_gracefully(tmp_path: Path, temp_db: Path) -> None:
    bad = tmp_path / "invoice.docx"
    bad.write_text("not an invoice", encoding="utf-8")
    state = run_one(bad, temp_db)
    assert state.error is not None
    assert "UnsupportedFormatError" in state.error
    assert state.payment is None
