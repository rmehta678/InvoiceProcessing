# Phase 8 - Full-corpus live reconciliation

Run date: 2026-08-06. Status: **COMPLETE - Phase 8 closed on run 2.** Run 1 surfaced one
approval-prompt defect on two artifacts (found, fixed, unit-tested; details in "Run 1 results"),
which triggered the §6.3 restart clause; the full-corpus restart (run 2, approved by the budget
owner) completed with every artifact terminal as scripted, exactly the five expected payments,
and zero review-cycle loops. Completion evidence: `artifacts/verification/workflow_phase8_run2.db`
(the phase database copy), run-1 evidence retained alongside it.

> **Historical evidence only:** This run predates the application-audit remediation. It does not
> prove the final integrated source snapshot, execution fencing, cancellation/orphan recovery,
> UI-request security, global admission, observability, finite critique-cycle, CLI, or browser/SSE
> contracts. It also does not reverify current xAI compatibility. The current release gate and a
> newly approved paid contract run remain pending; no result below is promoted to current release
> evidence by documentation alone.

Mechanics per [`REMEDIATION_PLAN.md`](REMEDIATION_PLAN.md) §6.1: fresh `workflow.db` (prior
database archived as `artifacts/verification/workflow_pre_phase8.db`), verified seeded
`inventory.db` with an empty alias table, `INVOICE_MAX_MESSAGES=40`,
`uv run invoice-agents batch --invoice-dir data/invoices --concurrency 2`, review queue worked
with the decision script below, every decided case resumed, failed cases investigated
individually (batch is never re-run as a retry).

## Reviewer attribution

Decision authority: **michael.schell@skynetai.dev**, who approved decision D4 (token budget) and
pre-approved the execution of the §6.2 expected-decision script below on 2026-08-06
("You have my approval for both of those"). Mechanical execution of the pre-approved script
against each persisted review package was performed by Claude (the implementing agent); every
recorded decision names the delegation explicitly in its reviewer field. Rule for deviations,
fixed before the run: a decision is entered only after checking the actual review package against
the expected reasons below; if the evidence materially deviates from the expectation, the
conservative option (REJECT or REQUEST_CORRECTION) is taken and the deviation is recorded here.

## Known mechanics adjustments, recorded before the run

The §6.2 matrix was written in single-file sequential phrasing ("first"/"second"). The mandated
batch mechanics prepare all 20 sources before any swarm runs, so identity visibility is
**symmetric by design**: both members of each duplicate/revision pair see each other, and
same-vendor cases (INV-1001 / INV-1016, both "Widgets Inc.") see each other as identity
CONFLICT candidates. Consequences, predicted before launch:

1. `invoice_1001.txt`, `invoice_1004.json`, and `invoice_1011.txt` will reach review on identity
   candidates instead of auto-approving; their payment outcomes are preserved through the review
   rulings below. This adds review cycles (~20 instead of the estimated 8-12 resumes) but no
   change to the expected payments ledger.
2. `invoice_1015.csv` and `invoice_1006.csv` reach review on genuine missing terms/currency
   evidence (deterministically verified before the run; see REMEDIATION_PLAN changelog deviation
   2) - the original plan's "clean" phrasing for these two does not hold against the evidence.
3. The 1004 pair: because the original also reviews, the economically correct single payment is
   taken on the **revision** (R1, USD 5,940) via `SUPERSEDE_REVISION`, and the original is
   rejected as superseded - avoiding paying the stale amount, which the sequential phrasing
   would have done. The documented adjustment-design limitation (a paid original would block a
   revision delta payment) is thereby not exercised and remains documented.

## Decision script (recorded before the run) and reconciliation

Expected PAID set (exit criterion): exactly `1001.txt`, `1004_revised.json` (R1),
`1010.txt` (post-mapping), `1011.txt`, `1015.csv` - five payments, zero double payments.

| Artifact | Expected review reasons (batch mechanics) | Pre-approved ruling | Expected terminal state | Actual status / decision / payment | Deviation notes |
|---|---|---|---|---|---|
| 1001.txt | identity CONFLICT (same vendor as 1016) | APPROVE - distinct invoice number; same-vendor conflict belongs to 1016's unknown-item case; evidence exact, stock fits | APPROVE → PAID 5000.00 USD | SUCCEEDED / APPROVE / PAID 5000.00 USD | matched script (run 2) |
| 1002.txt | threshold; GadgetX 20>5; due=invoice date despite Net 30 (ordering + terms-tolerance) | REJECT - stock impossible, terms contradiction, over threshold | REJECT, no payment | SUCCEEDED / REJECT / no payment | matched script (run 2) |
| 1003.txt | threshold; FakeItem 100 vs 0; relative due date; urgent wire language; missing due_date | REJECT - fraud indicators | REJECT, no payment | SUCCEEDED / REJECT / no payment | run 1 looped pre-fix (see Run 1); clean on the fixed system |
| 1004.json | identity POSSIBLE_REVISION (sees R1) | REJECT - superseded by revision R1; do not pay stale amount | REJECT, no payment | SUCCEEDED / REJECT / no payment | matched script (run 2) |
| 1004_revised.json | identity POSSIBLE_REVISION (sees original) | SUPERSEDE_REVISION (superseded case = original) - revision is the payable document | APPROVE → PAID 5940.00 USD | SUCCEEDED / APPROVE / PAID 5940.0 USD | matched script (run 2) |
| 1005.json | threshold; GadgetX 8>5 | REJECT - stock exceedance at threshold amount | REJECT, no payment | SUCCEEDED / REJECT / no payment | matched script (run 2) |
| 1006.csv | USD-convention currency ambiguity; non-recomputable declared zero tax | REQUEST_CORRECTION - reissue with explicit currency and terms | HOLD, no payment | SUCCEEDED / HOLD / no payment | matched script (run 2) |
| 1007.csv | threshold; WidgetA 20>15; WidgetB 15>10; USD 110 total delta; missing terms; currency convention | REJECT - unexplained discrepancy plus stock exceedances | REJECT, no payment | SUCCEEDED / REJECT / no payment | matched script (run 2) |
| 1008.txt | unknown SuperGizmo/MegaSprocket; missing terms | REJECT - items not in inventory | REJECT, no payment | SUCCEEDED / REJECT / no payment | matched script (run 2) |
| 1009.json | missing vendor/due date/terms; quantity -5; inconsistent subtotal | REJECT - invalid payable | REJECT, no payment | SUCCEEDED / REJECT / no payment | run 1 looped pre-fix (see Run 1); clean on the fixed system |
| 1010.txt | "WidgetA (rush order)" unresolved (AMBIGUOUS); non-recomputable declared tax | ESTABLISH_MAPPING "WidgetA (rush order)"=SKU-WIDGET-A - deterministic recompute must clear the blocker (aggregate 12 ≤ 15) before resume | APPROVE → PAID 7185.00 USD (exercises G6 end to end) | SUCCEEDED / APPROVE / PAID 7185.00 USD | recompute.after_human_mapping verified before resume (run 2) |
| 1011.pdf | identity DUPLICATE_REPRESENTATION; missing terms row | REJECT - duplicate representation; TXT is the payable copy | REJECT, no payment | SUCCEEDED / REJECT / no payment | matched script (run 2) |
| 1011.txt | identity DUPLICATE_REPRESENTATION | APPROVE - complete payable copy; PDF rejected as duplicate | APPROVE → PAID 3000.00 USD | SUCCEEDED / APPROVE / PAID 3000.00 USD | matched script (run 2) |
| 1012.pdf | OCR ambiguity; unresolved spaced aliases; identity DUPLICATE_REPRESENTATION | REJECT - duplicate of correction-pending TXT | REJECT, no payment | SUCCEEDED / REJECT / no payment | matched script (run 2) |
| 1012.txt | OCR date/amount ambiguity; unresolved "Widget A"/"Gadget X" | REQUEST_CORRECTION - no mapping is established from an OCR-damaged document; reissue clean copy | HOLD, no payment | SUCCEEDED / HOLD / no payment | matched script (run 2) |
| 1013.json | threshold; aggregates WidgetA 22>15, WidgetB 18>10, GadgetX 9>5; USD 50 total delta | REJECT - aggregate stock impossible plus unexplained delta | REJECT, no payment | SUCCEEDED / REJECT / no payment | matched script (run 2) |
| 1013.pdf | same as json plus identity DUPLICATE_REPRESENTATION | REJECT - duplicate carrying the same defects | REJECT, no payment | SUCCEEDED / REJECT / no payment | matched script (run 2) |
| 1014.xml | EUR has no FX/threshold policy | REJECT (D5 ruling) - no approved FX policy; vendor may reissue in USD | REJECT, no payment | SUCCEEDED / REJECT / no payment | matched script (run 2) |
| 1015.csv | missing payment terms; USD-convention currency ambiguity | APPROVE - arithmetic exact, quantities fit; USD operating convention accepted; terms absence is this CSV format's known limitation | APPROVE → PAID 6500.00 USD | SUCCEEDED / APPROVE / PAID 6500.00 USD | matched script (run 2) |
| 1016.json | WidgetC unknown; identity CONFLICT (same vendor as 1001) | REJECT - unknown item cannot be validated | REJECT, no payment | SUCCEEDED / REJECT / no payment | matched script (run 2) |

## Run 2 results (2026-08-06, the completion run)

Fresh workflow and inventory databases (run-1 pair archived as
`artifacts/verification/workflow_phase8_run1.db` / `inventory_phase8_run1.db`). Batch:
`total=20 failed_or_incomplete=0 needs_human=20`, review reasons again matching the script's
expected markers for all 20 artifacts - verified mechanically this run: the queue driver
compared every review package against the script's expected markers before recording a decision
and reported `decided=20 mismatched=0`. All 20 resumes exited 0 on the first attempt:
13 `DECISION_REJECT`, 2 `DECISION_HOLD`, 5 `APPROVED_PAYMENT_RECORDED`.

- Payments ledger: exactly the five expected rows (5000.00 + 5940.0 + 7185.00 + 3000.00 +
  6500.00 USD), zero `DUPLICATE`, zero double payments.
- `review_requests` contains **zero sequences greater than 1**: the run-1 loop class is gone on
  the fixed system; 1003 and 1009 terminated as direct forced REJECTs.
- `recompute.after_human_mapping` ran once (1010) between the mapping decision and the resume.
- The independent critic exercised its granular tools live throughout:
  `tool.critic_inventory_recheck` 26 calls and `tool.critic_line_recompute` 26 calls across all
  20 cases - closing the R0 exit criterion's live-case requirement with margin.
- Run 2 usage: prompt=1,652,006 completion=52,889 model_calls=436. Phase total across both runs:
  ~3.69M prompt tokens, 896 model calls (versus the plan's 0.8-1.0M estimate; overrun analysis
  in REMEDIATION_PLAN changelog, restart explicitly approved by the budget owner).

## Exit criteria (from §6.3)

- [x] All 20 artifacts reached a terminal or resolved-and-resumed state; run 2 required no code
  changes (run 1's defect triggered the fix-and-restart path this clause prescribes).
- [x] Payments ledger contains exactly the five expected PAID rows and zero double payments.
- [x] Every deviation from the script above is explained in the table or the run sections.
- [x] README links this document; IMPLEMENTATION_PLAN status header states Phase 8 is closed.

## Run 1 results (2026-08-06)

Batch: all 20 artifacts prepared and processed, `total=20 failed_or_incomplete=0 needs_human=20`
(exit 2) - every case reached review, matching the symmetric-identity prediction above, with
review reasons matching the expected-reasons column **exactly for all 20 artifacts** (including
both new R1 policy checks: the INV-1002 Net-30 tolerance reason and the identity CONFLICT pair
1001/1016). All 20 pre-approved decisions were recorded as scripted; the ESTABLISH_MAPPING
decision wrote the `widgetarushorder → SKU-WIDGET-A` alias with full provenance and the
`recompute.after_human_mapping` event ran deterministically before the 1010 resume.

Resume outcomes: **18 of 20 cases reached their scripted terminal state on the first resume**,
including all five expected payments:

| Payment | Case (artifact) | Vendor | Amount |
|---|---|---|---|
| pay_80bb4b69... | 1001.txt | Widgets Inc. | 5000.00 USD |
| pay_7a13b6ac... | 1004_revised.json (R1) | Precision Parts Ltd. | 5940.0 USD |
| pay_d009a9a2... | 1010.txt (post-mapping) | Consolidated Materials Group | 7185.00 USD |
| pay_3bc2266e... | 1011.txt | Summit Manufacturing Co. | 3000.00 USD |
| pay_1ef14ecf... | 1015.csv | Reliable Components Inc. | 6500.00 USD |

The ledger contains exactly these five rows, zero `DUPLICATE` rows, zero double payments.
REJECT landed as scripted for 1002, 1004-original, 1005, 1007, 1008, 1011.pdf, 1012.pdf,
1013.json, 1013.pdf, 1014, 1016; HOLD landed for 1006 and 1012.txt (REQUEST_CORRECTION).

**Defect (visible, safe, and terminal for the run): 1003 and 1009 could not finish.** On resume
after the human REJECT, the approval agent re-escalated to a new review cycle (sequence 2, then
again sequence 3 after a re-affirmed REJECT) instead of submitting the forced REJECT. Root
cause: the approval agent's system prompt directed a second review whenever "a resolved human
decision exists" with non-empty `unaddressed_blocking_evidence`, without scoping that directive
to *authorizing* decisions - and 1003 (OUT_OF_STOCK) and 1009 (INVALID_QUANTITY) reproducibly
triggered the misreading on every attempt, while the 9 other reject-with-blockers cases obeyed.
The failure mode was conservative throughout: no approval, no payment, always another human
cycle - nothing was masked - but the two cases loop rather than terminate, which fails §6.3
("none required code changes to finish").

**Fix (implemented and locally verified):**

1. `decision_rules.assert_new_review_cycle_permitted`: a deterministic, fail-loud guard - a
   RESOLVED REJECT or REQUEST_CORRECTION is final, and any attempt to open another review over
   it raises `HUMAN_DECISION_MUST_BE_OBEYED` (an instructive tool error the agent recovers from
   by submitting the forced decision). Authorizing-decision second cycles (§3.5) are untouched.
2. The approval prompt's second-review directive is now explicitly scoped: "A resolved REJECT is
   final ... A resolved REQUEST_CORRECTION is final ... Only after an AUTHORIZING decision ...".
3. Unit tests covered the guard for both final kinds, all three authorizing kinds, and
   absent/pending reviews. The then-current formatting, lint, strict-type, and local-test gate was
   recorded as passing; its historical test count is intentionally omitted because it is not the
   final integrated release gate.

**§6.3 consequence:** because two artifacts required a code change to finish, the phase-restart
clause applies. Run-1 spend measured from persisted case usage: **2,042,352 prompt tokens,
53,345 completion tokens, 460 model calls** - roughly 2x the plan's §6.1 estimate (the
remediated system runs a richer critic, and symmetric batch identity routes all 20 cases through
review + resume rather than the estimated 8-12). A full restart is estimated at a similar
additional spend. The restart-versus-targeted-reverification decision is the budget owner's and
is pending; run-1 evidence (databases `artifacts/verification/workflow_phase8_run1.db`, batch
and resume logs under `artifacts/`) is retained either way.

## Run log

- Run 1 batch: 2026-08-06, exit 2 (20 × NEEDS_HUMAN, zero failures), output
  `artifacts/phase8_batch_output.txt`
- Run 1 decisions: 20 recorded (attribution per "Reviewer attribution"), then 2 re-affirmed
  REJECT cycles on 1003/1009 during defect diagnosis
- Run 1 resumes: 18 terminal as scripted, 2 looped (defect above), output
  `artifacts/phase8_resume_output.txt`
- Run 1 usage: prompt=2,042,352 completion=53,345 model_calls=460
