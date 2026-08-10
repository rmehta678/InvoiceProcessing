# ADR-001: Composite one-tool-per-specialist design, plus two granular critic tools

Status: **Accepted**, 2026-08-06. Remediation decision D1 adopted with its default: keep the
composite tools and grant the critic its own granular tools.

Related: [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md) §5.1/§6 (original granular design;
§5.1 carries a one-line amendment pointing here), [`REMEDIATION_PLAN.md`](../REMEDIATION_PLAN.md)
§2 (R0) and §6.1 (observed live costs), [module boundaries](../../Docs/REFERENCE.md#module-boundaries).

## Context

[`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md) §5.1 and §6.1-6.2 name roughly twelve
granular per-agent tools (`get_source_metadata`, the per-format readers, `render_pdf_page`,
`lookup_inventory_exact`, `lookup_item_alias`, `search_inventory_candidates`,
`aggregate_quantities`, `compute_invoice_totals`, `find_prior_invoice_candidates`, ...). The
implementation instead ships one composite deterministic tool per specialist agent in
`src/invoice_agents/agents/team.py`:

- `extract_and_record_invoice` - document evidence agent
- `compare_and_record_inventory` - inventory comparison agent
- `analyze_and_record_financial_risk` - financial and risk analyst

The granular functions all exist, but as internal Python called by the composites, not as
model-visible tools. The 2026-08-06 review classified this deviation as **documentation drift,
not a safety regression**: the deterministic tools still produce and persist all required
evidence, and no path converts a failure into an approval, so the plan's §2 principles hold. The
composite design also passed live end-to-end verification.

## Decision

1. **Keep the composite one-tool-per-specialist design.** Each specialist calls its single
   deterministic tool, which runs the full evidence pipeline for that stage and persists the
   result; the agent's job is to read, flag, and hand off - not to sequence primitive calls.
2. **Grant only the independent critic two read-only granular tools**, because the critic's value
   is checking work, not narrating it. Both live in `src/invoice_agents/agents/team.py`, write
   audit events, and are announced in the critic's prompt:
   - `recheck_inventory_item(item_name)` - wraps `InventoryReader.lookup_inventory_exact` and
     `lookup_item_alias`; returns the exact inventory row and approved-alias provenance.
   - `recompute_line(quantity, unit_price)` - wraps `recompute_line_extension`; exact `Decimal`
     multiplication, no state.

## Consequences

- Plan, README, and code agree on the tool architecture: §5.1 carries the amendment line, and the
  README module table names `agents/team.py`'s composite tools and the critic tools.
- The critic can independently re-derive inventory rows and line extensions instead of trusting
  the specialists' narration; disagreement feeds the `CRITIC_DISAGREEMENT_UNRESOLVED` decision
  rule in `src/invoice_agents/agents/decision_rules.py`.
- Fewer model round trips per case and a smaller surface for malformed tool arguments than the
  granular design.
- Specialists cannot vary their evidence-gathering sequence; any future need for selective
  per-primitive tool access is a new ADR, not a silent tool split.

## Rejected alternative: granular retrofit

Retrofitting the ~12 granular tools named in §5.1/§6 would make each specialist sequence several
model-driven tool calls where the composite makes one. The observed live baseline
([`REMEDIATION_PLAN.md`](../REMEDIATION_PLAN.md) §6.1) is ~18 model calls and 35-47k prompt tokens
per successful case; a granular retrofit projects to roughly 2-4x the model round trips - about
36-72 calls and proportionally higher prompt-token cost per case - while producing no evidence the
composites do not already persist. Every specialist prompt would also need re-tuning and
re-verification live.

| Design | Model calls per successful case | Prompt tokens per successful case | New evidence gained |
|---|---|---|---|
| Composite (shipped) | ~18 (observed live) | 35-47k (observed live) | baseline |
| Granular retrofit (~12 tools) | ~36-72 (projected, 2-4x) | proportionally higher (2-4x) | none |

The retrofit was rejected as cost without benefit; the one place granular access adds real value -
independent verification by the critic - is covered by the two read-only critic tools above.
