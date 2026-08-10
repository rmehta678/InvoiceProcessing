# Technical reference

The operational reference: exact commands, provider contract checks, framework facts, module
boundaries, data limitations, and troubleshooting. How to run the system is covered in the
[README](../README.md). How it works - the architecture, the eight agents, the decision safety
net, review and resume, payment idempotency, storage, and the web console - is covered with
diagrams in [`SYSTEM_GUIDE.md`](SYSTEM_GUIDE.md); [`DEMO.md`](DEMO.md) is the short business
walkthrough.

## Database lifecycle

`invoice-agents ui` migrates, seeds, and verifies both databases on start (idempotently). The same
steps can be run explicitly - normal case processing never creates or repairs a database:

```powershell
uv run python -m invoice_agents.db migrate --db inventory.db
uv run python -m invoice_agents.db seed --db inventory.db
uv run python -m invoice_agents.db verify --db inventory.db

uv run python -m invoice_agents.db migrate --db workflow.db --kind workflow
uv run python -m invoice_agents.db verify --db workflow.db --kind workflow
```

Database migrations are runtime package resources, so these commands work from an installed wheel
as well as from a source checkout. The source repository's `data/invoices/` directory is only a
demonstration corpus; installed batch users must pass their own `--invoice-dir PATH`.

Both databases must use SQLite `DELETE` journal mode. This is a fixed safety contract for atomic
human-decision writes across the workflow and inventory files, exposed as
`INVOICE_SQLITE_JOURNAL_MODE=DELETE` only for explicit validation. `PERSIST`, `TRUNCATE`, and
`WAL` are rejected rather than converted, and retained `-journal`, `-wal`, or `-shm` sidecars
must be recovered or removed by an operator before verification or migration can proceed.

Verification pins the inventory seed verbatim; it contains only facts from the challenge README:

| SKU | Item | Available stock |
|---|---|---:|
| `SKU-WIDGET-A` | `WidgetA` | 15 |
| `SKU-WIDGET-B` | `WidgetB` | 10 |
| `SKU-GADGET-X` | `GadgetX` | 5 |
| `SKU-FAKE-ITEM` | `FakeItem` | 0 |

Alias rows start empty. Fuzzy candidates are suggestions only; an exact item name, persisted
approved alias, or recorded human mapping is required to establish a canonical SKU. The schema and
storage model are described in [SYSTEM_GUIDE §7](SYSTEM_GUIDE.md#7-persistence-and-audit).

## Human review commands

What routes a case to review, the five decisions, and their exact consequences are covered in
[SYSTEM_GUIDE §5](SYSTEM_GUIDE.md#5-human-in-the-loop-review); the policy knobs
(`INVOICE_REVIEW_THRESHOLD_*`, `INVOICE_DUE_DATE_TOLERANCE_DAYS`) are listed in
[SYSTEM_GUIDE §9](SYSTEM_GUIDE.md#9-configuration). Those values are business policy; changing
them requires business-owner review.

```powershell
uv run invoice-agents review list
uv run invoice-agents review show REVIEW_ID

uv run invoice-agents review decide REVIEW_ID `
  --reviewer vp@example.com `
  --decision REJECT `
  --reason "Requested quantity is not authorized."

uv run invoice-agents review resume CASE_ID
```

`--decision` accepts `APPROVE`, `REJECT`, `REQUEST_CORRECTION`, `ESTABLISH_MAPPING`, and
`SUPERSEDE_REVISION`. Mapping decisions require one or more `--mapping "raw item=SKU"` arguments;
revision decisions require `--superseded-case-id`. Reviewer and reason are mandatory. Review
packages for PDF sources include rendered page images under `artifacts/reviews/{review_id}/`.

## Web console operations

The console's pages and design rules are covered in
[SYSTEM_GUIDE §8](SYSTEM_GUIDE.md#8-the-web-console), with the page-by-page plan in
[`UI_PLAN.md`](../plans/UI_PLAN.md). Operational facts: the server binds to localhost only and
there is no auth layer - the CLI refuses non-loopback hosts without
`--allow-remote-i-understand`. The key value is never rendered, only whether it is present. All
assets ship in-repo (no CDN, no Node toolchain). Opt-in browser smokes exist behind
`RUN_UI_SMOKE=1` (`uv run playwright install chromium` first).

## Provider compatibility contract

The paid live suite proves authentication, the exact model and endpoint, strict typed tools,
sequential tool iterations, Pydantic structured output after tool evidence, Swarm handoff,
agent-name configuration, stopped-state save/load/human resume, and tool-exception visibility:

```powershell
uv run invoice-agents contract --live
```

Without `--live`, the CLI reports `NOT RUN` and exits nonzero; a skipped contract is never
described as a pass. The verified xAI-safe message-name configuration is fixed at
`include_name_in_message=False` and `add_name_prefixes=True`, not selected as a fallback.

## Tests and quality checks

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -m "not live"

# Explicitly opt into paid xAI tests:
$env:RUN_LIVE_XAI="1"
uv run pytest -m live
```

What each suite locks down, per area, is tabulated in
[SYSTEM_GUIDE §11](SYSTEM_GUIDE.md#11-what-the-tests-lock-down).

### Verification artifacts

`artifacts/verification/` holds the two `workflow_e2e*.db` databases retained from the 2026-08-06
live end-to-end verification of `INV-1001` (clean approval and payment) and `INV-1002` (policy
review path), plus archived local workflow databases from completed run phases. The directory is
gitignored local evidence.

The Phase 8 full-corpus live reconciliation - all 20 artifacts processed end to end on the
remediated system, with the expected-decision script, per-artifact outcomes, exactly five
payments, and both run records - is documented in
[`PHASE8_RECONCILIATION.md`](../plans/PHASE8_RECONCILIATION.md).

## Framework notes

AutoGen 0.7.5 is in maintenance mode and identifies Microsoft Agent Framework as its successor.
This prototype remains on AutoGen because that is the selected framework; it does not change
frameworks at runtime. The team is a `Swarm` with model-driven handoff tools;
`SelectorGroupChat` is intentionally not used.

## Module boundaries

| Area | Responsibility |
|---|---|
| `src/invoice_agents/models.py` | Strict Pydantic contracts and explicit statuses |
| `tools/evidence.py` | File hash/metadata, format readers, PDF text/render evidence, provenance-preserving extraction |
| `tools/comparison.py` | Read-only SQL, explicit aliases, aggregation, Decimal totals, dates, identity, policy evidence |
| `agents/team.py` | Narrow prompts, one composite deterministic tool per specialist plus two read-only critic recheck tools ([ADR-001](../plans/adr/001-composite-agent-tools.md)), handoffs, critic loop, approval and payment agent construction |
| `agents/decision_rules.py` | Pure final-decision rules; `APPROVE` requires the critic's agreement or a resolved authorizing human decision |
| `orchestration.py` | Preflight, fresh-team streaming, circuit breakers, state save/resume, final status mapping |
| `hitl/` | Complete review packages and attributable human decisions |
| `payment/` | Approval checks, mock transaction ledger, idempotency, injected failure |
| `db/` and `migrations/` | Explicit versioned database lifecycle and workflow persistence |
| `observability/` | Redacted correlated events and local OpenTelemetry spans |

## Evidence and data limitations

The inventory database authoritatively supports only item existence and available stock. Vendor,
purchase order, price, tax, currency/FX, bank-account, and fraud-master reconciliation are
explicitly reported as unavailable; agents must not imply those checks occurred. Inventory is an
independent per-invoice validation reference - processing does not reserve or decrement stock, so
a production batch could overcommit inventory; reservation and shared-commit semantics must be
designed before enabling real payment.

PDF extraction uses `pypdf`; `render_pdf_page` uses PyMuPDF when layout evidence is needed. A
successful text extraction is not treated as visual proof. xAI vision is not required by the
current workflow, so invoice page images are not sent to the provider.

Invoice evidence, agent messages, and results are sensitive local data. `.env`, databases,
rendered pages, results, and other runtime artifacts are ignored by Git. Logs recursively redact
credential keys and bearer values. AutoGen streamed events are persisted locally; ordinary console
logging does not duplicate full prompts. SDK retry attempts are correlated to the active case.
AutoGen's pinned OpenAI-compatible client does not expose xAI's Zero Data Retention response
header, so each run records that status as `NOT_EXPOSED` rather than making a ZDR claim. xAI
remains the one remote dependency, so business owners must approve its retention terms, serving
regions, and any Zero Data Retention requirement before production use.

## Troubleshooting

The full failure taxonomy is in [SYSTEM_GUIDE §10](SYSTEM_GUIDE.md#10-failure-taxonomy); common
stop reasons and their fixes:

- `DATABASE_MISSING` / `DATABASE_SCHEMA_MISMATCH`: restart `invoice-agents ui` (which migrates and
  seeds), or run the explicit migrate, seed, and verify commands; case processing will not repair
  the database.
- `PROVIDER_AUTHENTICATION_FAILED`: verify `XAI_API_KEY`; there is no alternate model.
- `PROVIDER_RATE_LIMIT_EXHAUSTED` / `PROVIDER_TIMEOUT`: configured bounded retries were exhausted;
  the case remains failed/incomplete and keeps the provider request ID when available.
- `NEEDS_HUMAN`: inspect the review package, record a reasoned decision, then resume the case.
- `MAX_MESSAGES_EXHAUSTED`: inspect persisted events; the latest prose is not accepted as a result.
- `PAYMENT_FAILED`: inspect the ledger/event error; no success event is synthesized.
