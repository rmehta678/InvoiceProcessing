# Galatiq Invoice Processing Agents

A local Python prototype for auditable invoice processing. A team of eight AI agents (AutoGen
`Swarm`, powered by xAI `grok-4.5`) reads each invoice, checks it against a local inventory
database, recomputes the money, and either approves it (recording an idempotent mock payment),
rejects it, or stops and asks a human to decide. Every step is persisted for audit, and failures
are transparent: a missing key, broken database, or invalid model output can never turn into an
approval.

## Requirements

- Python 3.12 and [`uv`](https://docs.astral.sh/uv/)
- An xAI API key with access to `grok-4.5` - processing invoices makes **paid** API calls
- Network access to `https://api.x.ai/v1`; everything else runs locally

## Run it

```powershell
uv sync --python 3.12 --extra dev --extra ui
Copy-Item .env.example .env    # then set XAI_API_KEY in .env
uv run invoice-agents ui
```

The last command brings everything up: it creates, migrates, and seeds both SQLite databases
(idempotent, so it is safe on every start) and serves the web console at
<http://127.0.0.1:8787>. Use `--port` to change the port and `--no-init-db` to skip database
setup. The server is localhost-only by default and rejects remote binds without an explicit risk
acknowledgement.

Local startup uses one Uvicorn worker and a fresh random session-signing key; restarting the
console intentionally invalidates every prior browser session. Remote binding additionally
requires exact `INVOICE_UI_ALLOWED_HOSTS`, `--allow-remote-i-understand`, and an explicit random
`INVOICE_UI_SESSION_SECRET` of at least 32 bytes. Use that same secret for every process when a
deployment runs multiple workers; there is no deterministic fallback.

`data/invoices/` is a source-repository demonstration corpus; it is not installed with the
application. Installed users provide their own directory with `--invoice-dir PATH` when running
`invoice-agents batch`.

## What to expect

The console opens on a **dashboard** with a preflight strip showing database and API-key health;
run actions stay disabled until it is green.

- **Submit**: pick one or more of the 20 sample invoices under `data/invoices/` (txt, json, csv,
  xml, pdf), or upload your own. One selection runs as a single case with a live event stream;
  several - or the whole directory - run as a batch with bounded concurrency.
- Every case ends in exactly one stored status:

  | Status | Meaning |
  |---|---|
  | `SUCCEEDED` | A valid final decision exists (`APPROVE`, `REJECT`, or `HOLD`); an approval also has a valid payment. A rejection is a successful workflow outcome. |
  | `NEEDS_HUMAN` | The team stopped cleanly and queued a review package for a human decision. |
  | `FAILED` | Credentials, provider, database, source, tool, or payment failure prevented a trustworthy result. A provider timeout is exactly `FAILED / PROVIDER_TIMEOUT`. |
  | `INCOMPLETE` | Execution stopped without a trustworthy final result. Caller cancellation is exactly `INCOMPLETE / CANCELLED`; expired-lease recovery is `INCOMPLETE / ORPHANED_EXECUTION`; message-limit exhaustion retains its own stop reason. |

- **Reviews**: cases flagged by policy (large amounts, stock exceptions, unknown items, date or
  identity issues, non-USD currency, and similar) wait here. Record a decision with reviewer and
  reason, then resume the case with one click.
- Approved invoices record a **mock payment** with an idempotency key, so duplicate submissions of
  the same invoice can never pay twice. The full audit trail lives in
  `artifacts/results/CASE_ID.json` and `workflow.db`.

## Current safety contract

- A submitted file is streamed into an immutable, content-addressed snapshot before its case is
  registered. Every later extraction or PDF render verifies the stored size and SHA-256; the
  application never falls back to a changed original path. Uploads have a hard byte ceiling, and
  PDF work runs in a bounded, killable child process with explicit timeout, crash, and page-limit
  failures.
- Blocking evidence has stable IDs. An authorizing human decision applies only to the exact
  blocker IDs, mapping candidates, or superseded case identified by its review. Workflow decision
  rows and inventory aliases commit together through one attached-database transaction or both
  roll back.
- Each case execution owns a database-issued lease and fencing token. Final decisions, evidence,
  terminal results, and payment writes reject stale owners. Payment re-reads the complete
  authorization snapshot inside the same write transaction that inserts the ledger row; a paid
  case's final decision is immutable.
- Cancellation, provider failure, and abandoned-process recovery are distinct. Cancellation is
  `INCOMPLETE / CANCELLED` and is re-raised to the asyncio caller after durable recording. Provider
  timeout is `FAILED / PROVIDER_TIMEOUT`. Startup recovery alone converts an expired nonterminal
  lease to `INCOMPLETE / ORPHANED_EXECUTION`; it is not reported as a provider failure and is not
  automatically resumed.
- The local console protects every mutation with a session-bound CSRF token, same-origin checks,
  trusted-host validation, and defensive response headers. A single application-wide semaphore
  bounds all paid model work, while durable submission/source claims make double-clicks and
  restarts idempotent.
- Critique cycles are persisted and finite: an initial critique that requests follow-up requires
  exactly one linked response that addresses the requested evidence; a third cycle is refused.
  CLI operational failures use the same sanitized, bounded error boundary, and missing or empty
  input never reports a successful zero-item batch.

## CLI (optional)

The console and CLI share the same services:

```powershell
uv run python main.py --invoice_path=data/invoices/invoice_1001.txt   # original challenge command
uv run invoice-agents process --invoice-path data/invoices/invoice_1001.txt
# Deliberate new paid run only after the prior source claim is terminal:
uv run invoice-agents process --invoice-path data/invoices/invoice_1001.txt --force-reprocess
uv run invoice-agents batch --invoice-dir data/invoices --concurrency 2
uv run invoice-agents review list
uv run invoice-agents review decide REVIEW_ID --reviewer vp@example.com --decision REJECT --reason "Not authorized."
uv run invoice-agents review resume CASE_ID
```

## Tests

```powershell
uv run pytest -m "not live"    # free - makes no API calls
```

Paid live checks are explicit opt-ins: `uv run invoice-agents contract --live` checks provider
compatibility for the exact checkout at the time it is run, and
`$env:RUN_LIVE_XAI="1"; uv run pytest -m live` runs the live suite. Historical live evidence is
not current release evidence. The remediated release still requires owner approval and a fresh
paid run; until then, current xAI compatibility is **NOT REVERIFIED**.

## More documentation

- [`Docs/SYSTEM_GUIDE.md`](Docs/SYSTEM_GUIDE.md) - illustrated system guide: how a case flows,
  what each of the eight agents does and how, the decision safety net, review/resume, payment
  idempotency, storage, and the console - with mermaid diagrams for each piece
- [`Docs/REFERENCE.md`](Docs/REFERENCE.md) - operational reference: database lifecycle and review
  CLI commands, provider contract, quality gates, module boundaries, data limitations, and
  troubleshooting
- [`DEMO.md`](Docs/DEMO.md) - a short business-facing walkthrough
- [`plans/UI_PLAN.md`](plans/UI_PLAN.md), [`plans/IMPLEMENTATION_PLAN.md`](plans/IMPLEMENTATION_PLAN.md),
  [`plans/PHASE8_RECONCILIATION.md`](plans/PHASE8_RECONCILIATION.md)

The Markdown files above are authoritative for current safety and operational behavior. The
checked-in [`SYSTEM_GUIDE.pdf`](Docs/SYSTEM_GUIDE.pdf),
[`REFERENCE.pdf`](Docs/REFERENCE.pdf), and [`DEMO.pdf`](Docs/DEMO.pdf) are historical derived
snapshots; they were not regenerated by the application-audit repair and must not be used as
release evidence.
