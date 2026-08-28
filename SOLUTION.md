# Invoice Processing Automation — Solution

A multi-agent pipeline that takes a supplier invoice in any of five formats and drives it
through ingestion, validation, VP approval, and payment — with the reasoning behind every
decision preserved for audit.

This is the engineering rationale: every decision, why it was made, and what it cost. For
what the system is worth to the business and how to run it, start at
[README.md](README.md). The original case brief is preserved at
[docs/CASE_BRIEF.md](docs/CASE_BRIEF.md).

---

## Quick start

Python 3.10+ (developed and tested on 3.14).

```bash
pip install -r requirements.txt
python scripts/init_db.py       # build the mock inventory database
```

That is the whole setup. Nothing is installed as a package and nothing is added to `PATH` —
`main.py` and the scripts put `src/` on `sys.path` themselves, so they run from a clone with
no editable install and no activation step.

A virtualenv is optional, not required:

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Worth knowing before you skip it: the dependencies are additive against a typical Python
install (`openai`, `langgraph` and its `langchain-core` chain, `pdfplumber`, `jinja2`,
`fpdf2`), but `langgraph` pins `langchain-core`, which in turn constrains `pydantic`. If you
already have another project sharing those, `pip install --dry-run -r requirements.txt`
will tell you whether anything would be upgraded before you commit to it.

**Prefer to click rather than type?** One page, four buttons, no extra dependency:

```bash
python scripts/serve.py         # opens http://127.0.0.1:8000
```

Each button shells out to the exact command documented below and shows its output, so the
page cannot drift away from the CLI — there is no second implementation behind it. It binds
to loopback only, and the browser sends an invoice *name* that the server looks up against
the invoice directory, so nothing a client sends reaches the command line as a path.

The fourth button runs `init_db.py --reset`, which clears the payment ledger — useful when
re-demonstrating an invoice that has already been paid. It asks for confirmation first,
since wiping the ledger makes paid invoices payable again.

**See it run without an API key** — replays scripted agent responses through the real
pipeline, so the orchestration, validation, policy engine, and reporting are all genuinely
exercised:

```bash
python scripts/demo.py                                        # all 20 sample files
python scripts/demo.py --invoice data/invoices/invoice_1003.txt --report out/report.html
```

**Run it for real** — needs an xAI credential. Put it in a `.env` file at the project root:

```bash
cp .env.example .env      # then edit XAI_API_KEY (get a key at https://console.x.ai)
python main.py --invoice_path=data/invoices/invoice_1001.txt
```

`.env` is gitignored, so the key never reaches the repository. A shell variable overrides
the file when you want a one-off:

```bash
XAI_API_KEY='xai-...' python main.py --invoice_path=data/invoices/invoice_1001.txt
```

| `.env` key | Purpose |
|---|---|
| `XAI_API_KEY` | xAI credential (`GROK_API_KEY` also accepted) |
| `INVOICE_FLOW_MODEL` | Override the Grok model; defaults to `grok-3` |
| `OPENROUTER_API_KEY` | Optional OpenRouter credential for the fallback provider |
| `OPENROUTER_MODEL` | Override the fallback model; defaults to a free tool-capable model |
| `OPENROUTER_REASONING_EFFORT` | Model-level reasoning depth; `low` by default (see below) |

A valid key still needs a funded account. If an xAI team has no credits the API returns
`403 permission-denied`, and the CLI says so explicitly rather than blaming the key — the two
failures look similar and get diagnosed very differently.

### Provider failover

xAI leads, because the case study names Grok as the reasoning engine. OpenRouter picks up
only when xAI **cannot serve at all**: no credential, a rejected credential, an unfunded
account, or a sustained outage. The CLI prints the chain at startup and reports every reason
together when nothing can serve:

```
providers: xai -> openrouter

No LLM provider could serve the request.
  - xai unavailable (account not funded): Error code: 403 ...
  - openrouter unavailable (no credential): Set OPENROUTER_API_KEY in .env to use this provider.

Fix one of the above, or run without live calls: python scripts/demo.py
```

OpenRouter is a gateway in front of many providers, so switching the fallback model is an
`OPENROUTER_MODEL` edit rather than a new backend. The pipeline needs **tool calling** and
**structured output**, and plenty of models have neither — picking one of those fails at the
third agent rather than at startup, so check a candidate's `supported_parameters` on
<https://openrouter.ai/models> before switching.

Two rules keep the behaviour sane, both in `llm/router.py`:

- **Only `ProviderUnavailable` triggers failover.** A malformed request or an unsatisfiable
  schema is *our* bug; retrying it on a second provider spends another account's credits to
  reproduce the same fault and hides it. Those raise `LLMError` and stop.
- **Failover is sticky for the run.** Once xAI has answered "no credits" it will answer that
  every time. Re-probing it on each of the dozen-odd calls in one invoice would add a doomed
  round trip to each and scatter the explanation through the log. The switch is traced once
  as `llm.provider_failover`, and per-provider call counts land in the run stats.

A third rule earns its place because free endpoints are flaky: **transient failures retry
rather than failing over.** A 429 or an overloaded upstream is the one case where retrying is
exactly right, and it is classified *before* the billing and credential buckets — otherwise a
502 body containing the word "capacity" gets diagnosed as an unfunded account.

The agents never learn which provider answered — they depend on the narrow `LLMClient`
protocol in `llm/base.py`. Both providers speak the OpenAI wire protocol, so retries, token
accounting, error classification, and the schema-repair loop all live once in
`llm/openai_compat.py`; the two backends differ only in a base URL, a credential, and two
tuning decisions.

### Two settings that came out of measurement, not preference

Both were found by running the real pipeline against a free model and watching it fail.

**Strict schema enforcement is off for OpenRouter** (`schema_mode = "object"`). Asking the
gateway to constrain decoding against the full `ExtractedInvoice` schema took **8,192 tokens
and 94 seconds**, finished with `length`, and produced unparseable truncated JSON. The same
extraction without it took **356 tokens and 2 seconds**. The schema moves into the prompt
instead; the caller validates against the Pydantic model either way, so the only thing lost
is the provider's own enforcement — which was the thing breaking. xAI implements strict mode
well and keeps it.

**Model-level reasoning is held to `low`.** An unconstrained reasoning model spends its whole
output budget thinking and gets truncated mid-JSON. The reasoning that matters here is
structural and already in the graph: the arithmetic-grounded extraction repair loop and the
VP/audit critique cycle. Paying a model to also ruminate duplicates that, slowly. Set
`OPENROUTER_REASONING_EFFORT=default` to leave it alone.

Every run writes `runs/<run_id>/` containing `trace.jsonl`, `result.json`, and
`report.html`.

| Flag | Effect |
|---|---|
| `--invoice_path=PATH` | The invoice to process |
| `--batch=DIR` | Sweep a directory and report the straight-through rate |
| `--json` | Machine-readable output only |
| `--verbose` | Show each tool call as the agents make them |
| `--report PATH` | Write the HTML report somewhere specific |
| `--model NAME` | Override the Grok model |

Exit codes: `0` approved, `1` rejected or escalated, `2` error. A batch exits `0` only when
every invoice was approved, so it drops straight into a CI check.

`--batch` runs the whole directory against **one** ledger, which is what makes the duplicate
controls meaningful: isolating each invoice would hide the control the stage exists to run.

```
INVOICE                      DECISION    PAYMENT           TOTAL  FINDINGS
--------------------------------------------------------------------------
invoice_1001.txt             APPROVED   success       $5,000.00
invoice_1016.json            REJECTED   skipped       $3,233.00  1 crit

1 approved  1 rejected  0 escalated  (50% straight-through)
$3,233.00 of payments held back for review or rejection.
```

---

## What the pipeline does

```
load ─► extract ◄──────┐            ingestion agent
          │            │
     verify_extraction ─┘  bounded repair cycle
          │
      validate                      validation agent (tool-calling)
          │
     vp_decide ◄───────┐            approval agent
          │            │
    audit_critique ────┘  bounded reflection cycle
          │
    finalise_approval
          │
       settle                       payment agent
```

Both self-correction loops are real cycles in the LangGraph topology, not `for` statements
inside a node. Each iteration is a separate, individually traced transition, and the loop
bounds are visible in the graph rather than buried in an agent.

### 1. Ingestion

Five formats converge on a single text representation that one extractor reads. The
alternative — a bespoke structural parser per format — looks tidier and fails exactly where
this dataset is hardest: `invoice_1006.csv` is a `field,value` sheet with the key `item`
repeated three times, and `csv.DictReader` silently keeps only the last one.

The extractor is instructed to be a **faithful transcriber, not an editor**. It copies dates
verbatim (`26-Jan-2O26`, `yesterday`), preserves item names exactly (`Widget A`), and never
recomputes a total that does not add up. Silently correcting bad source data destroys the
evidence the downstream checks depend on.

**The repair loop is grounded in arithmetic rather than a second model opinion.** Two models
critiquing each other can disagree indefinitely; a subtotal cannot be argued with. After each
pass the draft is reconciled against itself and any inconsistency goes back as a specific,
checkable complaint. Two guards stop the loop spinning on invoices that are genuinely wrong:
the repair prompt explicitly tells the model to re-transcribe unchanged figures when the
invoice itself is defective, and identical consecutive drafts end the loop early.

### 2. Validation

Deterministic checks produce the authoritative findings; the LLM agent queries the same
database through `lookup_item` / `check_stock` / `list_catalog` and writes the summary a
clerk reads. **The model cannot overturn a deterministic finding** — that boundary is what
stops a fluent-but-wrong summary from authorising a payment.

Three details the data forces:

- **Quantities aggregate per product before the stock check.** `invoice_1013` bills WidgetA
  across three lines totalling 22 against stock of 15. Checking each line alone (15, 5, 2)
  passes all three and approves an unfulfillable order.
- **Item names normalise but never auto-substitute.** Casefolding and stripping
  non-alphanumerics resolves the OCR-spaced `Widget A` to `WidgetA`. `WidgetC` scores ~0.86
  similarity against `WidgetA` and is reported as a near-miss for a human — auto-correcting
  it would authorise payment for a product Acme does not stock.
- **Date sanity anchors on the invoice date, not today.** The sample set is historical;
  anchoring on the current date would flag all 18 invoices as overdue and teach the approval
  agent nothing.
- **A missing total is derived, not refused — and the derivation is flagged.** The reconstructed
  figure always agrees with itself, so the cross-check that catches a vendor's arithmetic error
  is gone for that invoice, and the report says so. Reconstruction runs only *after* the repair
  loop settles: filling the total mid-loop would leave the arithmetic check comparing a derived
  number against itself, masking exactly the discrepancies the loop exists to find. Approval is
  blocked only when nothing can be derived at all — no total and no priced line items.
- **Tax is per-invoice, never computed.** The data settles it: 1001 states 0%, 1002 states no
  tax line, and 1010 states a $335 amount with no rate. Any "apply the standard rate" logic
  would bill something the vendor did not. Tax is verified only when both rate and amount are
  present, and absent tax is treated as zero rather than invented.

### 3. Approval

Three components in order of authority:

1. **The policy engine sets a ceiling.** With a critical finding outstanding, `APPROVED` is
   not discouraged in a prompt — it is *removed from the set of permitted outcomes*. This is
   the difference between a system that cannot pay a fraudulent invoice and one that merely
   tends not to. Three rules remove it: any hard block, any escalation trigger, and **two or
   more fraud signals**. Any one signal is explicable — a rushed vendor writes "URGENT", a
   small supplier prefers wire — but urgency, wire-transfer preference, an unparseable due
   date, and an amount parked just under the threshold are, in combination, the shape payment
   fraud takes. Two forces ESCALATED: a human's five minutes, not a refusal.
2. **The VP agent** reasons within that ceiling.
3. **The critic agent** challenges the draft; an objection buys the VP one revision.

Both rounds, and any policy override, are preserved in the report. An auditor needs to see
that the guard rail fired, not just its result.

**Three outcomes, not two.** `ESCALATED` exists because a binary system has nowhere to put
`invoice_1014`, which is denominated in EUR — neither paying it at face value nor rejecting
it is correct. Escalation is also the safe default when policy blocks approval but the cause
looks clerical: a wrong rejection costs a supplier relationship, a wrong approval costs
money, a wrong escalation costs somebody five minutes.

### 4. Payment

Every outcome writes to the ledger, not just the payments — a rejection nobody recorded gets
re-submitted next week and paid on the second attempt.

**Duplicates split into three different problems that look alike.** Each ledger entry stores a
fingerprint hashed over the payable facts (vendor, aggregated quantities, total) and nothing
else, so format is invisible to it:

- **Same number, same fingerprint** — 1011, 1012 and 1013 each ship in two formats carrying
  identical totals and quantities. That is one document seen twice. Noted as INFO; the ledger
  still blocks the second payment.
- **Same number, different fingerprint, earlier version *paid*** — 1004 is paid at $1,890,
  then 1004_revised arrives at $5,940. This **escalates**, and the reason is arithmetic
  rather than caution: approving it pays the vendor $7,830 for a $5,940 invoice, because the
  first $1,890 has already left. The finding hands the human the number they actually need —
  *the amount outstanding is $4,050.00 — settle the difference or void the first payment*.
  The system cannot reverse a payment it already made, so choosing between those is not its
  call.
- **Same number, different fingerprint, earlier version *not paid*** — nothing was disbursed,
  so a corrected submission is just a better invoice. Noted as a WARNING and judged on its own
  merits. Escalating here would punish a vendor for fixing the very problem that got their
  first attempt rejected.

That last distinction is the whole point of keying on `payment_status` rather than on the
fingerprint alone. "Which version is real" is only an emergency once money has moved.

**A revision marker is evidence, not authority.** 1004_revised declares `"revision": "R1"` in
the document itself, and the extractor now captures it so it appears in the finding a human
reads. It never decides anything: vendor-supplied text claiming to supersede a paid invoice
is the exact shape of the fraud this control exists to catch, and `INV-1004 rev R2` for
$50,000 would carry the same marker. Same restraint as the near-miss item names — surfaced
for a human, never acted on automatically.

This check runs at *validation*, not at payment. Blocking a second payment after a VP has
already approved it is a backstop; catching it before the decision is a control.

---

## What the sample data actually contains

Reading all 18 invoices turned up several traps beyond those the brief lists. Each is a
golden test case in `tests/test_pipeline.py`.

| Invoice | Trap | Outcome |
|---|---|---|
| 1001 | clean baseline | APPROVED |
| 1002 | header typos (`INVOCE`, `Vndr`); GadgetX 20 vs stock 5; due date == invoice date | REJECTED |
| 1003 | FakeItem 100 vs stock 0; due date `"yesterday"`; $100K; urgency + wire-transfer language | REJECTED |
| 1004 / 1004_revised | **same invoice number**, $1,890 vs $5,940 | first paid; second ESCALATED as a conflict |
| 1005 | GadgetX 8 vs 5 | REJECTED |
| 1006 | `field,value` CSV with repeated keys | APPROVED |
| 1007 | stock shortfalls **and a $110 arithmetic error** (subtotal + 6% tax = $15,635, states $15,525) | REJECTED |
| 1008 | email body; unknown items; $9,900 — just under the threshold | REJECTED |
| 1009 | empty vendor, `null` due date, qty `-5`, total `-$250` | REJECTED |
| 1010 | WidgetA billed twice (8 + 4 rush) — must aggregate | APPROVED |
| 1011 | ships as **both** PDF and TXT | paid once |
| 1012 | OCR artifacts: `2O26`, `$3,500.O0`, `Widget A`; $9,975 just under the threshold | APPROVED + flagged |
| 1013 | 8 lines aggregating to 22/18/9 against 15/10/5; **states a total $50 above its own subtotal + tax** | REJECTED |
| 1014 | **EUR**, not USD | ESCALATED |
| 1015 | clean | APPROVED |
| 1016 | WidgetC not in catalogue | REJECTED |

Two findings not mentioned in the brief: **invoice 1007 carries a $110 internal arithmetic
error**, and **invoice 1013's $50 discrepancy is inside each file** (its stated total exceeds
its own subtotal plus tax) rather than being a PDF-vs-JSON difference — both files agree with
each other and both are internally inconsistent.

---

## Testing

```bash
python -m pytest              # 139 tests, no API key required
```

`python -m pytest` rather than a bare `pytest` on purpose: the bare command only exists on
`PATH` once a virtualenv is activated, so it fails with `command not found` for anyone who
skipped the optional venv above. Running it as a module works either way, and from any
directory.

The scripted VP in `tests/fake_llm.py` **tries to approve everything, and its critic always
agrees**. Every correct outcome in the golden suite is therefore produced by the policy
engine and the deterministic checks rather than by a cooperative model. A test whose scripted
VP already knows the right answer proves nothing; this one demonstrates that the system's
safety rests on code under our control. `test_scripted_vp_approves_everything` guards that
property directly.

Also covered: bounded reflection loops, LLM-outage fallback to rule-based decisions,
duplicate-payment blocking, HTML autoescaping of hostile vendor text, and graceful handling
of unsupported files.

**The suite's one blind spot, and what guards it.** Extractions are replayed from a fixture,
so the goldens prove the pipeline is correct *given a stable extraction* — and an LLM does
not give you one. That gap produced a real bug: most invoices show `$` and no ISO code, so
the extractor returned `"USD"` on one pass and `null` on the next, both faithful to the
document. The payable fingerprint hashed them differently, and a second read of a clean
INV-1001 was escalated as a conflicting revision. An absent currency is now read as the base
currency, and `test_fingerprint_survives_extraction_variance_on_optional_fields` asserts that
optional-field presence never changes a fingerprint while a genuine currency difference still
does. The general lesson holds beyond this one field: anything hashed as a canonical fact has
to be canonical first.

---

## Layout

```
main.py                       CLI entry point
.env.example                  template for the local credential file
scripts/init_db.py            build + seed the inventory database
scripts/demo.py               run the pipeline with scripted responses (no key needed)
scripts/serve.py              one-page browser UI; shells out to the commands above
src/invoice_flow/
  config.py                   thresholds, policy constants, and .env loading
  models.py                   Pydantic contracts shared by every agent
  graph.py                    LangGraph topology, including both cycles
  llm/base.py                 LLMClient protocol, error taxonomy, JSON extraction
  llm/openai_compat.py        the client: transport, retries, tracing, schema repair
  llm/grok.py                 xAI backend (primary)
  llm/openrouter.py           OpenRouter backend (fallback)
  llm/router.py               failover policy across the two backends
  llm/prompts.py              every system prompt, reviewable as policy text
  agents/                     ingestion, validation, approval, payment
  tools/                      loaders, inventory, arithmetic, dates, risk, payment
  reporting/                  Rich console and self-contained HTML report
  observability/trace.py      JSONL event log, flushed per event
tests/                        loaders, pipeline goldens, reporting
```

---

## Notes on the brief

- **The xAI snippet in the brief (`from xai import Grok`) is not a published package.** This
  uses the OpenAI-compatible endpoint at `https://api.x.ai/v1` via the `openai` SDK, which
  gives tool calling and schema-constrained output over a conventional, well-tested client.
  The official `xai-sdk` is the alternative.
- **The brief asks for Grok specifically, and Grok is what runs.** OpenRouter is a fallback,
  not a substitute: it engages only when xAI cannot serve, and the CLI and trace say plainly
  which provider answered. Leave `OPENROUTER_API_KEY` unset to run xAI-only.
- **The brief's seed snippet uses a plain `INSERT`,** which raises on the second run against
  the PRIMARY KEY. `scripts/init_db.py` uses `INSERT OR IGNORE` and is safe to re-run.
- **"No internet for external APIs"** is read as: the inventory database and the payment rail
  are simulated locally, which they are. The LLM is the reasoning engine the brief requires,
  so it does make network calls; `python scripts/demo.py` runs the whole pipeline with
  scripted responses for anything that must not.
- **`.env` is parsed in `config.py` rather than via `python-dotenv`.** Holding one API key
  needs about twenty lines, and doing it in-repo means the credential is picked up whether or
  not an extra package happens to be installed. Shell variables take precedence over the
  file, matching the usual dotenv convention.

## Future add-ons

- FX conversion for non-USD invoices (currently flagged and escalated, not converted), and
  unit-price variance against the catalogue.
- Touch-rate metrics beyond the straight-through rate `--batch` already reports — time per
  invoice and human-minutes saved need a baseline measurement this prototype has no source for.
- A batch run builds one client per invoice, so a provider that is down gets re-probed once
  per invoice instead of once per run. Harmless when the primary is healthy; wasteful in a
  degraded batch.
