# Invoice Processing Automation

A four-agent pipeline that takes a supplier invoice — PDF, CSV, JSON, XML or plain text —
and carries it through ingestion, validation, VP approval, and payment, keeping the reasoning
behind every decision for audit.

Built with LangGraph and xAI's Grok against the [Galatiq case brief](docs/CASE_BRIEF.md).
The engineering rationale lives in **[SOLUTION.md](SOLUTION.md)**.

---

## What this is worth

The brief puts Acme's manual process at **$2M a year**, a **30% error rate**, and
**five-day delays**. Run against the 20 sample invoices, here is what the system does about
each of those:

| Acme's problem | The decision that addresses it | Evidence from the sample run |
|---|---|---|
| **30% error rate** | Deterministic checks decide; the model only explains. Whether 22 units exceed a stock of 15 is arithmetic, and an LLM adds nothing but variance. | **11 of 20 invoices stopped**, including a **$110** arithmetic error the vendor introduced on INV-1007 and a **$50** discrepancy inside INV-1013 |
| **5-day delays** | 45% go straight through with no human involvement at all | The other 55% arrive at a human with a decision, a rationale, and the findings already attached — not a blank page and a PDF |
| **$2M leakage** | A payment ledger and fraud-pattern rules run *before* the approval decision, not after | **$4,050** double payment blocked on INV-1004; a **$100,000** invoice combining urgency language, wire-transfer preference and a zero-stock item rejected outright |
| **Trusting it** | The policy engine *removes* APPROVED from the set of permitted outcomes — it is not discouraged in a prompt, it is unavailable | The test suite's VP tries to approve **everything** and its critic always agrees; every correct outcome still comes out correct, because the safety lives in code |

**$213,823.60 of payments held back** for review or rejection across the sample set.

Two things it deliberately will not do: convert currencies, or decide which of two conflicting
versions of an invoice is the real one. Both escalate to a human. A wrong rejection costs a
supplier relationship, a wrong approval costs money, and a wrong escalation costs somebody
five minutes — so where the system cannot be sure, it buys the five minutes.

---

## Run it

Python 3.10+. No installation, no editable package, no activation step.

```bash
pip install -r requirements.txt
python scripts/init_db.py       # build the mock inventory database
```

**The quickest look — a browser page, no API key needed:**

```bash
python scripts/serve.py         # opens http://127.0.0.1:8000
```

Four buttons: process one invoice live, sweep all 20 with scripted agent responses, build an
HTML audit report, or reset the database. Each one shells out to the command below and shows
its output, so nothing on the page is a second implementation.

**Or from the terminal:**

```bash
python scripts/demo.py                          # all 20 files, no API key required
python main.py --invoice_path=data/invoices/invoice_1001.txt
python main.py --batch=data/invoices/           # straight-through rate over the whole set
python -m pytest                                # 139 tests, no API key required
```

Live runs need an xAI key in `.env` (`cp .env.example .env`). Everything else — the inventory
database, the payment rail — is simulated locally.

```
INVOICE                      DECISION    PAYMENT           TOTAL  FINDINGS
--------------------------------------------------------------------------
invoice_1003.txt             REJECTED   skipped     $100,000.00  1 crit 4 warn
invoice_1004.json            APPROVED   success       $1,890.00
invoice_1004_revised.json    ESCALATED  skipped       $5,940.00  1 crit

1 approved  1 rejected  1 escalated  (33% straight-through)
$105,940.00 of payments held back for review or rejection.
```

---

## How it works

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

Both self-correction loops are real cycles in the LangGraph topology rather than `for`
statements inside a node, so every iteration is an individually traced transition and the
loop bounds are visible in the wiring.

Three decisions carry most of the weight:

- **The extraction repair loop is grounded in arithmetic, not a second model opinion.** Two
  models critiquing each other can disagree forever; a subtotal cannot be argued with.
- **Deterministic checks are authoritative; the LLM interprets them.** The model writes the
  summary a clerk reads and can never overturn a finding.
- **Approval is a three-way outcome.** APPROVED, REJECTED, and ESCALATED — because an invoice
  denominated in EUR is neither clean nor clearly wrong.

Every run writes `runs/<run_id>/` with a JSONL trace, the structured result, and a
self-contained HTML report.

---

## Where things are

| | |
|---|---|
| **[SOLUTION.md](SOLUTION.md)** | Every engineering decision, its reasoning, and its cost |
| **[docs/CASE_BRIEF.md](docs/CASE_BRIEF.md)** | The original Galatiq brief, unaltered |
| `main.py` | CLI entry point — single invoice or `--batch` |
| `scripts/serve.py` | The browser UI |
| `scripts/demo.py` | Scripted run, no API key |
| `src/invoice_flow/agents/` | ingestion, validation, approval, payment |
| `src/invoice_flow/graph.py` | The LangGraph topology, including both cycles |
| `src/invoice_flow/llm/` | Grok client, OpenRouter fallback, failover policy, prompts |
| `tests/` | 139 tests, no API key required |
