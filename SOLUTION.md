# Galatiq: invoice processing that actually stops the bleeding

Acme is losing **$2M/year** on a five-day, 30%-error AP process: staff retype PDFs, email a VP, then key a payment. This prototype takes an invoice file and either **pays**, **rejects with a reason**, or **pauses for a human** — in seconds, with a durable audit trail.

## What a controller should believe

| Today | This system |
|---|---|
| 5 days in email | Straight-through for clean invoices under $10k |
| 30% keying / qty errors | JSON is parsed in code; messy PDF/CSV/XML/text go to Grok. Stock is SQLite, not guessed |
| VP buried in every invoice | VP (or `--auto-approve` in demo) only sees >$10k, low-confidence, or flagged invoices |
| No record of *why* | `output/<invoice>.json` + `.log` plus a checkpointed thread id |

The $10k gate is the existing policy, encoded. Everything under it that passes inventory and a two-pass review pays without a human. That is the cycle-time win. The $2M is error + delay; we attack both by refusing to pay unknown SKUs or empty vendors, and by not waiting five days to do it.

## How the work is split (and why)

This is a **workflow**, not a chat swarm. Each box has one job so a failure is attributable.

1. **Ingest** — JSON is parsed in code. PDF, text, CSV, and XML go to Grok with a JSON schema. Sending a clean JSON invoice through the model is how you reintroduce the 30% error rate.
2. **Validate** — Inventory, quantities, vendor, amount. Code + SQLite. The model does not get a vote on whether WidgetA is in stock.
3. **Correct (once)** — If the *only* problem is an unknown item name, Grok may call `list_catalog` / `lookup_stock` and remap OCR names (`Widget-A` → `WidgetA`). True unknowns (SuperGizmo) still reject. One hop, then stop.
4. **Review → Challenge** — First reviewer recommends approve/reject/escalate, with tools if item identity is unclear. A second reviewer is paid to disagree. Concrete flags send the packet back to review **once**. Rubber-stamp “NONE” is not a loop.
5. **Human gate** — Amount > $10k, remaining flags, low confidence, or escalate. The graph **interrupts** and checkpoints. Kill the process: payment has not fired. Resume the same thread.
6. **Pay or reject** — Different terminals. A rejection is not a failed ACH.

## Technical decisions that are business decisions

- **Parsers before models.** Cheap, deterministic, testable. LLM spend is reserved for documents a clerk would squint at.
- **Tools are inventory, not theater.** `lookup_stock` and `list_catalog` are the same functions validation uses. The correction/review agents cannot invent a catalog.
- **Self-correction is bounded.** Name remap once; adversarial review once. Infinite agent ping-pong is how you miss the SLA and still pay FakeItem.
- **$10k interrupt is the VP email chain.** LangGraph checkpoint + `thread_id` is how you prove it to audit: the run paused, a human said yes, then payment ran.
- **What we cut.** No CrewAI/AutoGen, no “four agents” that are four `print`s, no regex-scraped APPROVED/REJECTED. If it does not change who gets paid, it is not in the graph.

## Run it

```bash
pip install -r requirements.txt
export XAI_API_KEY=...      

python main.py --invoice_path=data/invoices/invoice_1001.txt --auto-approve
pytest
```

Paused for a person (non-interactive, no `--auto-approve`):

```bash
python main.py --invoice_path=data/invoices/invoice_1004.json
python main.py --invoice_path=data/invoices/invoice_1004.json --resume --thread-id <id> --auto-approve
```

Straight-through pay is the happy path (INV-1001, 1004, 1006, 1011, 1014, 1015). Stock/unknown/integrity failures never call a reviewer (INV-1007, 1009). Unknown SKUs get one remap pass, then reject (INV-1008, 1016). Fraudster + FakeItem dies on inventory before anyone debates the vendor name (INV-1003).

## What “done” looks like for Acme

A clerk drops a file. Clean PO-match invoices under policy clear the same day. Exceptions show up as a one-screen question (`Pay 15225 to …?`) with the reviewer’s flags, not a 40-message email thread. Every decision is in `output/` with a thread id finance can replay. That is the MVP: stop paying the wrong SKUs, stop waiting a week to pay the right ones.
