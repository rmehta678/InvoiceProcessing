# Invoice Processing Pipeline

This prototype takes an invoice file and either **pays**, **rejects with a reason**, or **pauses for a human** — in seconds, with a durable audit trail. The agentic system is using LangGraph for orchestration, LLM is xAI Grok, and database is SQLite. Pytests are included for unit testing and everything else is stdlib.

## Design Decisions

1. **Ingest**: JSON is parsed in code. PDF, text, CSV, and XML go to Grok with a JSON schema.
2. **Validate**: Inventory, quantities, vendor, amount. Code + SQLite.
3. **Correct (once)**: If the only problem is an unknown item name, Grok may call `list_catalog` / `lookup_stock` and remap OCR names (`Widget-A` → `WidgetA`). True unknowns (SuperGizmo) still reject.
4. **Review → Challenge**: First reviewer recommends approve/reject/escalate, with tools if item identity is unclear. A second reviewer acts as a critic. Concrete flags send the packet back to review. 
5. **Human gate** — Amount > $10k, remaining flags, low confidence, or escalate. The graph **interrupts** and checkpoints. Kill the process: payment has not fired. Resume the same thread.
6. **Pay or reject** — Outcome of pipeline

## Technical decisions that are business decisions

- **Tools for inventory**: `lookup_stock` and `list_catalog` are the same functions validation uses. The correction/review agents cannot invent a catalog.
- **Self-correction is bounded.**: Name remap once; adversarial review once.
- **$10k interrupt is the VP email chain.**: LangGraph checkpoint + `thread_id` for audit trail: the run paused, a human said yes, then payment ran.

## Run it

```bash
pip install -r requirements.txt
export XAI_API_KEY=...      

python main.py --invoice_path=data/invoices/invoice_1001.txt --auto-approve
pytest
```

Paused for a person (non-interactive, no `--auto-approve`). Try setting APPROVAL_THRESHOLD like so to see a HITL scenario.

```bash
export APPROVAL_THRESHOLD=1
python main.py --invoice_path=data/invoices/invoice_1004.json
```

## What this MVP Accomplishes for ACME

1. Clerk provides a file
2. Invoices within policy process within minutes
3. Clear rule violations are rejected, but otherwise one screen question of approve/reject given with context
4. Every decision is output with thread for replay/audit

## Next Steps for Production
1. Eval Harness
2. VP Gate (HITL) a Robust Product: Auth/Audit Logs of Approval/SLA Timeouts/No command-line approval but via a ticketing system
3. Auth/Tenancy/PII: Encrypt at rest, retention policy
4. Model Constraints: Pin model version + schema, timeouts, retries, fallbacks to humans upon failures
