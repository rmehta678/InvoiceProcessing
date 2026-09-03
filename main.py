from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings

warnings.filterwarnings("ignore", message=".*allowed_objects.*")

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel

from config import CHECKPOINT_PATH, OUTPUT_DIR
from db import init_db
from graph import compile_graph
from models import Event, Invoice


def main() -> int:
    parser = argparse.ArgumentParser(description="Invoice processing workflow")
    parser.add_argument("--invoice_path", required=True)
    parser.add_argument("--auto-approve", action="store_true")
    parser.add_argument("--thread-id")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    path = Path(args.invoice_path)
    log = _log_to(path.stem)
    init_db()

    thread_id = args.thread_id or f"{path.stem}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    config = {"configurable": {"thread_id": thread_id}}

    CHECKPOINT_PATH.parent.mkdir(exist_ok=True)
    with SqliteSaver.from_conn_string(str(CHECKPOINT_PATH)) as saver:
        saver.setup()
        graph = compile_graph(saver)

        if args.resume:
            decision = _ask(None, args, log)
            if decision is None:
                log.error("Resume needs --auto-approve or an interactive terminal")
                return 2
            step: dict | Command = Command(resume=decision)
        else:
            if not path.exists():
                print(f"Error: File not found: {path}", file=sys.stderr)
                return 1
            print(f"  {path.name}")
            log.info(f"Processing {path.name}")
            step = {"invoice_path": str(path)}

        trail: list[tuple[str, str]] = []
        while True:
            pause, steps = _run(graph, step, config, log)
            trail.extend(steps)
            if pause is None:
                break
            decision = _ask(pause, args, log)
            if decision is None:
                values = graph.get_state(config).values
                out = _write(path, thread_id, values, trail, outcome="needs_review", extra={"interrupt": pause})
                _print_report(path, thread_id, values, trail, "needs_review", out)
                print(f"Resume: python main.py --invoice_path={path} --resume --thread-id {thread_id}")
                return 2
            step = Command(resume=decision)

        values = graph.get_state(config).values
        out = _write(path, thread_id, values, trail)
        outcome = values.get("outcome") or "unknown"
        _print_report(path, thread_id, values, trail, outcome, out)
        return 0 if outcome == "paid" else 1


def _run(graph, step, config, log):
    steps: list[tuple[str, str]] = []
    for chunk in graph.stream(step, config, stream_mode="updates"):
        if "__interrupt__" in chunk:
            hits = chunk["__interrupt__"]
            return (getattr(hits[0], "value", None) if hits else None), steps
        for node, update in chunk.items():
            if not update:
                continue
            events = update.get("events") or []
            if not events:
                steps.append((node, "completed"))
                print(f"    {node:<12} completed", flush=True)
                log.info(f"{node}: completed")
                continue
            for event in events:
                name, msg = _event_parts(event, node)
                steps.append((name, msg))
                print(f"    {name:<12} {msg}", flush=True)
                log.info(f"{name}: {msg}")
    return None, steps


def _ask(payload, args, log):
    if args.auto_approve:
        log.info("human_gate: auto-approved")
        return "approve"
    if isinstance(payload, dict):
        print()
        print("  VP review needed")
        print(f"  {payload.get('question')}")
        review = payload.get("review") or {}
        if review.get("reason"):
            print(f"  Reviewer: {review.get('recommendation')} — {review['reason']}")
        if review.get("flags"):
            print("  Flags: " + "; ".join(review["flags"]))
        print()
    if sys.stdin.isatty():
        answer = input("Approve payment? [approve/reject]: ").strip().lower()
        return "approve" if answer.startswith("a") else "reject"
    return None


def _print_report(path: Path, thread_id: str, values: dict, trail: list, outcome: str, out: Path) -> None:
    invoice = values.get("invoice")
    label = {"paid": "PAID", "rejected": "REJECTED", "needs_review": "NEEDS REVIEW"}.get(outcome, outcome.upper())
    print()
    print(f"  {label}")
    if invoice is not None:
        print(f"  {_headline(invoice)}")
    issues = values.get("issues") or []
    reason = values.get("reason")
    if reason and not issues:
        print(f"  {reason}")
    for issue in issues:
        detail = issue.detail if hasattr(issue, "detail") else issue.get("detail")
        print(f"    - {detail}")
    print(f"  {out}")
    print()


def _headline(invoice) -> str:
    if isinstance(invoice, Invoice):
        vendor = invoice.vendor or "(no vendor)"
        amount = invoice.amount
        n = len(invoice.items)
        due = invoice.due_date or "no due date"
    else:
        vendor = invoice.get("vendor") or "(no vendor)"
        amount = invoice.get("amount")
        n = len(invoice.get("items") or [])
        due = invoice.get("due_date") or "no due date"
    return f"{vendor}  ${_money(amount)}  ·  {n} items  ·  {due}"


def _write(path: Path, thread_id: str, values: dict, trail: list, outcome: str | None = None, extra: dict | None = None) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    outcome = outcome or values.get("outcome")
    body = {
        "outcome": outcome,
        "summary": values.get("reason") or outcome,
        "file": str(path),
        "thread_id": thread_id,
        "invoice": _invoice_json(values.get("invoice")),
        "issues": [_json(i) for i in values.get("issues") or []],
        "review": _json(values.get("review")),
        "challenge": _json(values.get("challenge")),
        "payment": _json(values.get("payment")),
        "steps": [{"stage": node, "detail": msg} for node, msg in trail],
    }
    if extra:
        body.update(extra)
    dest = OUTPUT_DIR / f"{path.stem}.json"
    dest.write_text(json.dumps(body, indent=2))
    return dest


def _invoice_json(invoice):
    data = _json(invoice)
    if not data:
        return None
    data["amount"] = _money(data.get("amount"))
    return data


def _json(value):
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _money(amount) -> str:
    try:
        return f"{Decimal(str(amount)):.2f}"
    except Exception:
        return str(amount)


def _event_parts(event, fallback: str) -> tuple[str, str]:
    if isinstance(event, Event):
        return event.node, event.message
    if isinstance(event, dict):
        return str(event.get("node", fallback)), str(event.get("message", ""))
    return fallback, str(event)


def _log_to(name: str) -> logging.Logger:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(OUTPUT_DIR / f"{name}.log")],
    )
    return logging.getLogger("galatiq")


if __name__ == "__main__":
    sys.exit(main())
