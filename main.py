from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel

from config import CHECKPOINT_PATH, OUTPUT_DIR
from db import init_db
from graph import compile_graph
from models import Event


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
    log.info(f"Thread {thread_id}")

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
                print(f"Error: File not found: {path}")
                return 1
            log.info(f"Processing {path.name}")
            step = {"invoice_path": str(path)}

        while True:
            pause = _run(graph, step, config, log)
            if pause is None:
                break
            decision = _ask(pause, args, log)
            if decision is None:
                _write(path, thread_id, graph.get_state(config).values, outcome="needs_review", extra={"interrupt": pause})
                log.info(f"Paused for review. Resume with --resume --thread-id {thread_id}")
                return 2
            step = Command(resume=decision)

        out = _write(path, thread_id, graph.get_state(config).values)
        log.info(f"Output written to {out}")
        return 0


def _run(graph, step, config, log):
    for chunk in graph.stream(step, config, stream_mode="updates"):
        if "__interrupt__" in chunk:
            hits = chunk["__interrupt__"]
            return getattr(hits[0], "value", None) if hits else None
        for node, update in chunk.items():
            if not update:
                continue
            events = update.get("events") or []
            if events:
                for event in events:
                    name, msg = (event.node, event.message) if isinstance(event, Event) else (
                        event.get("node", node) if isinstance(event, dict) else "event",
                        event.get("message", event) if isinstance(event, dict) else str(event),
                    )
                    log.info(f"{name}: {msg}")
            else:
                log.info(f"{node} completed")
    return None


def _ask(payload, args, log):
    if args.auto_approve:
        log.info("human_gate: auto-approved")
        return "approve"
    if payload:
        log.info(f"human_gate: {payload.get('question') if isinstance(payload, dict) else payload}")
    if sys.stdin.isatty():
        answer = input("Approve payment? [approve/reject]: ").strip().lower()
        return "approve" if answer.startswith("a") else "reject"
    return None


def _write(path: Path, thread_id: str, values: dict, outcome: str | None = None, extra: dict | None = None) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    body = {
        "file": str(path),
        "thread_id": thread_id,
        "outcome": outcome or values.get("outcome"),
        "reason": values.get("reason"),
        "invoice": _json(values.get("invoice")),
        "issues": [_json(i) for i in values.get("issues") or []],
        "review": _json(values.get("review")),
        "challenge": _json(values.get("challenge")),
        "payment": _json(values.get("payment")),
    }
    if extra:
        body.update(extra)
    dest = OUTPUT_DIR / f"{path.stem}.json"
    dest.write_text(json.dumps(body, indent=2))
    return dest


def _json(value):
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _log_to(name: str) -> logging.Logger:
    OUTPUT_DIR.mkdir(exist_ok=True)
    # basicConfig is a no-op if handlers already exist (e.g. a second invoice in one process)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(OUTPUT_DIR / f"{name}.log"), logging.StreamHandler()],
    )
    return logging.getLogger("galatiq")


if __name__ == "__main__":
    sys.exit(main())
