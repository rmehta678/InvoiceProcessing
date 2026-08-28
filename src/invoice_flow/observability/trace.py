"""Structured run tracing.

Replacing a VP's email approval means being able to show, months later, exactly
why an invoice was paid. Every agent step, LLM call, and tool invocation is
appended to ``runs/<run_id>/trace.jsonl`` as it happens, so a crashed run still
leaves behind everything that led up to the crash.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator


def new_run_id() -> str:
    """A sortable, human-readable run identifier."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


@dataclass
class TokenUsage:
    """Running total of tokens consumed across a run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    # Calls served per provider, so a run that failed over says so in its stats.
    providers: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "providers": dict(self.providers),
        }


@dataclass
class Tracer:
    """Collects and persists the event log for one invoice run."""

    run_id: str
    run_dir: Path | None = None
    echo: Callable[[str, dict[str, Any]], None] | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    _fh: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._fh = (self.run_dir / "trace.jsonl").open("a", encoding="utf-8")

    # -- event recording ---------------------------------------------------

    def emit(self, event_type: str, **payload: Any) -> None:
        """Record one event. Flushed immediately so a crash keeps the trail."""
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": event_type,
            **payload,
        }
        if self._fh is not None:
            self._fh.write(json.dumps(event, default=_json_default) + "\n")
            self._fh.flush()
        if self.echo is not None:
            self.echo(event_type, event)

    @contextmanager
    def span(self, name: str, **payload: Any) -> Iterator[dict[str, Any]]:
        """Time a named stage, emitting start/end events around it.

        The context value is a mutable dict; anything put in it is merged into
        the completion event, which is how nodes attach their results.
        """
        self.emit(f"{name}.start", **payload)
        extra: dict[str, Any] = {}
        started = time.perf_counter()
        try:
            yield extra
        except Exception as exc:
            self.emit(
                f"{name}.error",
                error_type=type(exc).__name__,
                error=str(exc),
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            raise
        else:
            self.emit(
                f"{name}.end",
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                **extra,
            )

    def record_llm_call(
        self,
        agent: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        provider: str | None = None,
        **payload: Any,
    ) -> None:
        self.usage.add(prompt_tokens, completion_tokens)
        if provider:
            self.usage.providers[provider] = self.usage.providers.get(provider, 0) + 1
        self.emit(
            "llm.call",
            agent=agent,
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            **payload,
        )

    def record_tool_call(self, agent: str, tool: str, arguments: Any, result: Any) -> None:
        self.emit("tool.call", agent=agent, tool=tool, arguments=arguments, result=result)

    # -- lifecycle ---------------------------------------------------------

    def write_result(self, payload: Any, filename: str = "result.json") -> Path | None:
        """Persist the final structured result next to the trace."""
        if self.run_dir is None:
            return None
        target = self.run_dir / filename
        data = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
        target.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
        return target

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
