"""A scripted stand-in for the Grok client, for deterministic tests.

The VP it scripts is deliberately naive: it tries to APPROVE everything, and
the critic always agrees with it. That is the point. If the golden outcomes
still come out correct, the safety of the system demonstrably rests on the
policy engine in `agents/approval.py` -- which is code we control -- and not on
the model happening to reason well on the day. A test whose scripted VP already
knows the right answer proves nothing.

Extractions are replayed from `fixtures/golden_extractions.json`, so the
pipeline below ingestion is exercised on exact, known-good input.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel

from invoice_flow.llm.base import LLMResponse
from invoice_flow.models import ApprovalDraft, Critique, Decision, ExtractedInvoice

FIXTURES = Path(__file__).parent / "fixtures"


def load_golden_extractions() -> dict[str, dict[str, Any]]:
    data = json.loads((FIXTURES / "golden_extractions.json").read_text(encoding="utf-8"))
    return {key: value for key, value in data.items() if not key.startswith("_")}


class ScriptedGrokClient:
    """Duck-typed replacement for `GrokClient` with no network access."""

    def __init__(
        self,
        *,
        critic_objects: bool = False,
        critic_suggestion: Decision | None = None,
        vp_decision: Decision = Decision.APPROVED,
        fail_on: set[str] | None = None,
    ) -> None:
        self.extractions = load_golden_extractions()
        self.critic_objects = critic_objects
        self.critic_suggestion = critic_suggestion
        self.vp_decision = vp_decision
        self.fail_on = fail_on or set()
        self.calls: list[str] = []
        self.tool_rounds: dict[str, int] = {}

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _source_file(messages: Sequence[dict[str, Any]]) -> str | None:
        for message in messages:
            content = message.get("content") or ""
            if isinstance(content, str):
                match = re.search(r"Source file: (\S+)", content)
                if match:
                    return match.group(1)
        return None

    def _maybe_fail(self, agent: str) -> None:
        from invoice_flow.llm.base import LLMError

        for token in self.fail_on:
            if agent.startswith(token):
                raise LLMError(f"Scripted failure for agent '{agent}'")

    # -- API surface -------------------------------------------------------

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        agent: str = "unknown",
        tools: list[dict[str, Any]] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        self.calls.append(agent)
        self._maybe_fail(agent)

        # Exercise the tool-dispatch path once, then summarise.
        if tools and self.tool_rounds.get(agent.split(".")[0], 0) == 0:
            self.tool_rounds[agent.split(".")[0]] = 1
            return LLMResponse(
                content=None,
                tool_calls=[{"id": "call_1", "name": "list_catalog", "arguments": "{}"}],
                prompt_tokens=100,
                completion_tokens=10,
            )

        return LLMResponse(
            content="Validation complete. See structured findings for detail.",
            tool_calls=[],
            prompt_tokens=120,
            completion_tokens=20,
        )

    def complete_structured(
        self,
        messages: Sequence[dict[str, Any]],
        response_schema: type[BaseModel],
        *,
        agent: str = "unknown",
    ) -> Any:
        self.calls.append(agent)
        self._maybe_fail(agent)

        if response_schema is ExtractedInvoice:
            source = self._source_file(messages)
            if source is None or source not in self.extractions:
                raise AssertionError(f"No golden extraction recorded for source file {source!r}")
            return ExtractedInvoice.model_validate(self.extractions[source])

        if response_schema is ApprovalDraft:
            return ApprovalDraft(
                decision=self.vp_decision,
                rationale="Scripted VP rationale: proceeding with payment.",
                key_factors=["scripted"],
                conditions=[],
            )

        if response_schema is Critique:
            if self.critic_objects:
                return Critique(
                    agrees=False,
                    objections=["Scripted objection: the decision ignores outstanding findings."],
                    suggested_decision=self.critic_suggestion or Decision.ESCALATED,
                    reasoning="Scripted audit reasoning.",
                )
            return Critique(agrees=True, objections=[], reasoning="Scripted audit concurrence.")

        raise AssertionError(f"ScriptedGrokClient has no script for {response_schema.__name__}")
