"""Error taxonomy, response shape, and JSON helpers shared by the LLM layer.

The error taxonomy is the important part. `ProviderUnavailable` means *this
provider cannot serve the request at all* -- no credential, no credits, a
sustained outage -- and is the only condition that should send traffic to a
different provider. `LLMError` means the request itself was bad, and failing
over would just reproduce the same fault somewhere more expensive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from pydantic import BaseModel

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.5


class LLMError(RuntimeError):
    """The model could not be reached, or could not produce valid output."""


class ProviderUnavailable(LLMError):
    """This provider cannot serve requests; another one should be tried.

    Raised for missing or rejected credentials, an unfunded account, and
    exhausted retries against an outage. Never raised for a malformed request --
    that is our bug, and failing over would only hide it.
    """

    def __init__(self, provider: str, reason: str, detail: str = "") -> None:
        self.provider = provider
        self.reason = reason
        self.detail = detail
        super().__init__(f"{provider} unavailable ({reason}): {detail}" if detail else
                         f"{provider} unavailable ({reason})")


class MissingAPIKeyError(ProviderUnavailable):
    """No credential is configured for this provider."""

    def __init__(self, provider: str, detail: str = "") -> None:
        super().__init__(provider, "no credential", detail)


@dataclass
class LLMResponse:
    """One completion, normalised across providers."""

    content: str | None
    tool_calls: list[dict[str, Any]]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None = None
    provider: str | None = None
    model: str | None = None


class LLMClient(Protocol):
    """What the agents need from a client.

    Deliberately narrow. The agents never learn which provider answered, which
    is what lets the router fail over underneath them and lets the test suite
    substitute a scripted client with no network access.
    """

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        agent: str = ...,
        tools: list[dict[str, Any]] | None = ...,
        response_schema: type[BaseModel] | None = ...,
    ) -> LLMResponse: ...

    def complete_structured(
        self,
        messages: Sequence[dict[str, Any]],
        response_schema: type[BaseModel],
        *,
        agent: str = ...,
    ) -> Any: ...


# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model into a strict-mode-compatible JSON schema.

    Both xAI and Anthropic require every property listed as required and
    ``additionalProperties: false`` on every object. Optional Pydantic fields
    already serialise as ``anyOf: [T, null]``, so marking them required costs
    nothing -- the model may still answer null.
    """

    def tighten(node: Any) -> Any:
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                properties = node.get("properties")
                if isinstance(properties, dict):
                    node["required"] = list(properties.keys())
            return {key: tighten(value) for key, value in node.items()}
        if isinstance(node, list):
            return [tighten(item) for item in node]
        return node

    return tighten(model.model_json_schema())


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response.

    Models occasionally wrap JSON in prose or a fenced code block even when
    asked not to. Rather than failing the run, decode the first balanced object
    that parses. The scan starts from *every* opening brace, not just the
    first: weaker models emit a stray leading `{` often enough to matter
    ("{\\n{\\n \\"invoice_number\\"..."), and anchoring on the first one makes the
    whole document unparseable when the real object starts two characters later.
    """
    text = text.strip()
    decoder = json.JSONDecoder()

    for candidate in (text, *(block.removeprefix("json").strip() for block in text.split("```"))):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    for start in (i for i, char in enumerate(text) if char == "{"):
        try:
            parsed, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise LLMError(f"Model response contained no parseable JSON object: {text[:300]!r}")
