"""The LLM client: chat completions over the OpenAI wire protocol.

Both xAI and OpenRouter speak that protocol, so they share everything except a
base URL, a credential, and the vocabulary each uses to say "no". Subclasses
supply those; the retry policy, schema degradation, tracing, and the
schema-repair loop live here once.

Error classification is the part worth getting right. `ProviderUnavailable`
means *this provider cannot serve at all* and is the only signal that should
move traffic elsewhere. A malformed request is our bug and must not fail over --
a second provider would reject it identically, having spent another credential
to do so.
"""

from __future__ import annotations

import json
import time
from typing import Any, Sequence, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import TEMPERATURE, Settings
from .base import (
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    LLMError,
    LLMResponse,
    MissingAPIKeyError,
    ProviderUnavailable,
    extract_json_object,
    strict_json_schema,
)

T = TypeVar("T", bound=BaseModel)

# Substrings meaning the account cannot be billed. Kept separate from credential
# markers because a valid key on an unfunded account needs different advice --
# telling someone to check a key that is actually fine wastes their time.
BILLING_MARKERS = (
    "credits",
    "license",
    "permission-denied",
    "quota",
    "billing",
    "insufficient",
    "402",
    "payment required",
)

# Substrings meaning the credential itself is wrong. xAI reports an unusable key
# as `400 invalid-argument: Incorrect API key`, not the 401 you would expect.
CREDENTIAL_MARKERS = (
    "401",
    "403",
    "incorrect api key",
    "invalid api key",
    "no auth credentials",
    "unauthorized",
)

# Substrings meaning the model name is not served. Not a provider outage -- the
# configuration is wrong, and failing over would hide a typo in a model id.
MODEL_MARKERS = ("404", "not found", "no endpoints found", "no allowed providers")

# Reasoning models spend output tokens thinking before they answer. Without an
# explicit ceiling a gateway's modest default can be consumed entirely by the
# reasoning trace, leaving `content` empty and the turn apparently blank.
DEFAULT_MAX_TOKENS = 8192

# Substrings meaning "try again shortly". Free and shared endpoints hit these
# constantly, and they are the one class of failure where retrying is exactly
# right -- neither failing over nor giving up.
TRANSIENT_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "overloaded",
    "rate limit",
    "rate-limit",
    "timeout",
    "timed out",
    "temporarily",
    "try again",
    "capacity",
)


class EmptyResponse(LLMError):
    """The provider returned no choices -- often a transient upstream error."""


class OpenAICompatibleClient:
    """Chat completions over the OpenAI wire protocol."""

    #: Set by subclasses.
    provider_name: str = "unknown"
    base_url: str = ""
    key_env_var: str = ""
    console_url: str = ""
    max_tokens: int = DEFAULT_MAX_TOKENS

    #: How to ask for JSON. ``strict`` sends a full ``json_schema`` response
    #: format and lets the provider constrain decoding. ``object`` asks only for
    #: valid JSON and puts the schema in the prompt.
    #:
    #: Strict mode is better when the provider implements it well. Behind a
    #: gateway it often is not: measured on OpenRouter's Nemotron, the identical
    #: extraction took 8,192 tokens and 94s under strict mode, finishing with
    #: `length` and unparseable output, against 356 tokens and 2s without it.
    #: The caller validates against the Pydantic model either way, so the only
    #: thing strict mode adds is the provider's own enforcement -- worth
    #: dropping when that enforcement is the thing breaking.
    schema_mode: str = "strict"

    def __init__(self, settings: Settings, tracer: Any = None) -> None:
        self.settings = settings
        self.tracer = tracer
        self._client: Any = None

    # -- subclass contract -------------------------------------------------

    def api_key(self) -> str | None:
        raise NotImplementedError

    def model_name(self) -> str:
        raise NotImplementedError

    def extra_headers(self) -> dict[str, str]:
        return {}

    def extra_body(self) -> dict[str, Any]:
        """Non-standard request fields this provider understands."""
        return {}

    # -- transport ---------------------------------------------------------

    def has_credentials(self) -> bool:
        """Whether a credential is configured. Cheap; no network access."""
        return bool(self.api_key())

    @property
    def client(self) -> Any:
        if self._client is None:
            key = self.api_key()
            if not key:
                raise MissingAPIKeyError(
                    self.provider_name,
                    f"Set {self.key_env_var} in .env or your environment"
                    + (f" (get a key at {self.console_url})." if self.console_url else "."),
                )
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - declared dependency
                raise LLMError("The `openai` package is required: pip install openai") from exc

            self._client = OpenAI(
                api_key=key,
                base_url=self.base_url,
                timeout=120.0,
                default_headers=self.extra_headers() or None,
            )
        return self._client

    # -- completion --------------------------------------------------------

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        agent: str = "unknown",
        tools: list[dict[str, Any]] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        """Run one completion, and record it in the trace."""
        response = self._invoke(messages, tools, response_schema, agent)
        response.provider = response.provider or self.provider_name
        response.model = response.model or self.model_name()

        if self.tracer is not None:
            self.tracer.record_llm_call(
                agent=agent,
                model=response.model,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                provider=response.provider,
                finish_reason=response.finish_reason,
            )
        return response

    def complete_structured(
        self,
        messages: Sequence[dict[str, Any]],
        response_schema: type[T],
        *,
        agent: str = "unknown",
    ) -> T:
        """Complete and validate into a Pydantic model.

        On a validation failure the model is shown its own output alongside the
        error and asked once to correct it. This is a transport-level retry for
        malformed JSON, distinct from the agents' semantic critique loops.
        """
        response = self.complete(messages, agent=agent, response_schema=response_schema)
        if not response.content:
            raise LLMError(f"{agent}: model returned an empty response")

        try:
            return response_schema.model_validate(extract_json_object(response.content))
        except (ValidationError, LLMError) as exc:
            repair_messages = [
                *messages,
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": (
                        "That response did not match the required schema.\n"
                        f"Error: {exc}\n\n"
                        "Return corrected JSON only -- no prose, no code fences."
                    ),
                },
            ]
            retry = self.complete(
                repair_messages,
                agent=f"{agent}.schema_repair",
                response_schema=response_schema,
            )
            if not retry.content:
                raise LLMError(f"{agent}: schema repair returned an empty response") from exc
            try:
                return response_schema.model_validate(extract_json_object(retry.content))
            except (ValidationError, LLMError) as final_exc:
                raise LLMError(
                    f"{agent}: model could not produce valid {response_schema.__name__}: "
                    f"{final_exc}"
                ) from final_exc

    def _invoke(
        self,
        messages: Sequence[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        response_schema: type[BaseModel] | None,
        agent: str,
    ) -> LLMResponse:
        """One live call, with retries. Raises `ProviderUnavailable` if this
        provider cannot serve requests at all."""
        chat = list(messages)
        if response_schema is not None and self.schema_mode != "strict":
            chat = self._with_schema_instruction(chat, response_schema)

        payload: dict[str, Any] = {
            "model": self.model_name(),
            "messages": chat,
            "temperature": TEMPERATURE,
            "max_tokens": self.max_tokens,
        }
        extra = self.extra_body()
        if extra:
            payload["extra_body"] = extra
        if tools:
            payload["tools"] = tools
        if response_schema is not None:
            payload["response_format"] = (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_schema.__name__,
                        "schema": strict_json_schema(response_schema),
                        "strict": True,
                    },
                }
                if self.schema_mode == "strict"
                else {"type": "json_object"}
            )

        last_error: Exception | None = None
        degraded = False

        for attempt in range(MAX_RETRIES):
            try:
                return self._normalise(self.client.chat.completions.create(**payload))
            except ProviderUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001 - provider errors vary widely
                last_error = exc
                message = str(exc).lower()

                # Not every model behind a gateway honours strict json_schema.
                # Degrade to plain JSON mode rather than failing over -- the
                # caller validates against the Pydantic model regardless.
                if (
                    not degraded
                    and "response_format" in payload
                    and any(t in message for t in ("schema", "response_format", "json"))
                ):
                    payload["response_format"] = {"type": "json_object"}
                    degraded = True
                    continue

                # Transient first: a 429 or an overloaded upstream must be
                # retried, not diagnosed as a billing or credential problem.
                # Several of those strings ("insufficient capacity", "402" in a
                # longer body) would otherwise match a permanent bucket below.
                if any(token in message for token in TRANSIENT_MARKERS):
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BASE_DELAY * (2**attempt))
                        continue
                    raise ProviderUnavailable(
                        self.provider_name, "upstream busy", str(exc)
                    ) from exc

                if any(token in message for token in BILLING_MARKERS):
                    raise ProviderUnavailable(
                        self.provider_name, "account not funded", str(exc)
                    ) from exc

                if any(token in message for token in CREDENTIAL_MARKERS):
                    raise ProviderUnavailable(
                        self.provider_name, "credential rejected", str(exc)
                    ) from exc

                if any(token in message for token in MODEL_MARKERS):
                    raise ProviderUnavailable(
                        self.provider_name,
                        f"model '{self.model_name()}' unavailable",
                        str(exc),
                    ) from exc

                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY * (2**attempt))

        raise ProviderUnavailable(
            self.provider_name, f"{MAX_RETRIES} attempts failed", f"{agent}: {last_error}"
        ) from last_error

    @staticmethod
    def _with_schema_instruction(
        messages: list[dict[str, Any]], response_schema: type[BaseModel]
    ) -> list[dict[str, Any]]:
        """Append the schema to the last user turn, for non-strict mode.

        Without the provider constraining decoding, the model has to be told the
        field names -- left to itself it invents its own (`vendor`, `items`)
        and nothing validates.
        """
        instruction = (
            "\n\nReturn a single JSON object matching this schema exactly. "
            "Use these field names, include every key, and use null for anything "
            "the document does not state. No prose, no code fences.\n\n"
            + json.dumps(strict_json_schema(response_schema), indent=2)
        )

        chat = list(messages)
        for index in range(len(chat) - 1, -1, -1):
            if chat[index].get("role") == "user":
                turn = dict(chat[index])
                turn["content"] = f"{turn.get('content', '')}{instruction}"
                chat[index] = turn
                return chat

        chat.append({"role": "user", "content": instruction.strip()})
        return chat

    def _normalise(self, completion: Any) -> LLMResponse:
        # A gateway can return an error envelope with HTTP 200 and no choices.
        choices = getattr(completion, "choices", None)
        if not choices:
            error = getattr(completion, "error", None)
            # Raised as a plain LLMError so the retry loop can classify it --
            # an overloaded upstream deserves a retry, not a failover.
            raise EmptyResponse(str(error or completion)[:400])

        choice = choices[0]
        message = choice.message
        tool_calls = [
            {"id": call.id, "name": call.function.name, "arguments": call.function.arguments}
            for call in (getattr(message, "tool_calls", None) or [])
        ]

        content = message.content
        if not content and not tool_calls:
            # Some reasoning models emit their answer into `reasoning` and leave
            # `content` empty, especially when the token budget runs out
            # mid-thought. The JSON we need is often in there; the caller
            # validates it either way, so looking is strictly better than
            # reporting a blank turn.
            extra = getattr(message, "model_extra", None) or {}
            reasoning = extra.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                content = reasoning

        usage = getattr(completion, "usage", None)
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            finish_reason=getattr(choice, "finish_reason", None),
            provider=self.provider_name,
            model=getattr(completion, "model", None),
        )
