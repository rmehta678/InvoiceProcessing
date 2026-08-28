"""Provider routing, failover policy, and the OpenAI-compatible transport."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from invoice_flow.config import DEFAULT_OPENROUTER_MODEL, Settings
from invoice_flow.llm.base import LLMError, LLMResponse, ProviderUnavailable
from invoice_flow.llm.router import AllProvidersUnavailable, RoutedLLMClient


class StubBackend:
    """Minimal backend double: answers, or fails in a chosen way."""

    def __init__(
        self,
        name: str,
        *,
        credentials: bool = True,
        raises: Exception | None = None,
    ) -> None:
        self.provider_name = name
        self.key_env_var = f"{name.upper()}_API_KEY"
        self._credentials = credentials
        self._raises = raises
        self.calls = 0

    def has_credentials(self) -> bool:
        return self._credentials

    def model_name(self) -> str:
        return f"{self.provider_name}-model"

    def complete(self, *, agent: str = "x", **kwargs: Any) -> LLMResponse:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return LLMResponse(
            content="ok",
            tool_calls=[],
            prompt_tokens=1,
            completion_tokens=1,
            provider=self.provider_name,
        )

    def complete_structured(self, *, agent: str = "x", **kwargs: Any) -> Any:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return {"provider": self.provider_name}


def route(*backends: StubBackend, tracer: Any = None) -> RoutedLLMClient:
    return RoutedLLMClient(Settings.from_env(), tracer=tracer, backends=backends)


# --------------------------------------------------------------------------
# Failover policy
# --------------------------------------------------------------------------


def test_primary_is_used_when_healthy() -> None:
    primary = StubBackend("xai")
    fallback = StubBackend("openrouter")
    response = route(primary, fallback).complete(messages=[])

    assert response.provider == "xai"
    assert fallback.calls == 0, "fallback must not be touched while the primary works"


def test_falls_back_when_primary_is_unavailable() -> None:
    primary = StubBackend("xai", raises=ProviderUnavailable("xai", "account not funded"))
    fallback = StubBackend("openrouter")

    response = route(primary, fallback).complete(messages=[])
    assert response.provider == "openrouter"
    assert fallback.calls == 1


def test_does_not_fall_back_on_a_bad_request() -> None:
    """An LLMError is our bug. Failing over spends a second credential to
    reproduce the same fault, and hides it."""
    primary = StubBackend("xai", raises=LLMError("schema did not validate"))
    fallback = StubBackend("openrouter")

    with pytest.raises(LLMError, match="schema did not validate"):
        route(primary, fallback).complete(messages=[])
    assert fallback.calls == 0


def test_failover_is_sticky_across_calls() -> None:
    """Once xAI says 'no credits' it will keep saying so; re-probing it on every
    call would add a doomed round trip to each one."""
    primary = StubBackend("xai", raises=ProviderUnavailable("xai", "account not funded"))
    fallback = StubBackend("openrouter")
    client = route(primary, fallback)

    for _ in range(4):
        client.complete(messages=[])

    assert primary.calls == 1, "primary should be probed once, not once per call"
    assert fallback.calls == 4


def test_backend_without_credentials_is_skipped_without_a_call() -> None:
    primary = StubBackend("xai", credentials=False)
    fallback = StubBackend("openrouter")

    response = route(primary, fallback).complete(messages=[])
    assert response.provider == "openrouter"
    assert primary.calls == 0


def test_all_providers_unavailable_names_every_reason() -> None:
    primary = StubBackend("xai", raises=ProviderUnavailable("xai", "account not funded"))
    fallback = StubBackend("openrouter", credentials=False)

    with pytest.raises(AllProvidersUnavailable) as exc_info:
        route(primary, fallback).complete(messages=[])

    message = str(exc_info.value)
    assert "xai" in message and "not funded" in message
    assert "openrouter" in message and "OPENROUTER_API_KEY" in message
    assert "demo.py" in message, "should point at the no-credentials escape hatch"


def test_failover_is_recorded_in_the_trace() -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    class FakeTracer:
        def emit(self, event: str, **payload: Any) -> None:
            events.append((event, payload))

    primary = StubBackend("xai", raises=ProviderUnavailable("xai", "account not funded", "403"))
    fallback = StubBackend("openrouter")
    route(primary, fallback, tracer=FakeTracer()).complete(messages=[], agent="ingestion")

    failover = [payload for name, payload in events if name == "llm.provider_failover"]
    assert len(failover) == 1
    assert failover[0]["provider"] == "xai"
    assert failover[0]["falling_back_to"] == "openrouter"
    assert failover[0]["agent"] == "ingestion"


def test_structured_calls_route_identically() -> None:
    primary = StubBackend("xai", raises=ProviderUnavailable("xai", "down"))
    fallback = StubBackend("openrouter")
    result = route(primary, fallback).complete_structured(messages=[], response_schema=BaseModel)
    assert result == {"provider": "openrouter"}


def test_describe_chain_flags_missing_credentials() -> None:
    chain = route(StubBackend("xai"), StubBackend("openrouter", credentials=False))
    assert chain.describe_chain() == "xai -> openrouter (no credential)"


# --------------------------------------------------------------------------
# OpenAI-compatible transport
# --------------------------------------------------------------------------


class _FakeToolCall:
    id = "call_1"

    class function:  # noqa: N801 - mirrors the SDK's attribute shape
        name = "lookup_item"
        arguments = '{"name": "WidgetA"}'


class _FakeChoice:
    finish_reason = "stop"

    class message:  # noqa: N801
        content = '{"decision": "APPROVED"}'
        tool_calls = None


class _FakeUsage:
    prompt_tokens = 120
    completion_tokens = 30


class _FakeCompletion:
    choices = [_FakeChoice()]
    usage = _FakeUsage()
    model = "z-ai/glm-5.2:free"


class _CapturedOpenAI:
    """Stands in for `openai.OpenAI`, recording the payload sent."""

    def __init__(self, completion: Any = None, raises: Exception | None = None) -> None:
        self.payload: dict[str, Any] = {}
        self.call_count = 0
        outer = self

        class Completions:
            def create(self, **kwargs: Any) -> Any:
                outer.payload = kwargs
                outer.call_count += 1
                if raises is not None:
                    raise raises
                return completion or _FakeCompletion()

        class Chat:
            completions = Completions()

        self.chat = Chat()


def openrouter_backend(
    monkeypatch: pytest.MonkeyPatch, transport: Any = None
) -> tuple[Any, Any]:
    from invoice_flow.llm.openrouter import OpenRouterClient

    settings = Settings.from_env(openrouter_api_key="sk-test")
    backend = OpenRouterClient(settings)
    captured = transport or _CapturedOpenAI()
    monkeypatch.setattr(type(backend), "client", property(lambda self: captured))
    return backend, captured


def test_openrouter_payload_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    class Draft(BaseModel):
        decision: str

    backend, captured = openrouter_backend(monkeypatch)
    backend.complete(
        [{"role": "system", "content": "You are a VP."}, {"role": "user", "content": "Decide."}],
        agent="approval.vp",
        response_schema=Draft,
    )

    payload = captured.payload
    # OpenRouter speaks the OpenAI wire protocol, so messages pass through
    # unchanged -- `system` stays a message role, and no translation layer is
    # needed at all.
    assert payload["messages"][0]["role"] == "system"
    assert payload["model"] == DEFAULT_OPENROUTER_MODEL
    # Reasoning models spend output tokens thinking; without an explicit
    # ceiling a gateway default can be consumed before any content is emitted.
    assert payload["max_tokens"] > 0

    # Non-strict mode: ask for JSON, carry the schema in the prompt. Gateway
    # constrained decoding measured pathologically slow on this model.
    assert payload["response_format"] == {"type": "json_object"}
    prompt = payload["messages"][-1]["content"]
    assert "decision" in prompt and "schema" in prompt.lower()


def test_openrouter_holds_model_reasoning_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pipeline supplies its own reasoning structure; an unconstrained
    reasoning model spends its whole budget thinking and gets truncated."""
    backend, captured = openrouter_backend(monkeypatch)
    backend.complete([{"role": "user", "content": "Hi"}], agent="test")
    assert captured.payload["extra_body"]["reasoning"]["effort"] == "low"


def test_grok_uses_strict_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """xAI implements strict decoding well, so it keeps the stronger guarantee."""
    from invoice_flow.llm.grok import GrokClient

    class Draft(BaseModel):
        decision: str

    backend = GrokClient(Settings.from_env(api_key="xai-test"))
    captured = _CapturedOpenAI()
    monkeypatch.setattr(type(backend), "client", property(lambda self: captured))
    backend.complete([{"role": "user", "content": "Hi"}], agent="t", response_schema=Draft)

    schema = captured.payload["response_format"]["json_schema"]
    assert captured.payload["response_format"]["type"] == "json_schema"
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False
    assert "decision" in schema["schema"]["required"]


def test_openrouter_tools_pass_through_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    from invoice_flow.tools.inventory import INVENTORY_TOOL_SCHEMAS

    backend, captured = openrouter_backend(monkeypatch)
    backend.complete(
        [{"role": "user", "content": "Check."}], agent="v", tools=INVENTORY_TOOL_SCHEMAS
    )

    assert captured.payload["tools"] == INVENTORY_TOOL_SCHEMAS
    assert "response_format" not in captured.payload


def test_openrouter_response_is_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, _ = openrouter_backend(monkeypatch)
    response = backend.complete([{"role": "user", "content": "Hi"}], agent="test")

    assert response.content == '{"decision": "APPROVED"}'
    assert response.provider == "openrouter"
    assert response.model == "z-ai/glm-5.2:free"
    assert response.prompt_tokens == 120


def test_tool_calls_are_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Choice:
        finish_reason = "tool_calls"

        class message:  # noqa: N801
            content = None
            tool_calls = [_FakeToolCall()]

    class WithTools:
        choices = [_Choice()]
        usage = _FakeUsage()
        model = "z-ai/glm-5.2:free"

    backend, _ = openrouter_backend(monkeypatch, _CapturedOpenAI(completion=WithTools()))
    response = backend.complete([{"role": "user", "content": "Hi"}], agent="test")

    assert response.tool_calls == [
        {"id": "call_1", "name": "lookup_item", "arguments": '{"name": "WidgetA"}'}
    ]


def test_attribution_headers_are_sent() -> None:
    from invoice_flow.llm.openrouter import OpenRouterClient

    headers = OpenRouterClient(Settings.from_env()).extra_headers()
    assert "HTTP-Referer" in headers and "X-Title" in headers


@pytest.mark.parametrize(
    "error,expected_reason",
    [
        (Exception("Error code: 402 - insufficient credits"), "account not funded"),
        (Exception("Error code: 401 - No auth credentials found"), "credential rejected"),
        (Exception("Error code: 404 - No endpoints found for model"), "unavailable"),
    ],
)
def test_provider_errors_are_classified(
    monkeypatch: pytest.MonkeyPatch, error: Exception, expected_reason: str
) -> None:
    """Each failure must be named accurately -- 'check your key' when the key is
    fine sends someone chasing the wrong problem."""
    backend, _ = openrouter_backend(monkeypatch, _CapturedOpenAI(raises=error))

    with pytest.raises(ProviderUnavailable) as exc_info:
        backend.complete([{"role": "user", "content": "Hi"}], agent="test")
    assert expected_reason in exc_info.value.reason


def test_unsupported_schema_degrades_before_failing_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not every model on a gateway honours strict json_schema. Degrade to plain
    JSON mode rather than abandoning the provider."""

    class Draft(BaseModel):
        decision: str

    calls: list[dict[str, Any]] = []

    class Flaky:
        def __init__(self) -> None:
            outer = self
            self.payload: dict[str, Any] = {}

            class Completions:
                def create(self, **kwargs: Any) -> Any:
                    calls.append(kwargs)
                    outer.payload = kwargs
                    if kwargs.get("response_format", {}).get("type") == "json_schema":
                        raise Exception("400 - json_schema response_format is not supported")
                    return _FakeCompletion()

            class Chat:
                completions = Completions()

            self.chat = Chat()

    from invoice_flow.llm.grok import GrokClient

    backend = GrokClient(Settings.from_env(api_key="xai-test"))
    monkeypatch.setattr(type(backend), "client", property(lambda self: Flaky()))
    response = backend.complete(
        [{"role": "user", "content": "Hi"}], agent="t", response_schema=Draft
    )

    assert response.content is not None
    assert len(calls) == 2
    assert calls[1]["response_format"] == {"type": "json_object"}


def test_empty_choices_is_retried_then_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gateway can return an error envelope with HTTP 200 and no choices."""

    class Empty:
        choices: list[Any] = []
        error = {"message": "upstream provider returned nothing"}

    transport = _CapturedOpenAI(completion=Empty())
    backend, _ = openrouter_backend(monkeypatch, transport)
    monkeypatch.setattr("invoice_flow.llm.openai_compat.RETRY_BASE_DELAY", 0)

    with pytest.raises(ProviderUnavailable):
        backend.complete([{"role": "user", "content": "Hi"}], agent="test")
    assert transport.call_count > 1, "an empty response should be retried, not given up on"


def test_transient_errors_are_retried_not_failed_over(monkeypatch: pytest.MonkeyPatch) -> None:
    """An overloaded free endpoint is the one failure where retrying is right.

    Classifying a 502 as a billing or credential problem would abandon a
    provider that is merely busy -- and would diagnose it wrongly.
    """
    attempts: list[int] = []

    class Flaky:
        def __init__(self) -> None:
            self.payload: dict[str, Any] = {}

            class Completions:
                def create(self, **kwargs: Any) -> Any:
                    attempts.append(1)
                    if len(attempts) == 1:
                        raise Exception("Error code: 502 - Service temporarily overloaded")
                    return _FakeCompletion()

            class Chat:
                completions = Completions()

            self.chat = Chat()

    monkeypatch.setattr("invoice_flow.llm.openai_compat.RETRY_BASE_DELAY", 0)
    backend, _ = openrouter_backend(monkeypatch, Flaky())
    response = backend.complete([{"role": "user", "content": "Hi"}], agent="test")

    assert response.content is not None
    assert len(attempts) == 2, "should have retried once and succeeded"


def test_reasoning_only_response_is_recovered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reasoning models sometimes leave `content` empty and put the answer in
    `reasoning`. Reporting a blank turn there loses a usable response."""

    class _Msg:
        content = None
        tool_calls = None
        model_extra = {"reasoning": '{"agrees": true}'}

    class _Choice:
        finish_reason = "stop"
        message = _Msg()

    class ReasoningOnly:
        choices = [_Choice()]
        usage = _FakeUsage()
        model = "reasoner"

    backend, _ = openrouter_backend(monkeypatch, _CapturedOpenAI(completion=ReasoningOnly()))
    response = backend.complete([{"role": "user", "content": "Hi"}], agent="test")

    assert response.content == '{"agrees": true}'
