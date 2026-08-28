"""Provider routing with failover.

Tries xAI first and falls back to OpenRouter when xAI cannot serve. Two rules
make this behave sensibly rather than merely working:

**Only `ProviderUnavailable` triggers failover.** A malformed request or a
schema the model cannot satisfy is our bug; retrying it on a second provider
burns another account's credits to reproduce the same fault.

**Failover is sticky for the rest of the run.** Once xAI has answered "no
credits", it will answer that for every subsequent call. Re-probing it on each
of the dozen-odd calls in one invoice would add a doomed round trip every time
and scatter the explanation through the log. The switch is recorded once,
prominently, and the run continues on one provider.
"""

from __future__ import annotations

from typing import Any, Sequence

from pydantic import BaseModel

from ..config import Settings
from .base import LLMError, LLMResponse, ProviderUnavailable
from .grok import GrokClient
from .openai_compat import OpenAICompatibleClient
from .openrouter import OpenRouterClient


class AllProvidersUnavailable(LLMError):
    """Every configured provider refused to serve."""

    def __init__(self, failures: list[ProviderUnavailable]) -> None:
        self.failures = failures
        lines = [f"  - {failure}" for failure in failures]
        super().__init__(
            "No LLM provider could serve the request.\n"
            + "\n".join(lines)
            + "\n\nFix one of the above, or run without live calls: python scripts/demo.py"
        )


class RoutedLLMClient:
    """Presents one client surface over an ordered list of backends."""

    def __init__(
        self,
        settings: Settings,
        tracer: Any = None,
        backends: Sequence[OpenAICompatibleClient] | None = None,
    ) -> None:
        self.settings = settings
        self.tracer = tracer
        # The `backends` seam exists so the failover policy can be tested
        # against doubles; nothing in the application passes it.
        self._backends: list[OpenAICompatibleClient] = list(backends) if backends is not None else [
            GrokClient(settings, tracer),
            OpenRouterClient(settings, tracer),
        ]
        self._disabled: dict[str, ProviderUnavailable] = {}

    def describe_chain(self) -> str:
        """Human-readable provider order, for startup output."""
        parts = []
        for backend in self._backends:
            mark = "" if backend.has_credentials() else " (no credential)"
            parts.append(f"{backend.provider_name}{mark}")
        return " -> ".join(parts)

    # -- routing -----------------------------------------------------------

    def _dispatch(self, call: str, agent: str, **kwargs: Any) -> Any:
        failures: list[ProviderUnavailable] = list(self._disabled.values())

        for backend in self._backends:
            name = backend.provider_name
            if name in self._disabled:
                continue

            # Skip a backend with no credential without burning a round trip,
            # but record it so the final error names every reason.
            if not backend.has_credentials():
                failure = ProviderUnavailable(
                    name,
                    "no credential",
                    f"Set {backend.key_env_var} in .env to use this provider.",
                )
                self._disable(name, failure, agent, silent=True)
                failures.append(failure)
                continue

            try:
                return getattr(backend, call)(agent=agent, **kwargs)
            except ProviderUnavailable as exc:
                self._disable(name, exc, agent)
                failures.append(exc)
                continue

        raise AllProvidersUnavailable(failures)

    def _disable(
        self,
        name: str,
        failure: ProviderUnavailable,
        agent: str,
        silent: bool = False,
    ) -> None:
        if name in self._disabled:
            return
        self._disabled[name] = failure

        remaining = [
            backend.provider_name
            for backend in self._backends
            if backend.provider_name not in self._disabled
        ]
        if self.tracer is not None and not silent:
            self.tracer.emit(
                "llm.provider_failover",
                provider=name,
                reason=failure.reason,
                detail=failure.detail,
                agent=agent,
                falling_back_to=remaining[0] if remaining else None,
            )

    # -- client surface ----------------------------------------------------

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        agent: str = "unknown",
        tools: list[dict[str, Any]] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        return self._dispatch(
            "complete",
            agent,
            messages=messages,
            tools=tools,
            response_schema=response_schema,
        )

    def complete_structured(
        self,
        messages: Sequence[dict[str, Any]],
        response_schema: type[BaseModel],
        *,
        agent: str = "unknown",
    ) -> Any:
        return self._dispatch(
            "complete_structured",
            agent,
            messages=messages,
            response_schema=response_schema,
        )


def build_client(settings: Settings, tracer: Any = None) -> RoutedLLMClient:
    """Construct the routed client for a run."""
    return RoutedLLMClient(settings, tracer)
