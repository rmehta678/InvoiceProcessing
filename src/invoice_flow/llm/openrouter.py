"""OpenRouter backend -- the fallback when xAI cannot serve.

OpenRouter is a gateway in front of many providers, speaking the OpenAI wire
protocol, so it reuses the same transport as xAI. Its value here is that one
credential reaches every model it fronts: switching the fallback is an
`OPENROUTER_MODEL` edit rather than a new backend.

Model ids are namespaced (``vendor/model``). Not every model behind the gateway
supports tool calling or schema-constrained output, and the pipeline needs both:
check a candidate's ``supported_parameters`` before switching.
"""

from __future__ import annotations

from typing import Any

from .openai_compat import OpenAICompatibleClient

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(OpenAICompatibleClient):
    """Chat completions through the OpenRouter gateway."""

    provider_name = "openrouter"
    base_url = OPENROUTER_BASE_URL
    key_env_var = "OPENROUTER_API_KEY"
    console_url = "https://openrouter.ai/keys"
    # Gateway-side strict decoding proved pathological here (see `schema_mode`
    # in openai_compat); ask for JSON and put the schema in the prompt instead.
    schema_mode = "object"

    def api_key(self) -> str | None:
        return self.settings.openrouter_api_key

    def model_name(self) -> str:
        return self.settings.openrouter_model

    def extra_headers(self) -> dict[str, str]:
        """Attribution headers OpenRouter uses for its public rankings.

        Optional, and harmless to send; they identify the calling app rather
        than the user.
        """
        return {
            "HTTP-Referer": "https://github.com/galatiq-case-invoices",
            "X-Title": "Invoice Processing Automation",
        }

    def extra_body(self) -> dict[str, Any]:
        """Hold model-level reasoning down; this pipeline does its own.

        Left unconstrained, a reasoning model spends its entire output budget
        thinking and gets truncated mid-JSON -- measured on Nemotron: 8,192
        tokens and ~150s per call, finishing with `length` and unparseable
        output, against 199 tokens and 6s at low effort for the same result.

        The reasoning that matters here is structural and already in the graph:
        the arithmetic-grounded extraction repair loop and the VP/audit critique
        cycle. Paying a model to also ruminate duplicates that, slowly.

        Ignored by providers that do not support it.
        """
        effort = self.settings.openrouter_reasoning_effort
        if not effort or effort == "default":
            return {}
        return {"reasoning": {"effort": effort}}
