"""xAI Grok backend -- the case study's stated reasoning engine, and the default.

The xAI API speaks the OpenAI wire protocol, so this is a thin configuration of
`OpenAICompatibleClient`: a base URL, a credential, and a model.

(The snippet in the case README, ``from xai import Grok``, does not correspond
to a published package. The official SDK is ``xai-sdk``; the OpenAI-compatible
endpoint used here is the better-tested path and keeps the tool-calling loop
conventional.)
"""

from __future__ import annotations

from .openai_compat import OpenAICompatibleClient


class GrokClient(OpenAICompatibleClient):
    """Chat completions against xAI's OpenAI-compatible endpoint."""

    provider_name = "xai"
    key_env_var = "XAI_API_KEY"
    console_url = "https://console.x.ai"

    @property
    def base_url(self) -> str:  # type: ignore[override]
        return self.settings.base_url

    def api_key(self) -> str | None:
        return self.settings.api_key

    def model_name(self) -> str:
        return self.settings.model
