"""Claude provider: structured outputs constrained to the classifier's JSON schema."""

from __future__ import annotations

import json
import warnings
from typing import Any

import anthropic

from switchboard.config import ModelConfig
from switchboard.providers.base import ProviderResponse


class ProviderError(RuntimeError):
    """The model answered, but not with usable output (refusal, truncation, bad JSON)."""


class AnthropicProvider:
    def __init__(self, settings: ModelConfig, client: Any | None = None):
        self.settings = settings
        self.model = settings.name
        self._client = client or anthropic.Anthropic(
            timeout=settings.timeout_s, max_retries=settings.max_retries
        )
        self._extra = self._sampling_overrides(settings)

    @staticmethod
    def _sampling_overrides(settings: ModelConfig) -> dict[str, Any]:
        """Temperature is not a parameter of the current API, so it travels in
        `extra_body` together with a pinned older `anthropic-version` header."""
        extra: dict[str, Any] = {}
        if settings.api_version:
            extra["extra_headers"] = {"anthropic-version": settings.api_version}
        if settings.temperature is not None:
            if settings.api_version:
                extra["extra_body"] = {"temperature": settings.temperature}
            else:
                warnings.warn(
                    "temperature is ignored: the current API version rejects sampling "
                    "parameters. Set model.api_version to an older version (and a model) "
                    "that accepts them.",
                    stacklevel=3,
                )
        return extra

    def complete(self, system: str, user: str, schema: dict) -> ProviderResponse:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if self.settings.effort is not None:
            output_config["effort"] = self.settings.effort.value

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.settings.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
            thinking={"type": "adaptive" if self.settings.thinking else "disabled"},
            **self._extra,
        )

        if response.stop_reason == "refusal":
            raise ProviderError("model refused the request")
        if response.stop_reason == "max_tokens":
            raise ProviderError(f"output truncated at max_tokens={self.settings.max_tokens}")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise ProviderError("response contained no text block")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"response was not valid JSON: {exc}") from exc

        return ProviderResponse(
            data=data,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
