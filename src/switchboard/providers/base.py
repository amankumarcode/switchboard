"""Provider interface: anything that turns a prompt + JSON schema into JSON."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel


class ProviderResponse(BaseModel):
    data: dict[str, Any]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMProvider(Protocol):
    def complete(self, system: str, user: str, schema: dict) -> ProviderResponse:
        """Return output conforming to `schema`. May raise on transport errors."""
        ...
