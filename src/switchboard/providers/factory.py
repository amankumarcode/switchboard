"""Pick a provider from the environment and apply SWITCHBOARD_* model overrides."""

from __future__ import annotations

import os
from collections.abc import Mapping

from switchboard.config import ModelConfig, RouterConfig
from switchboard.providers.base import LLMProvider
from switchboard.providers.mock import MockProvider

# env variable -> ModelConfig field
_MODEL_ENV_OVERRIDES = {
    "SWITCHBOARD_MODEL": "name",
    "SWITCHBOARD_MAX_TOKENS": "max_tokens",
    "SWITCHBOARD_EFFORT": "effort",
    "SWITCHBOARD_THINKING": "thinking",
    "SWITCHBOARD_TEMPERATURE": "temperature",
    "SWITCHBOARD_API_VERSION": "api_version",
    "SWITCHBOARD_TIMEOUT_S": "timeout_s",
    "SWITCHBOARD_MAX_RETRIES": "max_retries",
}


def model_settings(config: RouterConfig, env: Mapping[str, str] | None = None) -> ModelConfig:
    """Config values, overridden by any non-empty SWITCHBOARD_* env variables."""
    env = os.environ if env is None else env
    overrides = {field: env[var] for var, field in _MODEL_ENV_OVERRIDES.items() if env.get(var)}
    if not overrides:
        return config.model
    return ModelConfig.model_validate(config.model.model_dump() | overrides)


def get_provider(config: RouterConfig, env: Mapping[str, str] | None = None) -> LLMProvider:
    env = os.environ if env is None else env
    name = env.get("SWITCHBOARD_PROVIDER", "mock").strip().lower()
    if name == "mock":
        return MockProvider(config)
    if name == "anthropic":
        from switchboard.providers.anthropic import AnthropicProvider

        return AnthropicProvider(model_settings(config, env))
    raise ValueError(f"unknown SWITCHBOARD_PROVIDER {name!r} (expected 'mock' or 'anthropic')")
