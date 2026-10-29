import json
import warnings
from types import SimpleNamespace

import pytest

from switchboard.classifier import Classifier
from switchboard.config import OUT_OF_SCOPE, ModelConfig
from switchboard.prompt import build_schema
from switchboard.providers.anthropic import AnthropicProvider, ProviderError
from switchboard.providers.factory import get_provider, model_settings
from switchboard.providers.mock import MockProvider

VALID = {
    "reasoning": "Card not received yet.",
    "intent": "card_arrival",
    "secondary_intents": [],
    "confidence": "high",
}
VALID_JSON = json.dumps(VALID)


class FakeClient:
    """Stands in for anthropic.Anthropic(); records the kwargs of each call."""

    def __init__(self, text=VALID_JSON, stop_reason="end_turn", content=None):
        self.calls: list[dict] = []
        self._response = SimpleNamespace(
            content=content if content is not None else [SimpleNamespace(type="text", text=text)],
            stop_reason=stop_reason,
            model="claude-haiku-5-5",
            usage=SimpleNamespace(input_tokens=812, output_tokens=41),
        )
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def _provider(client=None, **settings):
    return AnthropicProvider(ModelConfig(**settings), client=client or FakeClient())


def test_request_shape_with_defaults(banking_config):
    client = FakeClient()
    schema = build_schema(banking_config)
    _provider(client).complete("SYSTEM", "<query>\nhi\n</query>", schema)

    call = client.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["max_tokens"] == 300
    assert call["system"] == "SYSTEM"
    assert call["messages"] == [{"role": "user", "content": "<query>\nhi\n</query>"}]
    assert call["output_config"] == {
        "format": {"type": "json_schema", "schema": schema},
        "effort": "low",
    }
    assert call["thinking"] == {"type": "disabled"}
    assert "extra_body" not in call and "extra_headers" not in call


def test_configurable_settings_reach_the_request():
    client = FakeClient()
    _provider(
        client, name="claude-sonnet-5-5", max_tokens=1000, effort="high", thinking=True
    ).complete("s", "u", {})
    call = client.calls[0]
    assert (call["model"], call["max_tokens"]) == ("claude-sonnet-5-5", 1000)
    assert call["output_config"]["effort"] == "high"
    assert call["thinking"] == {"type": "adaptive"}


def test_effort_can_be_omitted():
    client = FakeClient()
    _provider(client, effort=None).complete("s", "u", {})
    assert "effort" not in client.calls[0]["output_config"]


def test_temperature_sent_only_with_pinned_api_version():
    client = FakeClient()
    _provider(client, temperature=0.0, api_version="2023-06-01").complete("s", "u", {})
    call = client.calls[0]
    assert call["extra_body"] == {"temperature": 0.0}
    assert call["extra_headers"] == {"anthropic-version": "2023-06-01"}


def test_temperature_without_api_version_is_ignored_with_warning():
    client = FakeClient()
    with pytest.warns(UserWarning, match="temperature is ignored"):
        provider = _provider(client, temperature=0.0)
    provider.complete("s", "u", {})
    assert "extra_body" not in client.calls[0]


def test_parses_json_and_usage():
    response = _provider().complete("s", "u", {})
    assert response.data == VALID
    assert (response.input_tokens, response.output_tokens) == (812, 41)
    assert response.model == "claude-haiku-5-5"


@pytest.mark.parametrize(
    ("client", "message"),
    [
        (FakeClient(stop_reason="refusal"), "refused"),
        (FakeClient(stop_reason="max_tokens"), "truncated"),
        (FakeClient(text="not json"), "not valid JSON"),
        (FakeClient(content=[]), "no text block"),
    ],
    ids=["refusal", "max-tokens", "bad-json", "no-text"],
)
def test_unusable_responses_raise(client, message):
    with pytest.raises(ProviderError, match=message):
        _provider(client).complete("s", "u", {})


def test_classifier_falls_back_on_provider_error(banking_config):
    provider = _provider(FakeClient(stop_reason="refusal"))
    result = Classifier(banking_config, provider).classify("anything")
    assert result.classification.intent == OUT_OF_SCOPE
    assert "ProviderError" in result.error


def test_classifier_end_to_end_with_fake_client(banking_config):
    result = Classifier(banking_config, _provider()).classify("where is my card?")
    assert result.classification.intent == "card_arrival"
    assert result.input_tokens == 812


def test_model_settings_env_overrides(banking_config):
    env = {
        "SWITCHBOARD_MODEL": "claude-sonnet-5-5",
        "SWITCHBOARD_MAX_TOKENS": "512",
        "SWITCHBOARD_EFFORT": "medium",
        "SWITCHBOARD_THINKING": "true",
        "SWITCHBOARD_TEMPERATURE": "",  # empty values are ignored
    }
    settings = model_settings(banking_config, env)
    assert settings.name == "claude-sonnet-5-5"
    assert settings.max_tokens == 512
    assert settings.effort.value == "medium"
    assert settings.thinking is True
    assert settings.temperature is None
    assert settings.pricing == banking_config.model.pricing


def test_model_settings_without_overrides_returns_config(banking_config):
    assert model_settings(banking_config, {}) is banking_config.model


def test_get_provider(banking_config):
    assert isinstance(get_provider(banking_config, {}), MockProvider)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        provider = get_provider(
            banking_config, {"SWITCHBOARD_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test"}
        )
    assert isinstance(provider, AnthropicProvider)
    with pytest.raises(ValueError, match="unknown SWITCHBOARD_PROVIDER"):
        get_provider(banking_config, {"SWITCHBOARD_PROVIDER": "openai"})
