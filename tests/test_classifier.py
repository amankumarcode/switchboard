import pytest

from switchboard.classifier import Classifier
from switchboard.config import OUT_OF_SCOPE, Confidence
from switchboard.providers import MockProvider


def _response(intent="change_pin", confidence="high", secondary=()):
    return {
        "reasoning": "test",
        "intent": intent,
        "secondary_intents": list(secondary),
        "confidence": confidence,
    }


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("I want to change my PIN", "change_pin"),
        ("How do I activate my new card?", "activate_my_card"),
        ("My card was stolen", "lost_or_stolen_card"),
        ("What's the weather like in Paris?", OUT_OF_SCOPE),
    ],
)
def test_mock_provider_classifies_obvious_queries(banking_config, query, expected):
    result = Classifier(banking_config, MockProvider(banking_config)).classify(query)
    assert result.classification.intent == expected
    assert result.error is None
    assert result.model == "mock"
    assert result.input_tokens > 0


def test_valid_response_passes_through(banking_config):
    provider = MockProvider(responses=[_response(secondary=["card_arrival"])])
    result = Classifier(banking_config, provider).classify("anything")
    c = result.classification
    assert (c.intent, c.confidence, c.secondary_intents) == (
        "change_pin",
        Confidence.HIGH,
        ["card_arrival"],
    )
    assert result.prompt_version == Classifier(banking_config, provider).prompt_version


def test_secondary_intents_are_cleaned(banking_config):
    provider = MockProvider(
        responses=[_response(secondary=["change_pin", "card_arrival", "card_arrival", "bogus"])]
    )
    result = Classifier(banking_config, provider).classify("anything")
    assert result.classification.secondary_intents == ["card_arrival"]


@pytest.mark.parametrize(
    "bad",
    [
        _response(intent="not_an_intent"),
        _response(confidence="very_high"),
        {"reasoning": "missing fields"},
    ],
    ids=["unknown-intent", "bad-confidence", "missing-fields"],
)
def test_invalid_output_falls_back_to_out_of_scope(banking_config, bad):
    result = Classifier(banking_config, MockProvider(responses=[bad])).classify("anything")
    assert result.classification.intent == OUT_OF_SCOPE
    assert result.classification.confidence is Confidence.LOW
    assert result.error.startswith("invalid_output")


def test_provider_error_falls_back_to_out_of_scope(banking_config):
    provider = MockProvider(responses=[TimeoutError("upstream timed out")])
    result = Classifier(banking_config, provider).classify("anything")
    assert result.classification.intent == OUT_OF_SCOPE
    assert result.classification.confidence is Confidence.LOW
    assert "TimeoutError" in result.error
