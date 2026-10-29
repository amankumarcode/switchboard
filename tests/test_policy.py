import pytest

from switchboard.config import OUT_OF_SCOPE, Confidence, PolicyConfig, Risk
from switchboard.models import Action, Classification, EscalationReason
from switchboard.policy import decide

CE = EscalationReason.CLASSIFIER_ERROR
OOS = EscalationReason.OUT_OF_SCOPE
LOW = EscalationReason.LOW_CONFIDENCE
RISK = EscalationReason.HIGH_RISK


@pytest.mark.parametrize(
    ("intent", "confidence", "error", "action", "handler", "reasons"),
    [
        # low-risk, high confidence -> automated to the intent's handler
        ("change_pin", "high", False, Action.AUTO, "pin_change", []),
        ("card_arrival", "high", False, Action.AUTO, "card_delivery_status", []),
        # below min_confidence (high)
        ("change_pin", "medium", False, Action.HUMAN, "human_queue", [LOW]),
        ("change_pin", "low", False, Action.HUMAN, "human_queue", [LOW]),
        # high-risk intents always go to a human, whatever the confidence
        ("lost_or_stolen_card", "high", False, Action.HUMAN, "human_queue", [RISK]),
        ("cancel_transfer", "high", False, Action.HUMAN, "human_queue", [RISK]),
        ("compromised_card", "low", False, Action.HUMAN, "human_queue", [LOW, RISK]),
        # out of scope
        (OUT_OF_SCOPE, "high", False, Action.HUMAN, "human_queue", [OOS]),
        (OUT_OF_SCOPE, "medium", False, Action.HUMAN, "human_queue", [OOS, LOW]),
        # classifier errors short-circuit to a single reason
        (OUT_OF_SCOPE, "low", True, Action.HUMAN, "human_queue", [CE]),
    ],
)
def test_decide(banking_config, intent, confidence, error, action, handler, reasons):
    decision = decide(
        Classification(intent=intent, confidence=confidence),
        banking_config,
        classifier_error=error,
    )
    assert decision.action is action
    assert decision.handler == handler
    assert decision.reasons == reasons
    assert decision.intent == intent  # reviewers still see the predicted intent


def _with_policy(config, **policy):
    return config.model_copy(update={"policy": PolicyConfig(**policy)})


def test_threshold_comes_from_config(banking_config):
    lenient = _with_policy(banking_config, min_confidence=Confidence.LOW)
    decision = decide(Classification(intent="change_pin", confidence="low"), lenient)
    assert decision.action is Action.AUTO


def test_risk_review_comes_from_config(banking_config):
    no_review = _with_policy(banking_config, min_confidence="high", human_review_risks=[])
    decision = decide(Classification(intent="lost_or_stolen_card", confidence="high"), no_review)
    assert decision.action is Action.AUTO
    assert decision.handler == "card_block"

    review_all = _with_policy(banking_config, human_review_risks=[Risk.LOW, Risk.HIGH])
    decision = decide(Classification(intent="change_pin", confidence="high"), review_all)
    assert decision.reasons == [RISK]
