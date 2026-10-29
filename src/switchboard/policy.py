"""Policy layer: plain code that decides whether a classification may be acted on
automatically. The LLM proposes; this module disposes."""

from __future__ import annotations

from switchboard.config import OUT_OF_SCOPE, RouterConfig
from switchboard.models import Action, Classification, EscalationReason, RoutingDecision


def decide(
    classification: Classification, config: RouterConfig, *, classifier_error: bool = False
) -> RoutingDecision:
    """Route to the intent's handler, or to the human queue with every reason that applies.

    A classifier error short-circuits: its fallback intent and confidence are synthetic,
    so the other rules would only add noise.
    """
    human_queue = config.out_of_scope.handler
    intent = classification.intent

    if classifier_error:
        return RoutingDecision(
            action=Action.HUMAN,
            intent=intent,
            handler=human_queue,
            reasons=[EscalationReason.CLASSIFIER_ERROR],
        )

    reasons: list[EscalationReason] = []
    if intent == OUT_OF_SCOPE:
        reasons.append(EscalationReason.OUT_OF_SCOPE)
    if classification.confidence.rank < config.policy.min_confidence.rank:
        reasons.append(EscalationReason.LOW_CONFIDENCE)
    if (
        intent != OUT_OF_SCOPE
        and config.get_intent(intent).risk in config.policy.human_review_risks
    ):
        reasons.append(EscalationReason.HIGH_RISK)

    if reasons:
        return RoutingDecision(
            action=Action.HUMAN, intent=intent, handler=human_queue, reasons=reasons
        )
    return RoutingDecision(
        action=Action.AUTO, intent=intent, handler=config.get_intent(intent).handler
    )
