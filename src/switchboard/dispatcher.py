"""Dispatcher: runs the handler the policy chose; a failing handler escalates to a human."""

from __future__ import annotations

from pydantic import BaseModel

from switchboard.config import RouterConfig
from switchboard.handlers.registry import (
    HandlerContext,
    HandlerResult,
    get_handler,
    load_handlers,
)
from switchboard.models import Action, Classification, EscalationReason, RoutingDecision


class DispatchOutcome(BaseModel):
    decision: RoutingDecision  # the decision actually carried out (may differ after a failure)
    result: HandlerResult
    handler_error: str | None = None


class Dispatcher:
    def __init__(self, config: RouterConfig):
        load_handlers(config)
        self._human_queue = config.out_of_scope.handler

    def dispatch(
        self,
        decision: RoutingDecision,
        *,
        query: str,
        classification: Classification,
        trace_id: str,
        customer_id: str,
    ) -> DispatchOutcome:
        ctx = HandlerContext(
            query=query,
            trace_id=trace_id,
            customer_id=customer_id,
            classification=classification,
            decision=decision,
        )
        try:
            return DispatchOutcome(decision=decision, result=get_handler(decision.handler)(ctx))
        except Exception as exc:
            if decision.handler == self._human_queue:
                raise  # nowhere left to escalate to
            error = f"{type(exc).__name__}: {exc}"

        escalated = RoutingDecision(
            action=Action.HUMAN,
            intent=decision.intent,
            handler=self._human_queue,
            reasons=[*decision.reasons, EscalationReason.HANDLER_ERROR],
        )
        result = get_handler(self._human_queue)(ctx.model_copy(update={"decision": escalated}))
        return DispatchOutcome(decision=escalated, result=result, handler_error=error)
