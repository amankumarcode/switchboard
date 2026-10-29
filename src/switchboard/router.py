"""Router: classify -> policy -> dispatch -> log, for one query at a time."""

from __future__ import annotations

import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from switchboard.classifier import Classifier
from switchboard.config import Confidence, RouterConfig, load_config
from switchboard.decision_log import DecisionLog
from switchboard.dispatcher import Dispatcher
from switchboard.handlers.registry import HandlerStatus
from switchboard.models import Action, EscalationReason
from switchboard.policy import decide
from switchboard.providers.base import LLMProvider
from switchboard.providers.factory import get_provider

MAX_QUERY_CHARS = 2000
DEFAULT_CONFIG = Path("configs/banking.yaml")


class RouteResult(BaseModel):
    trace_id: str
    intent: str
    secondary_intents: list[str]
    confidence: Confidence
    reasoning: str
    action: Action
    handler: str
    reasons: list[EscalationReason]
    status: HandlerStatus
    message: str
    data: dict[str, Any]
    prompt_version: str
    model: str
    latency_ms: float


class Router:
    def __init__(
        self,
        config: RouterConfig,
        provider: LLMProvider,
        *,
        decision_log: DecisionLog | None = None,
        default_customer_id: str | None = None,
    ):
        self.config = config
        self.classifier = Classifier(config, provider)
        self.dispatcher = Dispatcher(config)
        self.decision_log = decision_log or DecisionLog()
        self.default_customer_id = (
            default_customer_id or os.environ.get("SWITCHBOARD_DEFAULT_CUSTOMER") or "anonymous"
        )

    @classmethod
    def from_env(cls, config_path: str | Path | None = None) -> Router:
        """Build from a config file (SWITCHBOARD_CONFIG or the banking default) and the
        provider selected by SWITCHBOARD_PROVIDER."""
        config = load_config(config_path or os.environ.get("SWITCHBOARD_CONFIG") or DEFAULT_CONFIG)
        return cls(config, get_provider(config))

    def route(self, query: str, customer_id: str | None = None) -> RouteResult:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query is longer than {MAX_QUERY_CHARS} characters")

        start = time.perf_counter()
        trace_id = uuid.uuid4().hex
        customer_id = customer_id or self.default_customer_id

        classified = self.classifier.classify(query)
        c = classified.classification
        decision = decide(c, self.config, classifier_error=classified.error is not None)
        outcome = self.dispatcher.dispatch(
            decision,
            query=query,
            classification=c,
            trace_id=trace_id,
            customer_id=customer_id,
        )
        final = outcome.decision
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        self.decision_log.write(
            {
                "trace_id": trace_id,
                "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "prompt_version": classified.prompt_version,
                "model": classified.model,
                "customer_id": customer_id,
                "query": query,
                "intent": c.intent,
                "secondary_intents": c.secondary_intents,
                "confidence": c.confidence.value,
                "action": final.action.value,
                "handler": final.handler,
                "reasons": [r.value for r in final.reasons],
                "status": outcome.result.status.value,
                "classifier_latency_ms": classified.latency_ms,
                "latency_ms": latency_ms,
                "input_tokens": classified.input_tokens,
                "output_tokens": classified.output_tokens,
                "classifier_error": classified.error,
                "handler_error": outcome.handler_error,
            }
        )

        return RouteResult(
            trace_id=trace_id,
            intent=c.intent,
            secondary_intents=c.secondary_intents,
            confidence=c.confidence,
            reasoning=c.reasoning,
            action=final.action,
            handler=final.handler,
            reasons=final.reasons,
            status=outcome.result.status,
            message=outcome.result.message,
            data=outcome.result.data,
            prompt_version=classified.prompt_version,
            model=classified.model,
            latency_ms=latency_ms,
        )
