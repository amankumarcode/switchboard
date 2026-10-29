"""Data models passed between pipeline stages."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from switchboard.config import Confidence


class Classification(BaseModel):
    """What the classifier decided about a query."""

    intent: str
    secondary_intents: list[str] = Field(default_factory=list)
    confidence: Confidence
    reasoning: str = ""


class ClassificationResult(BaseModel):
    """A classification plus the metadata needed for logging and evaluation."""

    classification: Classification
    prompt_version: str
    model: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


class Action(StrEnum):
    AUTO = "auto"
    HUMAN = "human"


class EscalationReason(StrEnum):
    CLASSIFIER_ERROR = "classifier_error"
    OUT_OF_SCOPE = "out_of_scope"
    LOW_CONFIDENCE = "low_confidence"
    HIGH_RISK = "high_risk"
    HANDLER_ERROR = "handler_error"


class RoutingDecision(BaseModel):
    """What the policy decided: handle automatically, or escalate to a human and why."""

    action: Action
    intent: str
    handler: str
    reasons: list[EscalationReason] = Field(default_factory=list)
