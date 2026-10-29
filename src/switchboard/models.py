"""Data models passed between pipeline stages."""

from __future__ import annotations

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
