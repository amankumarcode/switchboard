"""Run the classifier over a labeled eval set and collect per-row predictions."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

from switchboard.classifier import Classifier
from switchboard.config import Confidence


class EvalRow(BaseModel):
    id: str
    text: str
    label: str
    source_label: str


class Prediction(BaseModel):
    id: str
    text: str
    label: str
    source_label: str
    predicted: str
    confidence: Confidence
    reasoning: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    error: str | None = None

    @property
    def correct(self) -> bool:
        return self.predicted == self.label


def load_eval_set(path: str | Path, limit: int | None = None) -> list[EvalRow]:
    with Path(path).open(encoding="utf-8") as f:
        rows = [EvalRow.model_validate_json(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def _predict(classifier: Classifier, row: EvalRow) -> Prediction:
    result = classifier.classify(row.text)
    c = result.classification
    return Prediction(
        **row.model_dump(),
        predicted=c.intent,
        confidence=c.confidence,
        reasoning=c.reasoning,
        latency_ms=result.latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        error=result.error,
    )


def run_eval(classifier: Classifier, rows: list[EvalRow], workers: int = 4) -> list[Prediction]:
    """Classify every row; order of the output matches the input."""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(lambda row: _predict(classifier, row), rows))


def write_predictions(predictions: list[Prediction], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p.model_dump(mode="json"), ensure_ascii=False) + "\n")
