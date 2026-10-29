"""Metrics over eval predictions: quality, calibration, latency and cost."""

from __future__ import annotations

import statistics
from collections import Counter

from pydantic import BaseModel
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support as prfs

from switchboard.config import OUT_OF_SCOPE, Confidence, PricingConfig
from switchboard.eval.harness import Prediction


class IntentMetrics(BaseModel):
    intent: str
    precision: float
    recall: float
    f1: float
    support: int


class BucketMetrics(BaseModel):
    confidence: Confidence
    count: int
    share: float
    accuracy: float | None


class EvalMetrics(BaseModel):
    n: int
    accuracy: float
    macro_f1: float
    per_intent: list[IntentMetrics]
    labels: list[str]
    confusion: list[list[int]]  # rows = true label, columns = predicted, in `labels` order
    by_confidence: list[BucketMetrics]
    false_accepts: int  # out_of_scope predicted as an in-scope intent
    false_rejects: int  # in-scope predicted as out_of_scope
    false_accept_sources: list[tuple[str, int]]  # Banking77 labels most often wrongly accepted
    errors: int
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    avg_input_tokens: float
    avg_output_tokens: float
    cost_per_1k_usd: float | None


def _percentile(values: list[float], pct: float) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(pct) - 1]


def compute_metrics(
    predictions: list[Prediction], labels: list[str], pricing: PricingConfig | None = None
) -> EvalMetrics:
    if not predictions:
        raise ValueError("no predictions to score")
    y_true = [p.label for p in predictions]
    y_pred = [p.predicted for p in predictions]
    n = len(predictions)

    precision, recall, f1, support = prfs(y_true, y_pred, labels=labels, zero_division=0)
    per_intent = [
        IntentMetrics(
            intent=label,
            precision=float(precision[i]),
            recall=float(recall[i]),
            f1=float(f1[i]),
            support=int(support[i]),
        )
        for i, label in enumerate(labels)
    ]

    by_confidence = []
    for level in Confidence:
        bucket = [p for p in predictions if p.confidence is level]
        by_confidence.append(
            BucketMetrics(
                confidence=level,
                count=len(bucket),
                share=len(bucket) / n,
                accuracy=sum(p.correct for p in bucket) / len(bucket) if bucket else None,
            )
        )

    accepted_oos = [
        p for p in predictions if p.label == OUT_OF_SCOPE and p.predicted != OUT_OF_SCOPE
    ]
    false_rejects = sum(
        p.label != OUT_OF_SCOPE and p.predicted == OUT_OF_SCOPE for p in predictions
    )

    latencies = [p.latency_ms for p in predictions]
    avg_in = statistics.fmean(p.input_tokens for p in predictions)
    avg_out = statistics.fmean(p.output_tokens for p in predictions)
    cost = (
        1000 * (avg_in * pricing.input_per_mtok + avg_out * pricing.output_per_mtok) / 1_000_000
        if pricing
        else None
    )

    return EvalMetrics(
        n=n,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        per_intent=per_intent,
        labels=labels,
        confusion=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        by_confidence=by_confidence,
        false_accepts=len(accepted_oos),
        false_rejects=false_rejects,
        false_accept_sources=Counter(p.source_label for p in accepted_oos).most_common(10),
        errors=sum(p.error is not None for p in predictions),
        latency_mean_ms=statistics.fmean(latencies),
        latency_p50_ms=statistics.median(latencies),
        latency_p95_ms=_percentile(latencies, 95),
        avg_input_tokens=avg_in,
        avg_output_tokens=avg_out,
        cost_per_1k_usd=cost,
    )
