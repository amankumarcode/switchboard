"""Render eval metrics as a markdown report."""

from __future__ import annotations

from datetime import UTC, datetime

from switchboard.config import ModelConfig
from switchboard.eval.harness import Prediction
from switchboard.eval.metrics import EvalMetrics


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x:.1%}"


def render_report(
    metrics: EvalMetrics,
    predictions: list[Prediction],
    *,
    model: str,
    settings: ModelConfig | None,
    prompt_version: str,
    dataset: str,
    max_errors_shown: int = 15,
) -> str:
    m = metrics
    lines = [
        f"# Eval report: `{model}` · prompt `{prompt_version}`",
        "",
        f"- Date: {datetime.now(UTC):%Y-%m-%d %H:%M} UTC",
        f"- Dataset: `{dataset}` ({m.n} rows)",
    ]
    if settings is not None:
        lines.append(
            f"- Settings: max_tokens={settings.max_tokens}, "
            f"effort={settings.effort.value if settings.effort else 'default'}, "
            f"thinking={'on' if settings.thinking else 'off'}, "
            f"temperature={settings.temperature if settings.api_version else 'not sent'}"
        )

    cost = "–" if m.cost_per_1k_usd is None else f"${m.cost_per_1k_usd:.3f}"
    lines += [
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {_pct(m.accuracy)} |",
        f"| Macro F1 | {m.macro_f1:.3f} |",
        f"| Out-of-scope accepted as in-scope | {m.false_accepts} |",
        f"| In-scope rejected as out-of-scope | {m.false_rejects} |",
        f"| Classifier errors | {m.errors} |",
        f"| Latency mean / p50 / p95 | {m.latency_mean_ms:.0f} / {m.latency_p50_ms:.0f} / "
        f"{m.latency_p95_ms:.0f} ms |",
        f"| Avg tokens in / out | {m.avg_input_tokens:.0f} / {m.avg_output_tokens:.0f} |",
        f"| Est. cost per 1,000 queries | {cost} |",
        "",
        "## Accuracy by confidence",
        "",
        "| Confidence | Count | Share | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {b.confidence.value} | {b.count} | {_pct(b.share)} | {_pct(b.accuracy)} |"
        for b in m.by_confidence
    ]

    lines += [
        "",
        "## Per intent",
        "",
        "| Intent | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {i.intent} | {i.precision:.3f} | {i.recall:.3f} | {i.f1:.3f} | {i.support} |"
        for i in m.per_intent
    ]

    idx = range(1, len(m.labels) + 1)
    lines += [
        "",
        "## Confusion matrix",
        "",
        "Rows are true labels, columns are predictions (numbered as in the row list).",
        "",
        "| # | True \\ Predicted | " + " | ".join(str(i) for i in idx) + " |",
        "|---|---|" + "---:|" * len(m.labels),
    ]
    lines += [
        f"| {i} | {label} | " + " | ".join(str(v) for v in row) + " |"
        for i, label, row in zip(idx, m.labels, m.confusion, strict=True)
    ]

    if m.false_accept_sources:
        lines += [
            "",
            "## Out-of-scope queries wrongly accepted, by Banking77 label",
            "",
            "| Banking77 label | Count |",
            "|---|---:|",
        ]
        lines += [f"| {label} | {count} |" for label, count in m.false_accept_sources]

    wrong = [p for p in predictions if not p.correct]
    if wrong:
        lines += [
            "",
            f"## Sample errors ({min(len(wrong), max_errors_shown)} of {len(wrong)})",
            "",
            "| Query | Expected | Predicted | Confidence |",
            "|---|---|---|---|",
        ]
        for p in wrong[:max_errors_shown]:
            text = p.text.replace("|", "\\|")
            lines.append(f"| {text} | {p.label} | {p.predicted} | {p.confidence.value} |")

    return "\n".join(lines) + "\n"
