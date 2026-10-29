import json

import pytest

from switchboard.classifier import Classifier
from switchboard.config import OUT_OF_SCOPE, Confidence, PricingConfig
from switchboard.eval.__main__ import main
from switchboard.eval.harness import Prediction, load_eval_set, run_eval
from switchboard.eval.metrics import compute_metrics, compute_routing
from switchboard.eval.report import render_report
from switchboard.providers import MockProvider

LABELS = ["a", "b", OUT_OF_SCOPE]


def _pred(label, predicted, confidence="high", source=None, latency=100.0, error=None):
    return Prediction(
        id="x",
        text=f"{label}->{predicted}",
        label=label,
        source_label=source or label,
        predicted=predicted,
        confidence=confidence,
        reasoning="",
        latency_ms=latency,
        input_tokens=1000,
        output_tokens=100,
        error=error,
    )


@pytest.fixture
def predictions():
    return [
        _pred("a", "a", latency=100),
        _pred("a", "b", "low", latency=200),
        _pred("b", "b", latency=300),
        _pred("b", OUT_OF_SCOPE, "medium", latency=400),
        _pred(OUT_OF_SCOPE, OUT_OF_SCOPE, latency=500),
        _pred(OUT_OF_SCOPE, "a", "low", source="pin_blocked", latency=600, error="boom"),
    ]


def test_quality_metrics(predictions):
    m = compute_metrics(predictions, LABELS)
    assert m.n == 6
    assert m.accuracy == pytest.approx(3 / 6)
    # per-class F1: a = 2*(1/2*1/2)/(1) = 0.5, b = 0.5, oos = 0.5
    assert m.macro_f1 == pytest.approx(0.5)
    a = m.per_intent[0]
    assert (a.intent, a.precision, a.recall, a.support) == ("a", 0.5, 0.5, 2)
    assert m.confusion == [[1, 1, 0], [0, 1, 1], [1, 0, 1]]


def test_confidence_buckets(predictions):
    buckets = {b.confidence: b for b in compute_metrics(predictions, LABELS).by_confidence}
    assert (buckets[Confidence.HIGH].count, buckets[Confidence.HIGH].accuracy) == (3, 1.0)
    assert (buckets[Confidence.MEDIUM].count, buckets[Confidence.MEDIUM].accuracy) == (1, 0.0)
    assert (buckets[Confidence.LOW].count, buckets[Confidence.LOW].accuracy) == (2, 0.0)
    assert buckets[Confidence.HIGH].share == pytest.approx(0.5)


def test_scope_errors_latency_and_cost(predictions):
    m = compute_metrics(predictions, LABELS, PricingConfig(input_per_mtok=1, output_per_mtok=5))
    assert (m.false_accepts, m.false_rejects) == (1, 1)
    assert m.false_accept_sources == [("pin_blocked", 1)]
    assert m.errors == 1
    assert m.latency_mean_ms == pytest.approx(350)
    assert m.latency_p50_ms == pytest.approx(350)
    assert m.latency_p95_ms == pytest.approx(575)
    # 1000 queries * (1000 in * $1/M + 100 out * $5/M) = $1.50
    assert m.cost_per_1k_usd == pytest.approx(1.5)


def test_cost_is_none_without_pricing(predictions):
    assert compute_metrics(predictions, LABELS).cost_per_1k_usd is None


def test_empty_predictions_rejected():
    with pytest.raises(ValueError):
        compute_metrics([], LABELS)


def test_routing_metrics(banking_config):
    preds = [
        _pred("change_pin", "change_pin", "high"),  # auto, correct
        _pred(OUT_OF_SCOPE, "change_pin", "high", source="pin_blocked"),  # auto, wrong
        _pred("change_pin", "change_pin", "medium"),  # human: low_confidence
        _pred("lost_or_stolen_card", "lost_or_stolen_card", "high"),  # human: high_risk
        _pred(OUT_OF_SCOPE, OUT_OF_SCOPE, "low", error="boom"),  # human: classifier_error
    ]
    r = compute_routing(preds, banking_config)
    assert (r.auto, r.auto_wrong, r.human) == (2, 1, 3)
    assert r.auto_share == pytest.approx(0.4)
    assert r.auto_error_rate == pytest.approx(0.5)
    assert dict(r.reasons) == {"low_confidence": 1, "high_risk": 1, "classifier_error": 1}


def test_report_contains_key_sections(predictions, banking_config):
    m = compute_metrics(predictions, LABELS)
    routing = compute_routing([_pred("change_pin", "change_pin")], banking_config)
    report = render_report(
        m,
        predictions,
        model="mock",
        settings=None,
        prompt_version="abc123",
        dataset="d.jsonl",
        routing=routing,
    )
    for heading in [
        "## Summary",
        "## Accuracy by confidence",
        "## Routing outcome (min_confidence = high)",
        "## Per intent",
        "## Confusion",
    ]:
        assert heading in report
    assert "| Accuracy | 50.0% |" in report
    assert "pin_blocked" in report
    assert "## Sample errors (3 of 3)" in report


def test_run_eval_preserves_order(banking_config, tmp_path):
    path = tmp_path / "eval.jsonl"
    rows = [
        {"id": "1", "text": "I want to change my PIN", "label": "change_pin"},
        {"id": "2", "text": "What's the weather in Paris?", "label": OUT_OF_SCOPE},
        {"id": "3", "text": "My card was stolen", "label": "lost_or_stolen_card"},
    ]
    path.write_text(
        "".join(json.dumps(r | {"source_label": r["label"]}) + "\n" for r in rows),
        encoding="utf-8",
    )
    loaded = load_eval_set(path)
    assert len(load_eval_set(path, limit=2)) == 2

    preds = run_eval(Classifier(banking_config, MockProvider(banking_config)), loaded, workers=3)
    assert [p.id for p in preds] == ["1", "2", "3"]
    assert all(p.correct for p in preds)


def test_cli_writes_report(tmp_path, monkeypatch):
    monkeypatch.setenv("SWITCHBOARD_PROVIDER", "mock")
    assert main(["--limit", "20", "--out-dir", str(tmp_path)]) == 0
    reports = list(tmp_path.glob("eval-mock-*.md"))
    assert len(reports) == 1
    assert "## Summary" in reports[0].read_text(encoding="utf-8")
    assert list(tmp_path.glob("eval-mock-*.predictions.jsonl"))
