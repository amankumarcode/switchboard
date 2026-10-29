import json

import pytest

from switchboard.cli import main
from switchboard.config import OUT_OF_SCOPE
from switchboard.decision_log import DecisionLog
from switchboard.handlers.banking import reset_accounts
from switchboard.handlers.registry import HandlerStatus
from switchboard.models import Action, EscalationReason
from switchboard.providers import MockProvider
from switchboard.router import Router

LOG_FIELDS = {
    "trace_id",
    "timestamp",
    "prompt_version",
    "model",
    "customer_id",
    "intent",
    "secondary_intents",
    "confidence",
    "action",
    "handler",
    "reasons",
    "status",
    "classifier_latency_ms",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "classifier_error",
    "handler_error",
}


def _response(intent, confidence="high"):
    return {"reasoning": "r", "intent": intent, "secondary_intents": [], "confidence": confidence}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("SWITCHBOARD_HUMAN_QUEUE_PATH", str(tmp_path / "queue.jsonl"))
    monkeypatch.setenv("SWITCHBOARD_DECISION_LOG_PATH", str(tmp_path / "decisions.jsonl"))
    monkeypatch.delenv("SWITCHBOARD_LOG_QUERIES", raising=False)
    monkeypatch.setenv("SWITCHBOARD_PROVIDER", "mock")
    reset_accounts()
    yield
    reset_accounts()


def _log_lines(tmp_path):
    return [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]


def _router(config, responses, **kwargs):
    return Router(
        config, MockProvider(responses=responses), default_customer_id="cust_001", **kwargs
    )


def test_auto_route_runs_handler_and_logs(banking_config, tmp_path):
    result = _router(banking_config, [_response("card_arrival")]).route("Where is my card?")

    assert (result.action, result.handler) == (Action.AUTO, "card_delivery_status")
    assert result.status is HandlerStatus.OK
    assert "2026-10-30" in result.message
    assert result.reasons == []

    (entry,) = _log_lines(tmp_path)
    assert set(entry) == LOG_FIELDS
    assert entry["trace_id"] == result.trace_id
    assert entry["action"] == "auto" and entry["intent"] == "card_arrival"
    assert "query" not in entry
    assert not (tmp_path / "queue.jsonl").exists()


def test_high_risk_goes_to_human_queue(banking_config, tmp_path):
    result = _router(banking_config, [_response("lost_or_stolen_card")]).route(
        "I think someone stole my card"
    )
    assert result.action is Action.HUMAN
    assert result.reasons == [EscalationReason.HIGH_RISK]
    assert result.status is HandlerStatus.QUEUED

    ticket = json.loads((tmp_path / "queue.jsonl").read_text())
    assert ticket["trace_id"] == result.trace_id
    assert ticket["query"] == "I think someone stole my card"  # humans need the text


def test_out_of_scope_and_low_confidence(banking_config):
    router = _router(banking_config, [_response(OUT_OF_SCOPE), _response("change_pin", "low")])
    assert router.route("weather?").reasons == [EscalationReason.OUT_OF_SCOPE]
    assert router.route("pin?").reasons == [EscalationReason.LOW_CONFIDENCE]


def test_classifier_error_is_logged_and_escalated(banking_config, tmp_path):
    result = _router(banking_config, [TimeoutError("slow")]).route("hello")
    assert result.reasons == [EscalationReason.CLASSIFIER_ERROR]
    assert "TimeoutError" in _log_lines(tmp_path)[0]["classifier_error"]


def test_handler_error_is_logged_and_escalated(banking_config, tmp_path):
    router = _router(banking_config, [_response("change_pin")])
    result = router.route("change my pin", customer_id="cust_999")
    assert result.action is Action.HUMAN
    assert result.reasons == [EscalationReason.HANDLER_ERROR]
    entry = _log_lines(tmp_path)[0]
    assert entry["customer_id"] == "cust_999"
    assert "LookupError" in entry["handler_error"]


def test_log_queries_opt_in(banking_config, tmp_path):
    log = DecisionLog(tmp_path / "decisions.jsonl", log_queries=True)
    _router(banking_config, [_response("change_pin")], decision_log=log).route("change my pin")
    assert _log_lines(tmp_path)[0]["query"] == "change my pin"


@pytest.mark.parametrize("query", ["", "   ", "x" * 2001])
def test_invalid_queries_rejected(banking_config, query):
    with pytest.raises(ValueError):
        _router(banking_config, []).route(query)


def test_cli_route_json(capsys):
    assert main(["route", "I want to change my PIN", "--customer", "cust_001", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["intent"] == "change_pin"
    assert out["action"] == "auto"


def test_cli_route_text(capsys):
    assert main(["route", "My card was stolen", "--customer", "cust_001"]) == 0
    out = capsys.readouterr().out
    assert "human -> human_queue" in out
    assert "high_risk" in out


def test_cli_route_empty_query(capsys):
    assert main(["route", "  "]) == 2
    assert "must not be empty" in capsys.readouterr().err


def test_cli_eval_delegates(tmp_path):
    assert main(["eval", "--limit", "5", "--out-dir", str(tmp_path)]) == 0
    assert list(tmp_path.glob("eval-mock-*.md"))
