import json

import pytest

from switchboard.config import OUT_OF_SCOPE, Confidence
from switchboard.dispatcher import Dispatcher
from switchboard.handlers import HandlerContext, HandlerStatus, get_handler, handler, load_handlers
from switchboard.handlers.banking import reset_accounts
from switchboard.models import Action, Classification, EscalationReason, RoutingDecision


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("SWITCHBOARD_HUMAN_QUEUE_PATH", str(tmp_path / "queue.jsonl"))
    reset_accounts()
    yield
    reset_accounts()


@pytest.fixture
def queue_file(tmp_path):
    return tmp_path / "queue.jsonl"


def _ctx(intent, customer_id="cust_001", reasons=(), confidence="high"):
    action = Action.HUMAN if reasons else Action.AUTO
    return HandlerContext(
        query="test query",
        trace_id="trace-1",
        customer_id=customer_id,
        classification=Classification(intent=intent, confidence=confidence),
        decision=RoutingDecision(action=action, intent=intent, handler="x", reasons=list(reasons)),
    )


def _run(name, intent=None, **kwargs):
    return get_handler(name)(_ctx(intent or name, **kwargs))


# --- registry -------------------------------------------------------------------


def test_load_handlers_registers_every_configured_name(banking_config):
    load_handlers(banking_config)
    for intent in banking_config.intents:
        assert callable(get_handler(intent.handler))
    assert callable(get_handler("human_queue"))


def test_missing_handler_fails_fast(banking_config):
    intents = [
        banking_config.intents[0].model_copy(update={"handler": "does_not_exist"}),
        *banking_config.intents[1:],
    ]
    bad = banking_config.model_copy(update={"intents": intents})
    with pytest.raises(ValueError, match="does_not_exist"):
        load_handlers(bad)


def test_unknown_handlers_module_fails(banking_config):
    bad = banking_config.model_copy(update={"handlers_module": "switchboard.handlers.nope"})
    with pytest.raises(ModuleNotFoundError):
        load_handlers(bad)


def test_duplicate_registration_rejected():
    with pytest.raises(ValueError, match="already registered"):

        @handler("human_queue")
        def impostor(ctx):  # pragma: no cover
            raise AssertionError


def test_get_handler_unknown():
    with pytest.raises(KeyError, match="no handler registered"):
        get_handler("nope")


# --- banking mock handlers -----------------------------------------------------------


@pytest.fixture(autouse=True)
def _load(banking_config):
    load_handlers(banking_config)


def test_card_delivery_status():
    result = _run("card_delivery_status", "card_arrival")
    assert result.status is HandlerStatus.OK
    assert "2026-10-30" in result.message
    assert _run("card_delivery_status", customer_id="cust_002").status is HandlerStatus.NOT_FOUND


def test_card_activation_changes_state():
    first = _run("card_activation")
    assert first.status is HandlerStatus.OK and "7730" in first.message
    assert _run("card_activation").status is HandlerStatus.NOT_FOUND  # now all active


def test_card_block_blocks_all_active_cards():
    result = _run("card_block")
    assert result.data["blocked"] == ["card_001a"]
    assert _run("card_block").status is HandlerStatus.NOT_FOUND


def test_transfer_cancellation_only_cancels_pending():
    result = _run("transfer_cancellation")
    assert result.status is HandlerStatus.OK
    assert result.data["transfer"]["status"] == "cancelled"
    assert _run("transfer_cancellation").status is HandlerStatus.NOT_FOUND
    assert _run("transfer_cancellation", customer_id="cust_002").status is HandlerStatus.NOT_FOUND


def test_refund_fee_and_pin_handlers():
    assert "ShoeHub" in _run("refund_status").message
    assert "foreign currency" in _run("fee_explanation").message
    assert _run("pin_change").status is HandlerStatus.OK
    assert _run("refund_status", customer_id="cust_002").status is HandlerStatus.NOT_FOUND


def test_reset_restores_fixture():
    _run("card_block")
    reset_accounts()
    assert _run("card_block").status is HandlerStatus.OK


def test_unknown_customer_raises():
    with pytest.raises(LookupError):
        _run("pin_change", customer_id="cust_999")


# --- human queue ---------------------------------------------------------------------


def test_human_queue_appends_jsonl(queue_file):
    reasons = [EscalationReason.HIGH_RISK]
    result = _run("human_queue", "lost_or_stolen_card", reasons=reasons, confidence="high")
    _run("human_queue", OUT_OF_SCOPE, reasons=[EscalationReason.OUT_OF_SCOPE])

    assert result.status is HandlerStatus.QUEUED
    entries = [json.loads(line) for line in queue_file.read_text().splitlines()]
    assert len(entries) == 2
    first = entries[0]
    assert first["ticket_id"] == result.data["ticket_id"]
    assert first["intent"] == "lost_or_stolen_card"
    assert first["reasons"] == ["high_risk"]
    assert first["query"] == "test query" and first["trace_id"] == "trace-1"


# --- dispatcher ----------------------------------------------------------------------


def _dispatch(banking_config, decision, customer_id="cust_001"):
    return Dispatcher(banking_config).dispatch(
        decision,
        query="q",
        classification=Classification(intent=decision.intent, confidence=Confidence.HIGH),
        trace_id="t",
        customer_id=customer_id,
    )


def test_dispatch_auto(banking_config, queue_file):
    decision = RoutingDecision(action=Action.AUTO, intent="change_pin", handler="pin_change")
    outcome = _dispatch(banking_config, decision)
    assert outcome.decision == decision
    assert outcome.result.status is HandlerStatus.OK
    assert outcome.handler_error is None
    assert not queue_file.exists()


def test_dispatch_human(banking_config, queue_file):
    decision = RoutingDecision(
        action=Action.HUMAN,
        intent="cancel_transfer",
        handler="human_queue",
        reasons=[EscalationReason.HIGH_RISK],
    )
    outcome = _dispatch(banking_config, decision)
    assert outcome.result.status is HandlerStatus.QUEUED
    assert len(queue_file.read_text().splitlines()) == 1


def test_failing_handler_escalates_to_human(banking_config, queue_file):
    decision = RoutingDecision(action=Action.AUTO, intent="change_pin", handler="pin_change")
    outcome = _dispatch(banking_config, decision, customer_id="cust_999")

    assert outcome.decision.action is Action.HUMAN
    assert outcome.decision.handler == "human_queue"
    assert outcome.decision.reasons == [EscalationReason.HANDLER_ERROR]
    assert "LookupError" in outcome.handler_error
    assert outcome.result.status is HandlerStatus.QUEUED
    entry = json.loads(queue_file.read_text())
    assert entry["reasons"] == ["handler_error"]
