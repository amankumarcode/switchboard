"""Mock retail-banking workflows backed by a JSON fixture.

Each function stands in for a call to a real backend (card platform, payments,
ticketing). State changes live in memory for the life of the process.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any

from switchboard.handlers.registry import HandlerContext, HandlerResult, HandlerStatus, handler

_DEFAULT_FIXTURE = Path(__file__).resolve().parents[3] / "data" / "fixtures" / "accounts.json"
_lock = threading.Lock()
_state: dict[str, Any] | None = None


def _accounts() -> dict[str, Any]:
    global _state
    with _lock:
        if _state is None:
            path = Path(os.environ.get("SWITCHBOARD_ACCOUNTS_PATH") or _DEFAULT_FIXTURE)
            _state = json.loads(path.read_text(encoding="utf-8"))["customers"]
        return _state


def reset_accounts() -> None:
    """Forget in-memory changes; the next call reloads the fixture."""
    global _state
    with _lock:
        _state = None


def _customer(ctx: HandlerContext) -> dict[str, Any]:
    try:
        return _accounts()[ctx.customer_id]
    except KeyError:
        # Raising lets the dispatcher escalate this to a human as a handler error.
        raise LookupError(f"unknown customer {ctx.customer_id!r}") from None


def _ok(message: str, **data: Any) -> HandlerResult:
    return HandlerResult(status=HandlerStatus.OK, message=message, data=copy.deepcopy(data))


def _not_found(message: str, **data: Any) -> HandlerResult:
    return HandlerResult(status=HandlerStatus.NOT_FOUND, message=message, data=data)


@handler("card_delivery_status")
def card_delivery_status(ctx: HandlerContext) -> HandlerResult:
    orders = _customer(ctx)["card_orders"]
    if not orders:
        return _not_found("We couldn't find a card order on your account.")
    order = orders[-1]
    return _ok(
        f"Your card was {order['status']} via {order['carrier']} and should arrive "
        f"by {order['expected_by']}.",
        order=order,
    )


@handler("card_activation")
def card_activation(ctx: HandlerContext) -> HandlerResult:
    inactive = [c for c in _customer(ctx)["cards"] if c["status"] == "inactive"]
    if not inactive:
        return _not_found("All of your cards are already active.")
    card = inactive[0]
    with _lock:
        card["status"] = "active"
    return _ok(f"Your card ending {card['last4']} is now active.", card=card)


@handler("card_block")
def card_block(ctx: HandlerContext) -> HandlerResult:
    active = [c for c in _customer(ctx)["cards"] if c["status"] == "active"]
    if not active:
        return _not_found("You have no active cards to block.")
    with _lock:
        for card in active:
            card["status"] = "blocked"
    last4 = ", ".join(c["last4"] for c in active)
    return _ok(
        f"We've blocked your card(s) ending {last4}. A replacement can be ordered in the app.",
        blocked=[c["card_id"] for c in active],
    )


@handler("pin_change")
def pin_change(ctx: HandlerContext) -> HandlerResult:
    _customer(ctx)
    return _ok(
        "You can change your PIN in the app under Card > Security > Change PIN, "
        "or at any ATM that supports PIN services."
    )


@handler("transfer_cancellation")
def transfer_cancellation(ctx: HandlerContext) -> HandlerResult:
    pending = [t for t in _customer(ctx)["transfers"] if t["status"] == "pending"]
    if not pending:
        return _not_found(
            "We couldn't find a pending transfer. Completed transfers can't be cancelled, "
            "but we can help you request the money back."
        )
    transfer = pending[-1]
    with _lock:
        transfer["status"] = "cancelled"
    return _ok(
        f"Your transfer of {transfer['amount']:.2f} {transfer['currency']} to "
        f"{transfer['to']} has been cancelled.",
        transfer=transfer,
    )


@handler("refund_status")
def refund_status(ctx: HandlerContext) -> HandlerResult:
    pending = [r for r in _customer(ctx)["refunds"] if r["status"] == "pending"]
    if not pending:
        return _not_found(
            "We don't see a pending refund yet. Merchants can take a few days to send it; "
            "if it's been over 5 business days, ask the merchant for a refund reference."
        )
    lines = [f"{r['amount']:.2f} {r['currency']} from {r['merchant']}" for r in pending]
    return _ok(
        "These refunds are on their way and usually appear within 5 business days of issue: "
        + "; ".join(lines)
        + ".",
        refunds=pending,
    )


@handler("fee_explanation")
def fee_explanation(ctx: HandlerContext) -> HandlerResult:
    fees = _customer(ctx)["fees"]
    if not fees:
        return _not_found("We don't see any card payment fees on your recent transactions.")
    fee = fees[-1]
    return _ok(
        f"The {fee['amount']:.2f} {fee['currency']} fee on {fee['date']} at {fee['merchant']} "
        f"was a {fee['reason']}.",
        fees=fees,
    )
