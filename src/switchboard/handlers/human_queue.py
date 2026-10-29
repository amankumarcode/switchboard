"""Built-in human queue: appends escalations to a JSONL file a support team would work from."""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from switchboard.handlers.registry import HandlerContext, HandlerResult, HandlerStatus, handler

DEFAULT_PATH = "logs/human_queue.jsonl"
_lock = threading.Lock()


def queue_path() -> Path:
    return Path(os.environ.get("SWITCHBOARD_HUMAN_QUEUE_PATH") or DEFAULT_PATH)


@handler("human_queue")
def human_queue(ctx: HandlerContext) -> HandlerResult:
    ticket_id = f"hq_{uuid.uuid4().hex[:10]}"
    entry = {
        "ticket_id": ticket_id,
        "trace_id": ctx.trace_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "customer_id": ctx.customer_id,
        "query": ctx.query,
        "intent": ctx.classification.intent,
        "confidence": ctx.classification.confidence.value,
        "reasons": [r.value for r in ctx.decision.reasons],
    }
    path = queue_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return HandlerResult(
        status=HandlerStatus.QUEUED,
        message="Thanks, a specialist will review your request and get back to you shortly.",
        data={"ticket_id": ticket_id},
    )
