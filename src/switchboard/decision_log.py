"""Decision log: one JSON line per routed request, for auditing and monitoring."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

DEFAULT_PATH = "logs/decisions.jsonl"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


class DecisionLog:
    """Appends records to a JSONL file. The raw query is dropped unless `log_queries`
    is on, since customer messages can contain personal data."""

    def __init__(self, path: str | Path | None = None, *, log_queries: bool | None = None):
        self.path = Path(path or os.environ.get("SWITCHBOARD_DECISION_LOG_PATH") or DEFAULT_PATH)
        self.log_queries = (
            _env_flag("SWITCHBOARD_LOG_QUERIES") if log_queries is None else log_queries
        )
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        if not self.log_queries:
            record = {k: v for k, v in record.items() if k != "query"}
        line = json.dumps(record, ensure_ascii=False, default=str)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
