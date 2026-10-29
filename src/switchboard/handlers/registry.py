"""Handler registry: config names a handler, this maps the name to a function."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from switchboard.config import RouterConfig
from switchboard.models import Classification, RoutingDecision

# Modules shipped with the engine that every config may use (e.g. human_queue).
_BUILTIN_MODULES = ["switchboard.handlers.human_queue"]


class HandlerStatus(StrEnum):
    OK = "ok"  # the workflow ran
    NOT_FOUND = "not_found"  # ran, but there was nothing to act on (no order, no transfer...)
    QUEUED = "queued"  # handed to a human


class HandlerContext(BaseModel):
    query: str
    trace_id: str
    customer_id: str
    classification: Classification
    decision: RoutingDecision


class HandlerResult(BaseModel):
    status: HandlerStatus
    message: str  # customer-facing reply
    data: dict[str, Any] = Field(default_factory=dict)


Handler = Callable[[HandlerContext], HandlerResult]

_REGISTRY: dict[str, Handler] = {}


def handler(name: str) -> Callable[[Handler], Handler]:
    """Register the decorated function under `name`."""

    def register(fn: Handler) -> Handler:
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not fn:
            raise ValueError(f"handler {name!r} is already registered by {existing.__module__}")
        _REGISTRY[name] = fn
        return fn

    return register


def get_handler(name: str) -> Handler:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no handler registered under {name!r}") from None


def load_handlers(config: RouterConfig) -> None:
    """Import built-in and domain handler modules, then check every configured name exists."""
    for module in [*_BUILTIN_MODULES, config.handlers_module]:
        if module:
            importlib.import_module(module)
    names = {i.handler for i in config.intents} | {config.out_of_scope.handler}
    missing = sorted(names - _REGISTRY.keys())
    if missing:
        raise ValueError(
            f"config names handlers with no implementation: {missing} "
            f"(handlers_module={config.handlers_module!r})"
        )
