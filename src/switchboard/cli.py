"""Command-line interface.

    switchboard route "my card hasn't arrived" [--customer cust_001] [--json]
    switchboard eval [--limit N] [--workers N] ...

The provider comes from SWITCHBOARD_PROVIDER (default: mock). To use a .env file:
    uv run --env-file .env switchboard route "..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from switchboard.router import Router, RouteResult


def _format(result: RouteResult) -> str:
    reasons = ", ".join(r.value for r in result.reasons) or "-"
    secondary = ", ".join(result.secondary_intents) or "-"
    return "\n".join(
        [
            f"intent      {result.intent} ({result.confidence.value})  secondary: {secondary}",
            f"reasoning   {result.reasoning}",
            f"decision    {result.action.value} -> {result.handler}  reasons: {reasons}",
            f"status      {result.status.value}",
            f"reply       {result.message}",
            f"trace       {result.trace_id}  model={result.model} "
            f"prompt={result.prompt_version} {result.latency_ms:.0f} ms",
        ]
    )


def _route(args: argparse.Namespace) -> int:
    router = Router.from_env(args.config)
    try:
        result = router.route(args.query, customer_id=args.customer)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(result.model_dump_json(indent=2) if args.json else _format(result))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="switchboard", description="LLM intent router")
    sub = parser.add_subparsers(dest="command", required=True)

    route = sub.add_parser("route", help="classify and route one query")
    route.add_argument("query")
    route.add_argument("--customer", help="customer id passed to handlers")
    route.add_argument("--config", type=Path, help="router config (default: banking)")
    route.add_argument("--json", action="store_true", help="print the full result as JSON")

    sub.add_parser("eval", help="run the eval harness (see `switchboard eval -h`)", add_help=False)

    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["eval"]:
        from switchboard.eval.__main__ import main as eval_main

        return eval_main(argv[1:])

    args = parser.parse_args(argv)
    return _route(args)


if __name__ == "__main__":
    raise SystemExit(main())
