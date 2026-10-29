"""Run the eval: `uv run python -m switchboard.eval [--limit N] [--workers N]`.

The provider comes from SWITCHBOARD_PROVIDER (default: mock).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from switchboard.classifier import Classifier
from switchboard.config import load_config
from switchboard.eval.harness import load_eval_set, run_eval, write_predictions
from switchboard.eval.metrics import compute_metrics
from switchboard.eval.report import render_report
from switchboard.providers.anthropic import AnthropicProvider
from switchboard.providers.factory import get_provider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m switchboard.eval", description=__doc__)
    parser.add_argument("--config", default="configs/banking.yaml", type=Path)
    parser.add_argument("--data", default="data/eval/banking77_eval.jsonl", type=Path)
    parser.add_argument("--limit", type=int, help="only evaluate the first N rows")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", default="reports", type=Path)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    provider = get_provider(config)
    settings = provider.settings if isinstance(provider, AnthropicProvider) else None
    classifier = Classifier(config, provider)
    model = getattr(provider, "model", "unknown")

    rows = load_eval_set(args.data, args.limit)
    print(f"Evaluating {len(rows)} rows with {model} (prompt {classifier.prompt_version})...")
    predictions = run_eval(classifier, rows, workers=args.workers)
    pricing = settings.pricing if settings else None
    metrics = compute_metrics(predictions, config.intent_names, pricing)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"eval-{re.sub(r'[^A-Za-z0-9._-]', '_', model)}-{classifier.prompt_version}"
    report_path = args.out_dir / f"{stem}.md"
    report_path.write_text(
        render_report(
            metrics,
            predictions,
            model=model,
            settings=settings,
            prompt_version=classifier.prompt_version,
            dataset=str(args.data),
        ),
        encoding="utf-8",
    )
    write_predictions(predictions, args.out_dir / f"{stem}.predictions.jsonl")

    print(
        f"accuracy={metrics.accuracy:.1%} macro_f1={metrics.macro_f1:.3f} errors={metrics.errors}"
    )
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
