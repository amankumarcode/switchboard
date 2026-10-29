"""Manual smoke test against the real Claude API (costs a fraction of a cent).

Usage:
    ANTHROPIC_API_KEY=... uv run python scripts/smoke_anthropic.py [--config configs/banking.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from switchboard.classifier import Classifier
from switchboard.config import load_config
from switchboard.providers.anthropic import AnthropicProvider
from switchboard.providers.factory import model_settings

ROOT = Path(__file__).resolve().parent.parent

QUERIES = [
    "My new card still hasn't shown up, it's been ten days.",
    "I think someone stole my card at the train station",
    "Can I change my PIN at an ATM?",
    "The shop refunded me a week ago but my balance hasn't changed",
    "What's the exchange rate for euros today?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=ROOT / "configs" / "banking.yaml", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    settings = model_settings(config)
    classifier = Classifier(config, AnthropicProvider(settings))
    print(f"model={settings.name} prompt_version={classifier.prompt_version}\n")

    for query in QUERIES:
        r = classifier.classify(query)
        c = r.classification
        print(f"{query}\n  -> {c.intent} ({c.confidence.value}) secondary={c.secondary_intents}")
        print(f"     {c.reasoning}")
        print(f"     {r.latency_ms:.0f} ms, {r.input_tokens} in / {r.output_tokens} out tokens")
        if r.error:
            print(f"     ERROR: {r.error}")
        print()


if __name__ == "__main__":
    main()
