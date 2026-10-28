"""Build the Banking77 eval set and pick few-shot examples for the config.

Downloads the official Banking77 CSVs (CC BY 4.0, PolyAI), then:
  * prints N train-split examples per in-scope intent as a YAML snippet to paste
    into the config (kept manual so hand-written config comments survive);
  * writes the eval set from the test split: every test query of the in-scope
    intents, plus out-of-scope queries relabeled `out_of_scope` (half near-miss
    intents that resemble ours, half random others).

Usage:
    uv run python scripts/prepare_data.py [--config configs/banking.yaml] [--seed 7]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

from switchboard.config import OUT_OF_SCOPE, load_config

BASE_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
ROOT = Path(__file__).resolve().parent.parent

# Banking77 intents that look like our in-scope ones; the hardest out-of-scope cases.
NEAR_MISS_INTENTS = [
    "card_delivery_estimate",
    "card_not_working",
    "pin_blocked",
    "transfer_not_received_by_recipient",
    "extra_charge_on_statement",
    "request_refund",
    "card_swallowed",
    "declined_card_payment",
    "get_physical_card",
    "reverted_card_payment?",
]


def fetch_split(split: str) -> dict[str, list[str]]:
    """Return {label: [texts]} for a Banking77 split, labels lowercased."""
    with urllib.request.urlopen(f"{BASE_URL}/{split}.csv", timeout=30) as resp:
        reader = csv.DictReader(io.StringIO(resp.read().decode("utf-8")))
        by_label: dict[str, list[str]] = defaultdict(list)
        for row in reader:
            by_label[row["category"].strip().lower()].append(row["text"].strip())
    return dict(by_label)


def unique(texts: list[str]) -> list[str]:
    return list(dict.fromkeys(texts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=ROOT / "configs" / "banking.yaml", type=Path)
    parser.add_argument("--out", default=ROOT / "data" / "eval" / "banking77_eval.jsonl", type=Path)
    parser.add_argument("--examples-per-intent", default=3, type=int)
    parser.add_argument("--oos-size", default=100, type=int)
    parser.add_argument("--seed", default=7, type=int)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    in_scope = [i.name for i in load_config(args.config).intents]
    train, test = fetch_split("train"), fetch_split("test")

    missing = [n for n in in_scope + NEAR_MISS_INTENTS if n not in test]
    if missing:
        raise SystemExit(f"labels not found in Banking77: {missing}")

    # Few-shot examples from the train split.
    print("# Few-shot examples (Banking77 train split) -- paste into the config:")
    for name in in_scope:
        print(f"  {name}:")
        for text in rng.sample(unique(train[name]), args.examples_per_intent):
            print(f"    - {json.dumps(text, ensure_ascii=False)}")

    # Eval set from the test split.
    rows: list[dict] = []
    for name in in_scope:
        rows += [{"text": t, "label": name, "source_label": name} for t in unique(test[name])]

    def oos_pool(labels: list[str]) -> list[tuple[str, str]]:
        return [(label, t) for label in labels for t in unique(test[label])]

    others = sorted(set(test) - set(in_scope) - set(NEAR_MISS_INTENTS))
    n_near = args.oos_size // 2
    oos = rng.sample(oos_pool(NEAR_MISS_INTENTS), n_near)
    oos += rng.sample(oos_pool(others), args.oos_size - n_near)
    rows += [{"text": t, "label": OUT_OF_SCOPE, "source_label": label} for label, t in oos]

    seen: set[str] = set()
    deduped = []
    for row in rows:
        if row["text"].lower() not in seen:
            seen.add(row["text"].lower())
            deduped.append(row)
    rng.shuffle(deduped)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(deduped):
            f.write(json.dumps({"id": f"b77-{i:04d}", **row}, ensure_ascii=False) + "\n")

    n_oos = sum(r["label"] == OUT_OF_SCOPE for r in deduped)
    print(f"\n# Wrote {len(deduped)} rows ({len(deduped) - n_oos} in-scope, {n_oos} out-of-scope)")
    print(f"# -> {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
