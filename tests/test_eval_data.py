import json
from collections import Counter
from pathlib import Path

import pytest

from switchboard.config import OUT_OF_SCOPE, load_config

ROOT = Path(__file__).parent.parent
EVAL_FILE = ROOT / "data" / "eval" / "banking77_eval.jsonl"


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    with EVAL_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_rows_have_expected_fields(rows):
    assert rows
    for row in rows:
        assert set(row) == {"id", "text", "label", "source_label"}
        assert row["text"].strip()


def test_labels_match_config(rows):
    config = load_config(ROOT / "configs" / "banking.yaml")
    labels = Counter(r["label"] for r in rows)
    assert set(labels) == set(config.intent_names)


def test_no_duplicate_ids_or_texts(rows):
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({r["text"].lower() for r in rows}) == len(rows)


def test_in_scope_rows_keep_their_source_label(rows):
    for row in rows:
        if row["label"] != OUT_OF_SCOPE:
            assert row["label"] == row["source_label"]


def test_few_shot_examples_not_in_eval_set(rows):
    config = load_config(ROOT / "configs" / "banking.yaml")
    eval_texts = {r["text"].lower() for r in rows}
    for intent in config.intents:
        for example in intent.examples:
            assert example.lower() not in eval_texts, (intent.name, example)
