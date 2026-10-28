from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from switchboard.config import OUT_OF_SCOPE, Confidence, Risk, RouterConfig, load_config

BANKING_CONFIG = Path(__file__).parent.parent / "configs" / "banking.yaml"


def _minimal(**overrides) -> dict:
    data = {
        "domain": "test",
        "version": "1",
        "intents": [
            {"name": "a", "description": "Intent A", "handler": "handle_a"},
            {"name": "b", "description": "Intent B", "handler": "handle_b", "risk": "high"},
        ],
        "out_of_scope": {"description": "Anything else", "handler": "human_queue"},
    }
    data.update(overrides)
    return data


def test_banking_config_loads():
    config = load_config(BANKING_CONFIG)
    assert len(config.intents) == 8
    assert config.intent_names[-1] == OUT_OF_SCOPE
    assert config.get_intent("lost_or_stolen_card").risk is Risk.HIGH
    assert config.policy.min_confidence is Confidence.MEDIUM


def test_minimal_config_defaults():
    config = RouterConfig.model_validate(_minimal())
    assert config.intent_names == ["a", "b", OUT_OF_SCOPE]
    assert config.get_intent("a").risk is Risk.LOW
    assert config.policy.human_review_risks == [Risk.HIGH]
    assert config.pricing is None


def test_confidence_rank_orders_levels():
    assert Confidence.LOW.rank < Confidence.MEDIUM.rank < Confidence.HIGH.rank


@pytest.mark.parametrize(
    "intents",
    [
        pytest.param(
            [
                {"name": "a", "description": "x", "handler": "h"},
                {"name": "a", "description": "y", "handler": "h"},
            ],
            id="duplicate-name",
        ),
        pytest.param(
            [{"name": OUT_OF_SCOPE, "description": "x", "handler": "h"}], id="reserved-name"
        ),
        pytest.param(
            [{"name": "a", "description": "x", "handler": "h", "risk": "medium"}], id="bad-risk"
        ),
        pytest.param([{"name": "a", "description": "   ", "handler": "h"}], id="blank-description"),
        pytest.param([{"name": "Bad Name", "description": "x", "handler": "h"}], id="bad-name"),
        pytest.param(
            [{"name": "a", "description": "x", "handler": "h", "typo": 1}], id="unknown-key"
        ),
        pytest.param([], id="no-intents"),
    ],
)
def test_invalid_configs_rejected(intents):
    with pytest.raises(ValidationError):
        RouterConfig.model_validate(_minimal(intents=intents))


def test_load_config_from_file(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(_minimal()), encoding="utf-8")
    assert load_config(path).domain == "test"
