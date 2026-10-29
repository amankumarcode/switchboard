from pathlib import Path

import pytest

from switchboard.config import RouterConfig, load_config

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def banking_config() -> RouterConfig:
    return load_config(ROOT / "configs" / "banking.yaml")
