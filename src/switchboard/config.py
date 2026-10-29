"""Router configuration: the only place domain knowledge lives.

The engine reads intents, descriptions, examples, handler names, risk levels and
policy thresholds from a YAML file validated by these models. Swapping the file
swaps the domain.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

OUT_OF_SCOPE = "out_of_scope"

_NAME_PATTERN = r"^[a-z][a-z0-9_]*$"


class Risk(StrEnum):
    LOW = "low"
    HIGH = "high"


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return list(Confidence).index(self)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IntentConfig(_Strict):
    name: str = Field(pattern=_NAME_PATTERN)
    description: str = Field(min_length=1)
    examples: list[str] = Field(default_factory=list)
    handler: str = Field(pattern=_NAME_PATTERN)
    risk: Risk = Risk.LOW

    @field_validator("description")
    @classmethod
    def _description_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("description must not be blank")
        return v.strip()


class OutOfScopeConfig(_Strict):
    description: str = Field(min_length=1)
    handler: str = Field(pattern=_NAME_PATTERN)


class PolicyConfig(_Strict):
    min_confidence: Confidence = Confidence.MEDIUM
    human_review_risks: list[Risk] = Field(default_factory=lambda: [Risk.HIGH])


class Effort(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class PricingConfig(_Strict):
    """USD per million tokens, used for cost estimates in eval reports."""

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


class ModelConfig(_Strict):
    """LLM call settings. Each field can be overridden by a SWITCHBOARD_* env variable."""

    name: str = "claude-haiku-5-5"
    max_tokens: int = Field(default=300, gt=0)
    effort: Effort | None = Effort.LOW
    thinking: bool = False
    # Current API versions reject sampling parameters on every model. Temperature
    # is only sent when `api_version` pins an older version that still accepts it.
    temperature: float | None = Field(default=None, ge=0, le=1)
    api_version: str | None = None
    timeout_s: float = Field(default=10.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    pricing: PricingConfig | None = None


class RouterConfig(_Strict):
    domain: str = Field(min_length=1)
    version: str = Field(min_length=1)
    intents: list[IntentConfig] = Field(min_length=1)
    out_of_scope: OutOfScopeConfig
    # Python module whose @handler functions implement this domain's handler names.
    handlers_module: str | None = None
    policy: PolicyConfig = PolicyConfig()
    model: ModelConfig = ModelConfig()

    @model_validator(mode="after")
    def _check_intent_names(self) -> RouterConfig:
        names = [i.name for i in self.intents]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate intent names: {dupes}")
        if OUT_OF_SCOPE in names:
            raise ValueError(f"'{OUT_OF_SCOPE}' is reserved and must not be declared as an intent")
        return self

    @property
    def intent_names(self) -> list[str]:
        """All labels the classifier may emit, including out_of_scope."""
        return [i.name for i in self.intents] + [OUT_OF_SCOPE]

    def get_intent(self, name: str) -> IntentConfig:
        for intent in self.intents:
            if intent.name == name:
                return intent
        raise KeyError(name)


def load_config(path: str | Path) -> RouterConfig:
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return RouterConfig.model_validate(data)
