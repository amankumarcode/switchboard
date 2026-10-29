"""LLM classifier: prompt in, validated Classification out, never raises."""

from __future__ import annotations

import time

from pydantic import ValidationError

from switchboard.config import OUT_OF_SCOPE, Confidence, RouterConfig
from switchboard.models import Classification, ClassificationResult
from switchboard.prompt import build_schema, build_system_prompt, build_user_message, prompt_version
from switchboard.providers.base import LLMProvider


class Classifier:
    def __init__(self, config: RouterConfig, provider: LLMProvider):
        self.provider = provider
        self.system_prompt = build_system_prompt(config)
        self.schema = build_schema(config)
        self.prompt_version = prompt_version(config)
        self._valid_intents = set(config.intent_names)

    def classify(self, query: str) -> ClassificationResult:
        start = time.perf_counter()
        model = getattr(self.provider, "model", "unknown")
        try:
            response = self.provider.complete(
                self.system_prompt, build_user_message(query), self.schema
            )
        except Exception as exc:  # fail safe: any provider failure goes to a human
            return self._fallback(start, model, f"provider_error: {type(exc).__name__}: {exc}")

        try:
            classification = self._validate(response.data)
        except (ValidationError, ValueError) as exc:
            return self._fallback(
                start,
                response.model,
                f"invalid_output: {exc}",
                response.input_tokens,
                response.output_tokens,
            )

        return ClassificationResult(
            classification=classification,
            prompt_version=self.prompt_version,
            model=response.model,
            latency_ms=_elapsed_ms(start),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    def _validate(self, data: dict) -> Classification:
        classification = Classification.model_validate(data)
        if classification.intent not in self._valid_intents:
            raise ValueError(f"unknown intent {classification.intent!r}")
        secondary = [
            s
            for s in dict.fromkeys(classification.secondary_intents)
            if s in self._valid_intents and s != classification.intent
        ]
        return classification.model_copy(update={"secondary_intents": secondary})

    def _fallback(
        self, start: float, model: str, error: str, input_tokens: int = 0, output_tokens: int = 0
    ) -> ClassificationResult:
        return ClassificationResult(
            classification=Classification(
                intent=OUT_OF_SCOPE,
                confidence=Confidence.LOW,
                reasoning=f"classifier_error: {error}",
            ),
            prompt_version=self.prompt_version,
            model=model,
            latency_ms=_elapsed_ms(start),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=error,
        )


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
