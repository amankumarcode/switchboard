"""Deterministic offline provider for tests, CI and local development.

Scores each intent by word overlap between the query and the intent's examples.
It is a stand-in, not a baseline worth reporting.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from switchboard.config import OUT_OF_SCOPE, RouterConfig
from switchboard.providers.base import ProviderResponse

_STOPWORD_TEXT = """
    a an and are am be been but can could did do does for from had has have how i i'm
    if in is it it's its me my now of on or so that the there this to was what when
    where who why will with would you your
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())
_QUERY_TAG = re.compile(r"<query>\s*(.*?)\s*</query>", re.DOTALL)


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w not in _STOPWORDS}


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class MockProvider:
    model = "mock"

    def __init__(
        self,
        config: RouterConfig | None = None,
        *,
        responses: Iterable[dict | Exception] | None = None,
        threshold: float = 0.2,
    ):
        """Score queries against `config`, or replay scripted `responses` in order."""
        if config is None and responses is None:
            raise ValueError("MockProvider needs a config or scripted responses")
        self._responses = iter(responses) if responses is not None else None
        self._threshold = threshold
        self._vocab: dict[str, list[set[str]]] = {}
        if config is not None:
            for intent in config.intents:
                name_tokens = _tokens(intent.name.replace("_", " "))
                self._vocab[intent.name] = [name_tokens] + [_tokens(e) for e in intent.examples]

    def complete(self, system: str, user: str, schema: dict) -> ProviderResponse:
        if self._responses is not None:
            data = next(self._responses)
            if isinstance(data, Exception):
                raise data
        else:
            match = _QUERY_TAG.search(user)
            data = self._score(match.group(1) if match else user)
        return ProviderResponse(
            data=data,
            model=self.model,
            input_tokens=_estimate_tokens(system + user),
            output_tokens=_estimate_tokens(str(data)),
        )

    def _score(self, query: str) -> dict:
        q = _tokens(query)
        scores = {
            name: max((len(q & ref) / len(q | ref) for ref in refs), default=0.0) if q else 0.0
            for name, refs in self._vocab.items()
        }
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        (best, top), runner_up = ranked[0], ranked[1][1] if len(ranked) > 1 else 0.0

        if top < self._threshold:
            return {
                "reasoning": f"mock: best overlap {top:.2f} below threshold",
                "intent": OUT_OF_SCOPE,
                "secondary_intents": [],
                "confidence": "medium",
            }
        margin = top - runner_up
        confidence = "high" if margin >= 0.2 else "medium" if margin >= 0.08 else "low"
        return {
            "reasoning": f"mock: overlap {top:.2f}, margin {margin:.2f}",
            "intent": best,
            "secondary_intents": [],
            "confidence": confidence,
        }
