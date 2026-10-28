# Switchboard

A small, config-driven intent router. An incoming user query is classified by an LLM
into one of a fixed set of intents, a policy layer decides whether it can be handled
automatically or needs a human, and a dispatcher triggers the matching workflow.
Everything domain-specific lives in one config file, so the engine can route banking,
IT helpdesk, or HR queries without code changes. The reference domain is retail
banking customer support.

> **Status:** work in progress.

## Development

```bash
uv sync
uv run ruff check
uv run pytest
```

## Data & credits

- **Banking77** — Casanueva et al., *Efficient Intent Detection with Dual Sentence
  Encoders* (2020). [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77),
  licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **CLINC150** (if used for out-of-scope samples) — Larson et al., *An Evaluation
  Dataset for Intent Classification and Out-of-Scope Prediction* (2019).
  [clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos), licensed under
  [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).

## License

Code is released under the [MIT License](LICENSE). Datasets retain their own licenses
(see above).
