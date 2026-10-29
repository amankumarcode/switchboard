from switchboard.config import OUT_OF_SCOPE
from switchboard.prompt import build_schema, build_system_prompt, build_user_message, prompt_version


def test_schema_enum_matches_config(banking_config):
    schema = build_schema(banking_config)
    props = schema["properties"]
    assert props["intent"]["enum"] == banking_config.intent_names
    assert props["secondary_intents"]["items"]["enum"] == banking_config.intent_names
    assert props["confidence"]["enum"] == ["low", "medium", "high"]
    assert set(schema["required"]) == set(props)
    assert schema["additionalProperties"] is False


def test_reasoning_precedes_intent(banking_config):
    keys = list(build_schema(banking_config)["properties"])
    assert keys.index("reasoning") < keys.index("intent")


def test_system_prompt_lists_every_intent_and_example(banking_config):
    prompt = build_system_prompt(banking_config)
    for intent in banking_config.intents:
        assert f"## {intent.name}" in prompt
        for example in intent.examples:
            assert example in prompt
    assert f"## {OUT_OF_SCOPE}" in prompt


def test_user_message_wraps_query():
    assert build_user_message("hello") == "<query>\nhello\n</query>"


def test_prompt_version_is_stable_and_config_sensitive(banking_config):
    v1 = prompt_version(banking_config)
    assert v1 == prompt_version(banking_config)
    assert len(v1) == 8

    changed = banking_config.model_copy(
        update={
            "intents": [
                banking_config.intents[0].model_copy(update={"description": "Something else."}),
                *banking_config.intents[1:],
            ]
        }
    )
    assert prompt_version(changed) != v1
