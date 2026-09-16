"""Tests for the shared contract: schema tightening and effort validation."""

from __future__ import annotations

import pytest

from claude_as_api.core import (
    DEFAULT_APP_TITLE,
    DEFAULT_HTTP_REFERER,
    DEFAULT_MODELS,
    LLMConfig,
    LLMUnsupportedError,
    strict_schema,
    validate_effort,
)


def test_strict_schema_marks_properties_required() -> None:
    """Every declared property becomes required and extras are banned."""
    out = strict_schema({"type": "object", "properties": {"a": {"type": "string"}}})
    assert out["required"] == ["a"]
    assert out["additionalProperties"] is False


def test_strict_schema_does_not_mutate_input() -> None:
    """The caller's schema object is left alone."""
    original = {"type": "object", "properties": {"a": {"type": "string"}}}
    strict_schema(original)
    assert "required" not in original


def test_strict_schema_recurses_into_nested_structures() -> None:
    """Nested objects, arrays and $defs are tightened too."""
    schema = {
        "type": "object",
        "properties": {
            "inner": {"type": "object", "properties": {"b": {"type": "number"}}},
            "items": {
                "type": "array",
                "items": {"type": "object", "properties": {"c": {}}},
            },
        },
        "$defs": {"d": {"type": "object", "properties": {"e": {}}}},
    }
    out = strict_schema(schema)
    assert out["properties"]["inner"]["required"] == ["b"]
    assert out["properties"]["items"]["items"]["required"] == ["c"]
    assert out["$defs"]["d"]["required"] == ["e"]


def test_strict_schema_is_idempotent() -> None:
    """Applying it twice changes nothing."""
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    once = strict_schema(schema)
    assert strict_schema(once) == once


def test_validate_effort_rejects_none_when_unsupported() -> None:
    """claude_code cannot disable reasoning, and says so usefully."""
    with pytest.raises(LLMUnsupportedError, match='effort="low"'):
        validate_effort("none", provider="claude_code", supports_none=False)


def test_validate_effort_allows_none_when_supported() -> None:
    """The remote backends can disable reasoning."""
    assert validate_effort("none", provider="openrouter", supports_none=True) == "none"


def test_validate_effort_rejects_unknown_level() -> None:
    """A typo is an error, not a silent fallback."""
    with pytest.raises(LLMUnsupportedError, match="unknown effort"):
        validate_effort("hgih", provider="anthropic", supports_none=True)


@pytest.mark.parametrize("provider", list(DEFAULT_MODELS))
def test_resolve_model_falls_back_to_provider_default(provider: str) -> None:
    """Each provider has its own default slug spelling."""
    config = LLMConfig(provider=provider, model=None)
    assert config.resolve_model() == DEFAULT_MODELS[provider]


def test_resolve_model_prefers_explicit_override() -> None:
    """An explicit per-call model wins over config and default."""
    config = LLMConfig(provider="claude_code", model="sonnet")
    assert config.resolve_model("haiku") == "haiku"


def test_resolve_model_rejects_unknown_provider() -> None:
    """A bad LLM_PROVIDER fails with a list of the valid ones."""
    with pytest.raises(LLMUnsupportedError, match="unknown provider"):
        LLMConfig(provider="gpt").resolve_model()


def test_from_env_defaults_with_an_empty_environment() -> None:
    """With nothing set, from_env matches the dataclass defaults."""
    config = LLMConfig.from_env()
    assert config.provider == "claude_code"
    assert config.model is None
    assert config.effort == "low"
    assert config.max_tokens == 4096
    assert config.timeout_s == 180.0
    assert config.max_workers == 4
    assert config.max_retries == 3
    assert config.claude_bin == "claude"
    assert config.scratch_dir is None
    assert config.http_referer == DEFAULT_HTTP_REFERER
    assert config.app_title == DEFAULT_APP_TITLE


def test_from_env_reads_every_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each setting is wired to its own environment variable, with casts."""
    for name, value in {
        "LLM_PROVIDER": "openrouter",
        "LLM_MODEL": "anthropic/claude-haiku-4.5",
        "LLM_EFFORT": "max",
        "LLM_MAX_TOKENS": "256",
        "LLM_TIMEOUT_S": "12.5",
        "LLM_MAX_WORKERS": "9",
        "LLM_MAX_RETRIES": "1",
        "CLAUDE_BIN": "/opt/bin/claude",
        "LLM_SCRATCH_DIR": "/tmp/elsewhere",  # noqa: S108
        "LLM_HTTP_REFERER": "https://example.org/mine",
        "LLM_APP_TITLE": "mine",
    }.items():
        monkeypatch.setenv(name, value)

    config = LLMConfig.from_env()
    assert config.provider == "openrouter"
    assert config.model == "anthropic/claude-haiku-4.5"
    assert config.effort == "max"
    assert config.max_tokens == 256
    assert config.timeout_s == 12.5
    assert config.max_workers == 9
    assert config.max_retries == 1
    assert config.claude_bin == "/opt/bin/claude"
    assert config.scratch_dir == "/tmp/elsewhere"  # noqa: S108
    assert config.http_referer == "https://example.org/mine"
    assert config.app_title == "mine"


def test_from_env_treats_blank_model_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty LLM_MODEL must become None, not "".

    docker-compose passes `LLM_MODEL: ${LLM_MODEL:-}`, so the variable is
    present-but-empty far more often than it is absent. An empty string would
    be sent to the provider as a model id.
    """
    monkeypatch.setenv("LLM_MODEL", "")
    assert LLMConfig.from_env().model is None
    assert LLMConfig.from_env().resolve_model() == DEFAULT_MODELS["claude_code"]
