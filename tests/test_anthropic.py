"""Tests for the Anthropic Messages API backend.

Never imports the ``anthropic`` package, so these pass with the optional
extra uninstalled - which is the state CI is in.
"""

from __future__ import annotations

from typing import Any

import pytest

from claude_as_api.anthropic_api import AnthropicBackend
from claude_as_api.core import LLMAuthError, LLMRateLimitError, LLMSchemaError

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"label": {"type": "string"}},
}


class FakeBlock:
    """One content block of a fake Message."""

    def __init__(self, text: str) -> None:
        """Store the text.

        Args:
            text: Block text.
        """
        self.type = "text"
        self.text = text


class FakeUsage:
    """Token counts on a fake Message."""

    input_tokens = 21
    output_tokens = 5


class FakeMessage:
    """A stand-in for an Anthropic Message."""

    def __init__(self, text: str) -> None:
        """Store the single text block.

        Args:
            text: Block text.
        """
        self.content = [FakeBlock(text)]
        self.usage = FakeUsage()
        self.stop_reason = "end_turn"


class FakeMessages:
    """Records the request and returns a canned Message."""

    def __init__(self, text: str, error: Exception | None = None) -> None:
        """Configure the fake.

        Args:
            text: Text the fake Message should carry.
            error: If given, raised instead of answering.
        """
        self.text = text
        self.error = error
        self.request: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> FakeMessage:  # noqa: ANN401
        """Record and answer.

        Args:
            **kwargs: The request.

        Returns:
            A canned Message.

        Raises:
            Exception: The configured error, if any.
        """
        self.request = kwargs
        if self.error is not None:
            raise self.error
        return FakeMessage(self.text)


class FakeClient:
    """Minimal stand-in for ``anthropic.Anthropic``."""

    def __init__(
        self, text: str = '{"label":"non_food"}', error: Exception | None = None
    ) -> None:
        """Configure the fake.

        Args:
            text: Text the fake Message should carry.
            error: If given, raised on create().
        """
        self.messages = FakeMessages(text, error)


def _generate(client: FakeClient, **overrides: Any) -> Any:  # noqa: ANN401
    """Run the backend against a fake client.

    Args:
        client: The fake.
        **overrides: Overrides for the generate() keyword arguments.

    Returns:
        The LLMResponse.
    """
    kwargs: dict[str, Any] = {
        "prompt": "PPR TWL 2PLY 30RL",
        "system": None,
        "model": "claude-opus-5",
        "schema": SCHEMA,
        "effort": "low",
        "max_tokens": 512,
        "timeout_s": 10.0,
    }
    kwargs.update(overrides)
    return AnthropicBackend(client=client).generate(**kwargs)


def test_never_sends_tools() -> None:
    """Tool use is not reachable through this backend."""
    request = AnthropicBackend().build_request(
        prompt="x", system=None, model="m", schema=None, effort="low", max_tokens=10
    )
    assert "tools" not in request


def test_always_sends_explicit_thinking() -> None:
    """Thinking is ON by default for Opus 5, so it must be stated."""
    request = AnthropicBackend().build_request(
        prompt="x", system=None, model="m", schema=None, effort="medium", max_tokens=10
    )
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"]["effort"] == "medium"


def test_effort_none_disables_thinking_without_setting_effort() -> None:
    """Disabling thinking is rejected above effort 'high', so omit effort."""
    request = AnthropicBackend().build_request(
        prompt="x", system=None, model="m", schema=None, effort="none", max_tokens=10
    )
    assert request["thinking"] == {"type": "disabled"}
    assert "effort" not in request.get("output_config", {})


def test_structured_output_uses_output_config_format() -> None:
    """The deprecated top-level output_format parameter must not be used."""
    request = AnthropicBackend().build_request(
        prompt="x", system=None, model="m", schema=SCHEMA, effort="low", max_tokens=10
    )
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert "output_format" not in request


def test_successful_call_is_parsed() -> None:
    """The first text block is parsed as the structured answer."""
    response = _generate(FakeClient())
    assert response.data == {"label": "non_food"}
    assert response.input_tokens == 21
    assert response.num_turns == 1


def test_non_json_reply_for_a_schema_is_a_schema_error() -> None:
    """A prompt bug, not a blip."""
    with pytest.raises(LLMSchemaError):
        _generate(FakeClient(text="sorry, I cannot"))


def test_sdk_errors_are_mapped_by_class_name() -> None:
    """Mapping by name means the SDK need not be importable."""
    rate_limit = type("RateLimitError", (Exception,), {})
    with pytest.raises(LLMRateLimitError):
        _generate(FakeClient(error=rate_limit("slow down")))

    auth = type("AuthenticationError", (Exception,), {})
    with pytest.raises(LLMAuthError):
        _generate(FakeClient(error=auth("bad key")))
