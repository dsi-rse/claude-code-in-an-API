"""Tests for the OpenRouter backend, using httpx.MockTransport (no network)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from claude_as_api.core import (
    DEFAULT_APP_TITLE,
    DEFAULT_HTTP_REFERER,
    LLMAuthError,
    LLMRateLimitError,
    LLMSchemaError,
    LLMTransientError,
)
from claude_as_api.openrouter import OpenRouterBackend

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"label": {"type": "string"}},
}

CANNED: dict[str, Any] = {
    "id": "gen-abc",
    "model": "anthropic/claude-opus-5",
    "choices": [
        {"message": {"content": '{"label":"non_food"}'}, "finish_reason": "stop"}
    ],
    "usage": {
        "prompt_tokens": 42,
        "completion_tokens": 11,
        "completion_tokens_details": {"reasoning_tokens": 3},
        "cost": 0.0012,
    },
}


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a dummy credential so tests reach the request-building code.

    Args:
        monkeypatch: pytest's environment patcher.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")


def _run(handler: Any, **overrides: Any) -> Any:  # noqa: ANN401
    """Call the backend against a mock transport.

    Args:
        handler: An httpx.MockTransport handler.
        **overrides: Overrides for the generate() keyword arguments.

    Returns:
        The LLMResponse.
    """
    backend = OpenRouterBackend(transport=httpx.MockTransport(handler))
    kwargs: dict[str, Any] = {
        "prompt": "PPR TWL 2PLY 30RL",
        "system": None,
        "model": "anthropic/claude-opus-5",
        "schema": SCHEMA,
        "effort": "low",
        "max_tokens": 512,
        "timeout_s": 10.0,
    }
    kwargs.update(overrides)
    return backend.generate(**kwargs)


def test_never_sends_tools_or_plugins() -> None:
    """Nothing web-related may be reachable, even accidentally."""
    body = OpenRouterBackend().build_body(
        prompt="x", system=None, model="m", schema=None, effort="low", max_tokens=10
    )
    assert "tools" not in body
    assert "plugins" not in body
    assert ":online" not in body["model"]


def test_always_sends_explicit_reasoning() -> None:
    """Reasoning is ON by default for Opus 5 here, so it must be stated."""
    body = OpenRouterBackend().build_body(
        prompt="x", system=None, model="m", schema=None, effort="low", max_tokens=10
    )
    assert body["reasoning"] == {"effort": "low"}


def test_effort_none_disables_reasoning() -> None:
    """OpenRouter can genuinely turn reasoning off, unlike the CLI."""
    body = OpenRouterBackend().build_body(
        prompt="x", system=None, model="m", schema=None, effort="none", max_tokens=10
    )
    assert body["reasoning"] == {"enabled": False}


def test_schema_request_requires_provider_parameter_support() -> None:
    """Otherwise OpenRouter may route to a provider that ignores the schema."""
    body = OpenRouterBackend().build_body(
        prompt="x", system=None, model="m", schema=SCHEMA, effort="low", max_tokens=10
    )
    assert body["provider"] == {"require_parameters": True}
    assert body["response_format"]["json_schema"]["strict"] is True
    assert (
        body["response_format"]["json_schema"]["schema"]["additionalProperties"]
        is False
    )


def test_successful_call_is_parsed() -> None:
    """Structured output and reasoning tokens are surfaced."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["Authorization"]
        return httpx.Response(200, json=CANNED)

    response = _run(handler)
    assert response.data == {"label": "non_food"}
    assert response.thinking_tokens == 3
    assert response.input_tokens == 42
    assert response.cost_usd == pytest.approx(0.0012)
    assert seen["auth"] == "Bearer or-test-key"


def test_missing_key_is_an_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail fast with a message naming the variable."""
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(LLMAuthError, match="OPENROUTER_API_KEY"):
        _run(lambda request: httpx.Response(200, json=CANNED))


def test_401_is_an_auth_error() -> None:
    """Never retried."""
    with pytest.raises(LLMAuthError):
        _run(lambda request: httpx.Response(401, text="bad key"))


def test_429_is_a_rate_limit_error_and_reports_retry_after() -> None:
    """Retry-After is worth surfacing to the student."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down", headers={"Retry-After": "30"})

    with pytest.raises(LLMRateLimitError, match="30"):
        _run(handler)


def test_503_is_transient() -> None:
    """Worth a retry."""
    with pytest.raises(LLMTransientError):
        _run(lambda request: httpx.Response(503, text="unavailable"))


def test_non_json_reply_for_a_schema_is_a_schema_error() -> None:
    """A prompt bug, not a blip: do not retry it."""
    payload = {**CANNED, "choices": [{"message": {"content": "sorry, I cannot"}}]}
    with pytest.raises(LLMSchemaError):
        _run(lambda request: httpx.Response(200, json=payload))


def test_sends_default_attribution_headers() -> None:
    """OpenRouter attributes usage by HTTP-Referer/X-Title, so both are sent."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, json=CANNED)

    _run(handler)
    assert seen["http-referer"] == DEFAULT_HTTP_REFERER
    assert seen["x-title"] == DEFAULT_APP_TITLE


def test_attribution_headers_are_configurable() -> None:
    """A host project must be able to attribute usage to itself, not to us."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, json=CANNED)

    backend = OpenRouterBackend(
        transport=httpx.MockTransport(handler),
        http_referer="https://github.com/dsi-rse/some-clinic-project",
        app_title="some-clinic-project",
    )
    backend.generate(
        prompt="x",
        system=None,
        model="anthropic/claude-opus-5",
        schema=None,
        effort="low",
        max_tokens=10,
        timeout_s=10.0,
    )
    assert seen["http-referer"] == "https://github.com/dsi-rse/some-clinic-project"
    assert seen["x-title"] == "some-clinic-project"
