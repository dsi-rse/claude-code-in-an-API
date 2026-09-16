"""Call an LLM as a plain function, with no API key required.

A Claude Enterprise seat is not Anthropic API access, but it does give you the
``claude`` CLI - and the default backend here drives that CLI as an ordinary
completion API. That is what makes this usable by anyone who has Claude Code
but no developer platform account.

Import everything from here, not from the backend modules::

    from claude_as_api import complete_json, map_dataframe

The active provider comes from ``LLM_PROVIDER`` in the environment or ``.env``:

* ``claude_code`` (default) - drives the Claude Code CLI. This is the one that
  works with a University of Chicago Claude Enterprise seat.
* ``anthropic`` - the Anthropic Messages API. Needs ``ANTHROPIC_API_KEY``.
* ``openrouter`` - OpenRouter. Needs ``OPENROUTER_API_KEY``.

All three are held to the same contract: no tools, no web access, an explicit
reasoning level on every call, and the same JSON Schema format. See the README.
"""

from __future__ import annotations

from typing import Any

from .client import LLMClient, build_backend, map_dataframe
from .core import (
    DEFAULT_APP_TITLE,
    DEFAULT_HTTP_REFERER,
    DEFAULT_MODELS,
    EFFORT_LEVELS,
    PROVIDERS,
    Effort,
    LLMAuthError,
    LLMBackend,
    LLMConfig,
    LLMError,
    LLMRateLimitError,
    LLMResponse,
    LLMSchemaError,
    LLMTimeoutError,
    LLMTransientError,
    LLMUnsupportedError,
    strict_schema,
    validate_effort,
)

__all__ = [
    "DEFAULT_APP_TITLE",
    "DEFAULT_HTTP_REFERER",
    "DEFAULT_MODELS",
    "EFFORT_LEVELS",
    "PROVIDERS",
    "Effort",
    "LLMAuthError",
    "LLMBackend",
    "LLMClient",
    "LLMConfig",
    "LLMError",
    "LLMRateLimitError",
    "LLMResponse",
    "LLMSchemaError",
    "LLMTimeoutError",
    "LLMTransientError",
    "LLMUnsupportedError",
    "build_backend",
    "complete",
    "complete_json",
    "map_dataframe",
    "map_prompts",
    "strict_schema",
    "validate_effort",
]

_default_client: LLMClient | None = None


def get_client() -> LLMClient:
    """Return the process-wide default client, creating it on first use.

    Built lazily so that importing this package reads no environment variables
    and touches no filesystem.

    Returns:
        The shared client.
    """
    global _default_client  # noqa: PLW0603
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def complete(prompt: str, **kwargs: Any) -> LLMResponse:  # noqa: ANN401
    """Run one completion with the default client.

    Args:
        prompt: The user prompt.
        **kwargs: Passed to :meth:`LLMClient.complete`.

    Returns:
        The completion.
    """
    return get_client().complete(prompt, **kwargs)


def complete_json(
    prompt: str,
    *,
    schema: dict[str, Any],
    **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """Run one completion and return the parsed structured output.

    Args:
        prompt: The user prompt.
        schema: JSON Schema the answer must satisfy.
        **kwargs: Passed to :meth:`LLMClient.complete`.

    Returns:
        The parsed object.
    """
    return get_client().complete_json(prompt, schema=schema, **kwargs)


def map_prompts(prompts: list[str], **kwargs: Any) -> list[LLMResponse | LLMError]:  # noqa: ANN401
    """Run many prompts concurrently with the default client.

    Args:
        prompts: The prompts to run.
        **kwargs: Passed to :meth:`LLMClient.map_prompts`.

    Returns:
        One entry per prompt, in input order.
    """
    return get_client().map_prompts(prompts, **kwargs)
