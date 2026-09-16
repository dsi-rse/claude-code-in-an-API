"""Core types shared by every LLM backend.

This module defines the contract that :mod:`claude_as_api.claude_code`,
:mod:`claude_as_api.anthropic_api` and :mod:`claude_as_api.openrouter` all implement, so
that pipeline code can switch providers without changing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol

# Reasoning depth, uniform across backends. "none" means "do not reason at all";
# it is not supported by every backend (see LLMBackend.supports_effort_none).
Effort = Literal["none", "low", "medium", "high", "xhigh", "max"]

EFFORT_LEVELS: tuple[Effort, ...] = ("none", "low", "medium", "high", "xhigh", "max")

#: Default model per provider. Note the slug spellings genuinely differ: the CLI
#: takes short aliases, the Anthropic API uses dashes, OpenRouter uses dots.
DEFAULT_MODELS: dict[str, str] = {
    "claude_code": "opus",
    "anthropic": "claude-opus-5",
    "openrouter": "anthropic/claude-opus-5",
}

PROVIDERS: tuple[str, ...] = tuple(DEFAULT_MODELS)


class LLMError(Exception):
    """Base class for every error raised by this package."""


class LLMAuthError(LLMError):
    """Credentials are missing, invalid, or the CLI is not installed.

    Never retried: retrying cannot fix a missing credential.
    """


class LLMRateLimitError(LLMError):
    """The provider rejected the call for rate-limit or quota reasons.

    Never retried automatically. On a Claude subscription, quota exhaustion
    lasts hours, so grinding through the rest of a batch is pure waste.
    """


class LLMTimeoutError(LLMError):
    """The call did not finish within the timeout. Retried."""


class LLMTransientError(LLMError):
    """A server-side or connection failure that may succeed on retry."""


class LLMSchemaError(LLMError):
    """A JSON Schema was requested but no valid structured result came back.

    Never retried: this is almost always a prompt or schema bug, and retrying
    just spends quota to get the same answer.
    """


class LLMUnsupportedError(LLMError):
    """The selected backend cannot honour a requested capability."""


@dataclass(frozen=True)
class LLMResponse:
    """One completion, in a shape that is identical across all backends.

    Attributes:
        text: The response text. Always populated.
        data: Parsed structured output, or None when no schema was requested.
        provider: Which backend produced this.
        model: The resolved model identifier that was actually sent.
        effort: The reasoning depth that was actually requested.
        input_tokens: Prompt tokens, when the provider reports them.
        output_tokens: Completion tokens, when the provider reports them.
        thinking_tokens: Reasoning tokens. A portability signal - a workflow
            that burns many of these per row leans on reasoning and may not
            survive a move to a bare LLM.
        num_turns: Model turns taken. Also a portability signal.
        cost_usd: A client-side ESTIMATE. On a Claude subscription this is not
            a bill; the call consumes plan quota instead.
        latency_ms: Wall-clock duration reported by the provider.
        raw: The provider's own payload, for debugging.
    """

    text: str
    data: dict[str, Any] | None
    provider: str
    model: str
    effort: Effort
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    num_turns: int | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class LLMBackend(Protocol):
    """The interface every backend implements.

    A Protocol rather than an ABC so that a test - or a student experimenting in
    a notebook - can supply a stand-in without importing or subclassing anything.
    """

    name: str
    supports_effort_none: bool

    def generate(
        self,
        *,
        prompt: str,
        system: str | None,
        model: str,
        schema: dict[str, Any] | None,
        effort: Effort,
        max_tokens: int,
        timeout_s: float,
    ) -> LLMResponse:
        """Run one completion.

        Args:
            prompt: The user prompt.
            system: System prompt, or None.
            model: Provider-specific model identifier.
            schema: JSON Schema for structured output, or None.
            effort: Reasoning depth.
            max_tokens: Output token ceiling.
            timeout_s: Per-call timeout in seconds.

        Returns:
            The completion.

        Raises:
            LLMError: Or one of its subclasses, on any failure.
        """
        ...  # pragma: no cover


def validate_effort(effort: str, *, provider: str, supports_none: bool) -> Effort:
    """Check that an effort level is valid for a backend.

    Args:
        effort: The requested reasoning depth.
        provider: Backend name, used only in the error message.
        supports_none: Whether this backend can disable reasoning entirely.

    Returns:
        The validated effort level.

    Raises:
        LLMUnsupportedError: If the level is unknown, or is "none" on a backend
            that cannot disable reasoning.
    """
    if effort not in EFFORT_LEVELS:
        msg = f"unknown effort {effort!r}; expected one of {', '.join(EFFORT_LEVELS)}"
        raise LLMUnsupportedError(msg)
    if effort == "none" and not supports_none:
        msg = (
            f"the {provider!r} backend cannot disable reasoning entirely; the "
            'CLI rejects --effort none. Use effort="low", the closest available '
            "setting, and check LLMResponse.thinking_tokens to see how much "
            "reasoning actually happened."
        )
        raise LLMUnsupportedError(msg)
    return effort  # type: ignore[return-value]


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a JSON Schema tightened for strict structured output.

    Sets ``additionalProperties: false`` and marks every declared property as
    required, recursively. Claude Code's ``--json-schema`` is lenient about
    both, but the Anthropic ``json_schema`` format and OpenRouter's
    ``strict: true`` reject schemas that omit them. Normalising here means one
    schema works on all three backends.

    Args:
        schema: Any JSON Schema fragment.

    Returns:
        A new dict; the input is not modified. Idempotent.
    """
    if not isinstance(schema, dict):
        return schema

    out: dict[str, Any] = dict(schema)

    if "properties" in out and isinstance(out["properties"], dict):
        out["properties"] = {
            key: strict_schema(value) for key, value in out["properties"].items()
        }
        out["additionalProperties"] = False
        out["required"] = list(out["properties"])

    for key in ("items", "additionalItems", "not"):
        if key in out:
            out[key] = strict_schema(out[key])

    for key in ("$defs", "definitions"):
        if key in out and isinstance(out[key], dict):
            out[key] = {k: strict_schema(v) for k, v in out[key].items()}

    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        if key in out and isinstance(out[key], list):
            out[key] = [strict_schema(item) for item in out[key]]

    return out


#: Sent to OpenRouter as ``HTTP-Referer``/``X-Title`` so usage is attributable
#: to something. Host projects should override these with their own repo and
#: name via LLM_HTTP_REFERER / LLM_APP_TITLE.
DEFAULT_HTTP_REFERER: Final = "https://github.com/dsi-rse/claude-code-in-an-API"
DEFAULT_APP_TITLE: Final = "claude-code-in-an-API"


@dataclass(frozen=True)
class LLMConfig:
    """Resolved configuration for an :class:`~claude_as_api.client.LLMClient`."""

    provider: str = "claude_code"
    model: str | None = None
    effort: Effort = "low"
    max_tokens: int = 4096
    timeout_s: float = 180.0
    max_workers: int = 4
    max_retries: int = 3
    claude_bin: str = "claude"
    #: Working directory for the claude_code backend's subprocess. None means
    #: ``.llm_scratch`` under the current working directory.
    scratch_dir: str | None = None
    http_referer: str = DEFAULT_HTTP_REFERER
    app_title: str = DEFAULT_APP_TITLE

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Build a config from environment variables, loading ``.env`` first.

        Called lazily rather than at import, so that ``import claude_as_api``
        reads no environment variables and touches no filesystem.

        ``load_dotenv`` does not override variables that are already set, so a
        real environment variable always wins over a ``.env`` entry.

        LLM_MODEL is deliberately None rather than a string when unset: "haiku"
        is a valid Claude Code alias but an invalid Anthropic API and OpenRouter
        model id, so the default has to be resolved per provider.

        Returns:
            A config populated from the environment.
        """
        from dotenv import load_dotenv

        load_dotenv()

        return cls(
            provider=os.environ.get("LLM_PROVIDER", "claude_code"),
            model=os.environ.get("LLM_MODEL") or None,
            effort=os.environ.get("LLM_EFFORT", "low"),  # type: ignore[arg-type]
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "4096")),
            timeout_s=float(os.environ.get("LLM_TIMEOUT_S", "180")),
            max_workers=int(os.environ.get("LLM_MAX_WORKERS", "4")),
            max_retries=int(os.environ.get("LLM_MAX_RETRIES", "3")),
            claude_bin=os.environ.get("CLAUDE_BIN", "claude"),
            scratch_dir=os.environ.get("LLM_SCRATCH_DIR") or None,
            http_referer=os.environ.get("LLM_HTTP_REFERER", DEFAULT_HTTP_REFERER),
            app_title=os.environ.get("LLM_APP_TITLE", DEFAULT_APP_TITLE),
        )

    def resolve_model(self, override: str | None = None) -> str:
        """Pick the model to send.

        Args:
            override: An explicit per-call model, if any.

        Returns:
            The first of: the override, the configured model, the provider default.

        Raises:
            LLMUnsupportedError: If the provider is not recognised.
        """
        if self.provider not in DEFAULT_MODELS:
            msg = (
                f"unknown provider {self.provider!r}; "
                f"expected one of {', '.join(PROVIDERS)}"
            )
            raise LLMUnsupportedError(msg)
        return override or self.model or DEFAULT_MODELS[self.provider]


def require_env(name: str, *, provider: str) -> str:
    """Read a required credential from the environment.

    Args:
        name: Environment variable name.
        provider: Backend name, used in the error message.

    Returns:
        The value.

    Raises:
        LLMAuthError: If the variable is unset or empty.
    """
    value = os.environ.get(name)
    if not value:
        msg = f"{name} is not set, which the {provider!r} backend requires"
        raise LLMAuthError(msg)
    return value
