"""Backend for the first-party Anthropic Messages API.

Requires ``ANTHROPIC_API_KEY``, i.e. a console.anthropic.com account. A Claude
Enterprise (claude.ai) seat is a different product and will not work here - use
the ``claude_code`` backend for that.

Install the optional dependency with ``uv sync --extra anthropic``.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Final

from .core import (
    Effort,
    LLMAuthError,
    LLMError,
    LLMRateLimitError,
    LLMResponse,
    LLMSchemaError,
    LLMTimeoutError,
    LLMTransientError,
    strict_schema,
    validate_effort,
)

if TYPE_CHECKING:  # pragma: no cover
    pass

_HTTP_SERVER_ERROR: Final = 500
_MS_PER_S: Final = 1000.0


class AnthropicBackend:
    """Calls ``client.messages.create`` and normalises the result."""

    name = "anthropic"
    supports_effort_none = True

    def __init__(self, client: object | None = None) -> None:
        """Configure the backend.

        Args:
            client: A pre-built Anthropic client. Mainly for tests; when None,
                one is created on first use from ``ANTHROPIC_API_KEY``.
        """
        self._client = client

    def _get_client(self) -> Any:  # noqa: ANN401
        """Return the Anthropic client, importing the SDK lazily.

        Returns:
            An ``anthropic.Anthropic`` instance.

        Raises:
            LLMError: If the optional dependency is not installed.
        """
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on install
            msg = (
                "the 'anthropic' package is not installed. "
                "Run `uv sync --extra anthropic`, or set LLM_PROVIDER=claude_code."
            )
            raise LLMError(msg) from exc
        # max_retries=0: our own retry layer is the only one. Nesting the SDK's
        # backoff inside ours would turn 3 attempts into 9 real calls.
        self._client = anthropic.Anthropic(max_retries=0)
        return self._client

    @staticmethod
    def reasoning_kwargs(effort: Effort) -> dict[str, Any]:
        """Build the thinking/effort portion of the request.

        Always sent explicitly: thinking is ON by default for Opus 5, so
        omitting this would silently give a different pipeline than the other
        backends produce.

        Args:
            effort: Reasoning depth.

        Returns:
            Keyword arguments for ``messages.create``.
        """
        if effort == "none":
            # No output_config.effort here on purpose: disabling thinking is
            # rejected above effort "high", and the default is "high".
            return {"thinking": {"type": "disabled"}}
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}

    def build_request(
        self,
        *,
        prompt: str,
        system: str | None,
        model: str,
        schema: dict[str, Any] | None,
        effort: Effort,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Assemble the request payload.

        Split out so tests can assert on it without a network call.

        Args:
            prompt: The user prompt.
            system: System prompt, or None.
            model: Model id.
            schema: JSON Schema for structured output, or None.
            effort: Reasoning depth.
            max_tokens: Output token ceiling.

        Returns:
            Keyword arguments for ``messages.create``. Never includes ``tools``.
        """
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            request["system"] = system

        reasoning = self.reasoning_kwargs(effort)
        request["thinking"] = reasoning["thinking"]
        output_config: dict[str, Any] = dict(reasoning.get("output_config", {}))

        if schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": strict_schema(schema),
            }
        if output_config:
            request["output_config"] = output_config
        return request

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
            model: Model id.
            schema: JSON Schema for structured output, or None.
            effort: Reasoning depth.
            max_tokens: Output token ceiling.
            timeout_s: Per-call timeout in seconds.

        Returns:
            The completion.

        Raises:
            LLMSchemaError: If a schema was requested but the reply is not JSON.
            LLMError: Or a subclass, on any other failure.
        """
        effort = validate_effort(effort, provider=self.name, supports_none=True)
        request = self.build_request(
            prompt=prompt,
            system=system,
            model=model,
            schema=schema,
            effort=effort,
            max_tokens=max_tokens,
        )
        client = self._get_client()

        started = time.monotonic()
        try:
            message = client.messages.create(timeout=timeout_s, **request)
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed LLMError
            raise self._classify(exc) from exc
        latency_ms = (time.monotonic() - started) * _MS_PER_S

        text = next(
            (b.text for b in message.content if getattr(b, "type", None) == "text"), ""
        )
        data = None
        if schema is not None:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                msg = f"expected JSON for the requested schema, got: {text[:300]}"
                raise LLMSchemaError(msg) from exc

        usage = getattr(message, "usage", None)
        return LLMResponse(
            text=text,
            data=data,
            provider=self.name,
            model=model,
            effort=effort,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            thinking_tokens=None,  # not broken out by the Messages API
            num_turns=1,
            cost_usd=None,
            latency_ms=latency_ms,
            raw={"stop_reason": getattr(message, "stop_reason", None)},
        )

    @staticmethod
    def _classify(exc: Exception) -> Exception:
        """Map an SDK exception onto this package's exception types.

        Matches on class name so that the SDK need not be importable.

        Args:
            exc: The exception the SDK raised.

        Returns:
            The exception to raise instead.
        """
        name = type(exc).__name__
        if name in {"AuthenticationError", "PermissionDeniedError"}:
            return LLMAuthError(f"Anthropic rejected the API key: {exc}")
        if name == "RateLimitError":
            return LLMRateLimitError(f"Anthropic rate limit reached: {exc}")
        if name == "APITimeoutError":
            return LLMTimeoutError(f"Anthropic request timed out: {exc}")
        if name == "APIConnectionError":
            return LLMTransientError(f"could not reach Anthropic: {exc}")
        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and status >= _HTTP_SERVER_ERROR:
            return LLMTransientError(f"Anthropic server error {status}: {exc}")
        if isinstance(exc, LLMError):
            return exc
        return LLMError(f"Anthropic request failed: {exc}")
