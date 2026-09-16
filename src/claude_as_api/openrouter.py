"""Backend for OpenRouter's OpenAI-compatible chat completions endpoint.

Requires ``OPENROUTER_API_KEY``.

Two OpenRouter quirks worth knowing:

* Model slugs use dots where the first-party Anthropic API uses dashes:
  ``anthropic/claude-haiku-4.5`` versus ``claude-haiku-4-5``.
* Never declare a custom function named ``web_search`` - OpenRouter hoists it
  to its server-side search tool. This backend sends no ``tools`` array at all,
  and no ``plugins``, so nothing web-related is ever enabled.
"""

from __future__ import annotations

import json
from typing import Any, Final

import httpx

from .core import (
    DEFAULT_APP_TITLE,
    DEFAULT_HTTP_REFERER,
    Effort,
    LLMAuthError,
    LLMRateLimitError,
    LLMResponse,
    LLMSchemaError,
    LLMTimeoutError,
    LLMTransientError,
    require_env,
    strict_schema,
    validate_effort,
)

_ENDPOINT: Final = "https://openrouter.ai/api/v1/chat/completions"
_HTTP_UNAUTHORIZED: Final = 401
_HTTP_FORBIDDEN: Final = 403
_HTTP_TOO_MANY_REQUESTS: Final = 429
_HTTP_SERVER_ERROR: Final = 500


class OpenRouterBackend:
    """Posts to OpenRouter and normalises the result."""

    name = "openrouter"
    supports_effort_none = True

    def __init__(
        self,
        transport: httpx.BaseTransport | None = None,
        http_referer: str = DEFAULT_HTTP_REFERER,
        app_title: str = DEFAULT_APP_TITLE,
    ) -> None:
        """Configure the backend.

        Args:
            transport: An httpx transport, so tests can inject
                ``httpx.MockTransport`` instead of reaching the network.
            http_referer: Sent as ``HTTP-Referer``; OpenRouter uses it to
                attribute usage. Host projects should pass their own repo URL.
            app_title: Sent as ``X-Title``, likewise.
        """
        self._transport = transport
        self._http_referer = http_referer
        self._app_title = app_title

    @staticmethod
    def reasoning_body(effort: Effort) -> dict[str, Any]:
        """Build the ``reasoning`` field.

        Always sent explicitly: reasoning is ON by default for
        ``anthropic/claude-opus-5`` and ``-sonnet-5`` on OpenRouter, so relying
        on the default would silently diverge from the other backends.

        Args:
            effort: Reasoning depth.

        Returns:
            The value for the request's ``reasoning`` key.
        """
        if effort == "none":
            return {"enabled": False}
        return {"effort": effort}

    def build_body(
        self,
        *,
        prompt: str,
        system: str | None,
        model: str,
        schema: dict[str, Any] | None,
        effort: Effort,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Assemble the request body.

        Args:
            prompt: The user prompt.
            system: System prompt, or None.
            model: OpenRouter model slug.
            schema: JSON Schema for structured output, or None.
            effort: Reasoning depth.
            max_tokens: Output token ceiling.

        Returns:
            The JSON body. Never includes ``tools`` or ``plugins``.
        """
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "reasoning": self.reasoning_body(effort),
        }
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "result",
                    "strict": True,
                    "schema": strict_schema(schema),
                },
            }
            # Without this, OpenRouter may route to a provider that treats the
            # schema as a hint rather than a guarantee.
            body["provider"] = {"require_parameters": True}
        return body

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
            model: OpenRouter model slug.
            schema: JSON Schema for structured output, or None.
            effort: Reasoning depth.
            max_tokens: Output token ceiling.
            timeout_s: Per-call timeout in seconds.

        Returns:
            The completion.

        Raises:
            LLMTimeoutError: If the request times out.
            LLMSchemaError: If a schema was requested but the reply is not JSON.
            LLMError: Or a subclass, on any other failure.
        """
        effort = validate_effort(effort, provider=self.name, supports_none=True)
        api_key = require_env("OPENROUTER_API_KEY", provider=self.name)
        body = self.build_body(
            prompt=prompt,
            system=system,
            model=model,
            schema=schema,
            effort=effort,
            max_tokens=max_tokens,
        )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self._http_referer,
            "X-Title": self._app_title,
        }

        try:
            with httpx.Client(timeout=timeout_s, transport=self._transport) as client:
                response = client.post(_ENDPOINT, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            msg = f"OpenRouter did not respond within {timeout_s}s"
            raise LLMTimeoutError(msg) from exc
        except httpx.RequestError as exc:
            msg = f"could not reach OpenRouter: {exc}"
            raise LLMTransientError(msg) from exc

        if response.status_code >= _HTTP_UNAUTHORIZED:
            raise self._classify(response)

        payload = response.json()
        message = payload["choices"][0]["message"]
        text = message.get("content") or ""

        data = None
        if schema is not None:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                msg = f"expected JSON for the requested schema, got: {text[:300]}"
                raise LLMSchemaError(msg) from exc

        usage: dict[str, Any] = payload.get("usage") or {}
        details: dict[str, Any] = usage.get("completion_tokens_details") or {}
        return LLMResponse(
            text=text,
            data=data,
            provider=self.name,
            model=payload.get("model", model),
            effort=effort,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            thinking_tokens=details.get("reasoning_tokens"),
            num_turns=1,
            cost_usd=usage.get("cost"),
            latency_ms=None,
            raw={
                "id": payload.get("id"),
                "finish_reason": payload["choices"][0].get("finish_reason"),
            },
        )

    @staticmethod
    def _classify(response: httpx.Response) -> Exception:
        """Map an HTTP error response onto this package's exception types.

        Args:
            response: The failed response.

        Returns:
            The exception to raise.
        """
        detail = response.text[:300]
        status = response.status_code
        if status in (_HTTP_UNAUTHORIZED, _HTTP_FORBIDDEN):
            return LLMAuthError(f"OpenRouter rejected the API key ({status}): {detail}")
        if status == _HTTP_TOO_MANY_REQUESTS:
            retry_after = response.headers.get("Retry-After")
            suffix = f" Retry-After: {retry_after}s." if retry_after else ""
            return LLMRateLimitError(f"OpenRouter rate limit reached.{suffix} {detail}")
        if status >= _HTTP_SERVER_ERROR:
            return LLMTransientError(f"OpenRouter server error {status}: {detail}")
        return LLMTransientError(f"OpenRouter returned {status}: {detail}")
