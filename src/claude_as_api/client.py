"""Provider-agnostic entry point for LLM calls.

Calling code should import from :mod:`claude_as_api` rather than from a backend
module directly, so that switching providers is a change to ``.env`` and
nothing else.
"""

from __future__ import annotations

import secrets
import sys
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from .core import (
    PROVIDERS,
    Effort,
    LLMBackend,
    LLMConfig,
    LLMError,
    LLMRateLimitError,
    LLMResponse,
    LLMTimeoutError,
    LLMTransientError,
    LLMUnsupportedError,
)

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

_BACKOFF_BASE_S: Final = 1.5
_BACKOFF_MAX_S: Final = 60.0
_PROGRESS_EVERY: Final = 25


def build_backend(config: LLMConfig) -> LLMBackend:
    """Instantiate the backend named by a config.

    Backend modules are imported lazily so that an uninstalled optional
    dependency only matters if you actually select that provider.

    Args:
        config: The resolved configuration.

    Returns:
        A ready backend.

    Raises:
        LLMUnsupportedError: If the provider name is not recognised.
    """
    if config.provider == "claude_code":
        from .claude_code import ClaudeCodeBackend

        return ClaudeCodeBackend(
            claude_bin=config.claude_bin,
            scratch_dir=Path(config.scratch_dir) if config.scratch_dir else None,
        )
    if config.provider == "anthropic":
        from .anthropic_api import AnthropicBackend

        return AnthropicBackend()
    if config.provider == "openrouter":
        from .openrouter import OpenRouterBackend

        return OpenRouterBackend(
            http_referer=config.http_referer, app_title=config.app_title
        )

    msg = (
        f"unknown provider {config.provider!r}; expected one of {', '.join(PROVIDERS)}"
    )
    raise LLMUnsupportedError(msg)


class LLMClient:
    """Adds retry, concurrency and dataframe helpers on top of a backend."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        backend: LLMBackend | None = None,
    ) -> None:
        """Configure the client.

        Args:
            config: Configuration; defaults to one built from the
                environment by :meth:`LLMConfig.from_env`.
            backend: An explicit backend, mainly for tests. Defaults to the one
                named by ``config.provider``.
        """
        self.config = config or LLMConfig.from_env()
        self.backend = backend or build_backend(self.config)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        schema: dict[str, Any] | None = None,
        effort: Effort | None = None,
        max_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        """Run one completion, retrying transient failures.

        Args:
            prompt: The user prompt.
            system: System prompt, or None.
            model: Model override; defaults to the provider's configured model.
            schema: JSON Schema for structured output, or None.
            effort: Reasoning depth override; defaults to the configured level.
            max_tokens: Output token ceiling override.
            timeout_s: Per-call timeout override.

        Returns:
            The completion.

        Raises:
            LLMError: Or a subclass, if every attempt fails.
        """
        resolved = {
            "prompt": prompt,
            "system": system,
            "model": self.config.resolve_model(model),
            "schema": schema,
            "effort": effort or self.config.effort,
            "max_tokens": max_tokens or self.config.max_tokens,
            "timeout_s": timeout_s or self.config.timeout_s,
        }

        last: LLMError | None = None
        for attempt in range(max(1, self.config.max_retries)):
            try:
                return self.backend.generate(**resolved)
            except (LLMTimeoutError, LLMTransientError) as exc:
                last = exc
                if attempt + 1 < self.config.max_retries:
                    time.sleep(self._backoff(attempt))
        raise last if last is not None else LLMError("no attempts were made")

    @staticmethod
    def _backoff(attempt: int) -> float:
        """Exponential backoff with jitter.

        Args:
            attempt: Zero-based attempt number.

        Returns:
            Seconds to sleep.
        """
        jitter = secrets.SystemRandom().uniform(0.0, 0.5)
        return min(_BACKOFF_MAX_S, _BACKOFF_BASE_S * (2**attempt)) + jitter

    def complete_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        **kwargs: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        """Run one completion and return only the parsed structured output.

        Args:
            prompt: The user prompt.
            schema: JSON Schema the answer must satisfy.
            **kwargs: Passed through to :meth:`complete`.

        Returns:
            The parsed object.
        """
        response = self.complete(prompt, schema=schema, **kwargs)
        return response.data or {}

    def map_prompts(
        self,
        prompts: Sequence[str],
        *,
        max_workers: int | None = None,
        on_error: str = "record",
        progress: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> list[LLMResponse | LLMError]:
        """Run many prompts concurrently, preserving input order.

        Results are returned in the order the prompts were given, never in
        completion order. Callers zip these back onto dataframe rows, so
        reordering would silently mislabel data.

        A rate-limit error aborts the whole batch: on a subscription, quota
        exhaustion lasts hours, so the remaining calls would all fail too.

        Args:
            prompts: The prompts to run.
            max_workers: Thread count override.
            on_error: "record" to place exceptions in the result list, or
                "raise" to propagate the first failure.
            progress: Whether to print a counter to stderr.
            **kwargs: Passed through to :meth:`complete`.

        Returns:
            One entry per prompt, in input order.

        Raises:
            LLMRateLimitError: If the provider's quota is exhausted mid-batch.
            LLMError: If ``on_error="raise"`` and any call fails.
        """
        workers = max_workers or self.config.max_workers
        results: list[LLMResponse | LLMError] = []
        done = 0

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures: list[Future[LLMResponse]] = [
                pool.submit(self.complete, prompt, **kwargs) for prompt in prompts
            ]
            for index, future in enumerate(futures):
                try:
                    results.append(future.result())
                except LLMRateLimitError as exc:
                    for pending in futures[index + 1 :]:
                        pending.cancel()
                    msg = (
                        f"{exc} Batch stopped after {index} of {len(prompts)} prompts; "
                        "re-run once the quota window resets."
                    )
                    raise LLMRateLimitError(msg) from exc
                except LLMError as exc:
                    if on_error == "raise":
                        for pending in futures[index + 1 :]:
                            pending.cancel()
                        raise
                    results.append(exc)
                done += 1
                if progress and (done % _PROGRESS_EVERY == 0 or done == len(prompts)):
                    print(f"llm: {done}/{len(prompts)}", file=sys.stderr)

        return results


def map_dataframe(
    frame: pd.DataFrame,
    *,
    template: str,
    schema: dict[str, Any],
    client: LLMClient | None = None,
    prefix: str = "",
    **kwargs: Any,  # noqa: ANN401
) -> pd.DataFrame:
    """Run one LLM call per row and expand the results into new columns.

    Args:
        frame: Input rows. Not modified.
        template: A ``str.format`` template referencing column names.
        schema: JSON Schema; each top-level property becomes a column.
        client: Client to use; defaults to a new one from the environment.
        prefix: Optional prefix for the generated column names.
        **kwargs: Passed through to :meth:`LLMClient.map_prompts`.

    Returns:
        A copy of ``frame`` with one column per schema property, plus an
        ``llm_error`` column holding the error string for failed rows.
    """
    active = client or LLMClient()
    prompts = [template.format(**row) for row in frame.to_dict(orient="records")]
    results = active.map_prompts(prompts, schema=schema, **kwargs)

    columns = list(schema.get("properties", {}))
    out = frame.copy()
    for column in columns:
        out[f"{prefix}{column}"] = [
            r.data.get(column) if isinstance(r, LLMResponse) and r.data else None
            for r in results
        ]
    out["llm_error"] = [str(r) if isinstance(r, LLMError) else None for r in results]
    # Parity signals, deliberately surfaced in the frame rather than buried in
    # a log: a pipeline burning hundreds of thinking tokens per row is leaning
    # on Claude's reasoning and may not port to a bare LLM.
    out["llm_thinking_tokens"] = [
        r.thinking_tokens if isinstance(r, LLMResponse) else None for r in results
    ]
    out["llm_num_turns"] = [
        r.num_turns if isinstance(r, LLMResponse) else None for r in results
    ]
    return out
