"""Tests for retry, concurrency and the dataframe helper."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import pytest

from claude_as_api import client as client_module
from claude_as_api.client import LLMClient, map_dataframe
from claude_as_api.core import (
    Effort,
    LLMConfig,
    LLMRateLimitError,
    LLMResponse,
    LLMTransientError,
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"label": {"type": "string"}},
}


class FakeBackend:
    """A backend that answers instantly and counts its calls."""

    name = "fake"
    supports_effort_none = True

    def __init__(
        self, *, fail_times: int = 0, delays: dict[str, float] | None = None
    ) -> None:
        """Configure the fake.

        Args:
            fail_times: How many leading calls should raise a transient error.
            delays: Optional per-prompt sleep, to shuffle completion order.
        """
        self.calls = 0
        self.prompts: list[str] = []
        self.efforts: list[Effort] = []
        self.fail_times = fail_times
        self.delays = delays or {}

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
        """Return a canned response.

        Args:
            prompt: Recorded.
            system: Ignored.
            model: Echoed back.
            schema: Controls whether ``data`` is populated.
            effort: Recorded.
            max_tokens: Ignored.
            timeout_s: Ignored.

        Returns:
            A canned response echoing the prompt.

        Raises:
            LLMTransientError: For the first ``fail_times`` calls.
        """
        del system, max_tokens, timeout_s
        self.calls += 1
        self.prompts.append(prompt)
        self.efforts.append(effort)
        if self.calls <= self.fail_times:
            msg = "temporary"
            raise LLMTransientError(msg)
        time.sleep(self.delays.get(prompt, 0.0))
        return LLMResponse(
            text=prompt,
            data={"label": prompt} if schema else None,
            provider=self.name,
            model=model,
            effort=effort,
            thinking_tokens=7,
            num_turns=1,
        )


def _client(backend: FakeBackend, **config: Any) -> LLMClient:  # noqa: ANN401
    """Build a client wired to a fake backend.

    Args:
        backend: The fake.
        **config: Overrides for LLMConfig.

    Returns:
        A client that touches no network.
    """
    defaults: dict[str, Any] = {
        "provider": "claude_code",
        "model": "fake-model",
        "max_retries": 3,
    }
    defaults.update(config)
    return LLMClient(config=LLMConfig(**defaults), backend=backend)


def test_complete_passes_the_configured_effort() -> None:
    """Effort is always resolved and forwarded, never left implicit."""
    backend = FakeBackend()
    _client(backend, effort="medium").complete("hello")
    assert backend.efforts == ["medium"]


def test_per_call_effort_overrides_config() -> None:
    """A caller can raise reasoning depth for one hard row."""
    backend = FakeBackend()
    _client(backend, effort="low").complete("hello", effort="max")
    assert backend.efforts == ["max"]


def test_transient_failures_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two blips then success should still return a result."""
    monkeypatch.setattr(client_module.time, "sleep", lambda _: None)
    backend = FakeBackend(fail_times=2)
    response = _client(backend).complete("hello")
    assert response.text == "hello"
    assert backend.calls == 3


def test_retries_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A permanently broken backend eventually gives up."""
    monkeypatch.setattr(client_module.time, "sleep", lambda _: None)
    backend = FakeBackend(fail_times=99)
    with pytest.raises(LLMTransientError):
        _client(backend, max_retries=3).complete("hello")
    assert backend.calls == 3


def test_map_prompts_preserves_input_order() -> None:
    """The highest-consequence guarantee in the package.

    Results get zipped back onto dataframe rows, so completion-order results
    would silently mislabel data. The delays make the first prompt finish last.
    """
    prompts = [f"p{i}" for i in range(6)]
    delays = {"p0": 0.05, "p1": 0.04, "p2": 0.03}
    backend = FakeBackend(delays=delays)
    results = _client(backend, max_workers=6).map_prompts(prompts, progress=False)
    assert [r.text for r in results] == prompts


def test_map_prompts_records_errors_by_default() -> None:
    """One bad row must not lose the other 4,999."""
    backend = FakeBackend(fail_times=3)
    results = _client(backend, max_workers=1, max_retries=1).map_prompts(
        ["a", "b", "c", "d"], progress=False
    )
    assert isinstance(results[0], LLMTransientError)
    assert isinstance(results[3], LLMResponse)


def test_map_prompts_can_raise_instead() -> None:
    """on_error='raise' propagates the first failure."""
    backend = FakeBackend(fail_times=1)
    with pytest.raises(LLMTransientError):
        _client(backend, max_workers=1, max_retries=1).map_prompts(
            ["a", "b"], on_error="raise", progress=False
        )


class RateLimitedBackend(FakeBackend):
    """A backend whose quota is exhausted from the second call onward."""

    def generate(self, **kwargs: Any) -> LLMResponse:  # noqa: ANN401
        """Answer once, then report quota exhaustion.

        Args:
            **kwargs: Forwarded to the parent.

        Returns:
            A canned response for the first call.

        Raises:
            LLMRateLimitError: On every call after the first.
        """
        if self.calls >= 1:
            self.calls += 1
            msg = "usage limit reached"
            raise LLMRateLimitError(msg)
        return super().generate(**kwargs)


def test_rate_limit_aborts_the_whole_batch() -> None:
    """Quota exhaustion lasts hours; grinding on would waste the run."""
    backend = RateLimitedBackend()
    with pytest.raises(LLMRateLimitError, match="Batch stopped"):
        _client(backend, max_workers=1).map_prompts(["a", "b", "c"], progress=False)


def test_map_dataframe_expands_schema_properties_into_columns() -> None:
    """The ergonomic entry point students will actually use."""
    frame = pd.DataFrame([{"desc": "broccoli"}, {"desc": "paper towel"}])
    backend = FakeBackend()
    labeled = map_dataframe(
        frame,
        template="{desc}",
        schema=SCHEMA,
        client=_client(backend),
        progress=False,
    )
    assert labeled["label"].tolist() == ["broccoli", "paper towel"]
    assert labeled["llm_error"].isna().all()
    assert labeled["llm_thinking_tokens"].tolist() == [7, 7]


def test_map_dataframe_does_not_mutate_the_input() -> None:
    """Students re-run cells; the source frame must stay clean."""
    frame = pd.DataFrame([{"desc": "broccoli"}])
    map_dataframe(
        frame,
        template="{desc}",
        schema=SCHEMA,
        client=_client(FakeBackend()),
        progress=False,
    )
    assert list(frame.columns) == ["desc"]
