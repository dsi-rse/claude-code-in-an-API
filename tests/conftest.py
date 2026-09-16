"""Shared fixtures. The suite must never need credentials or a network."""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING, Any

import dotenv
import pytest

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator

_CREDENTIAL_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "OPENROUTER_API_KEY",
)
_CONFIG_VARS = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_EFFORT",
    "LLM_MAX_TOKENS",
    "LLM_TIMEOUT_S",
    "LLM_MAX_WORKERS",
    "LLM_MAX_RETRIES",
    "LLM_SCRATCH_DIR",
    "LLM_HTTP_REFERER",
    "LLM_APP_TITLE",
    "CLAUDE_BIN",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every test independent of the developer's own environment and .env.

    Every variable LLMConfig.from_env reads must be listed above, or a
    developer's own environment silently changes what the suite asserts.

    Args:
        monkeypatch: pytest's environment patcher.
    """
    for name in (*_CREDENTIAL_VARS, *_CONFIG_VARS):
        monkeypatch.delenv(name, raising=False)
    # LLMConfig.from_env calls load_dotenv, which searches upward from this
    # package for a .env and would put back everything just deleted. Dotenv
    # discovery is not what these tests are about.
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail loudly if a test tries to open a socket.

    A blunt tripwire, so an accidental live call fails on a laptop rather than
    mysteriously in CI.

    Args:
        monkeypatch: pytest's attribute patcher.

    Yields:
        None.
    """

    def _blocked(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        msg = "tests must not use the network"
        raise RuntimeError(msg)

    monkeypatch.setattr(socket, "socket", _blocked)
    yield
