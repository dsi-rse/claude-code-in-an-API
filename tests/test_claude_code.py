"""Tests for the Claude Code subprocess backend.

CANNED below is a real ``--output-format json`` payload captured from
``claude`` v2.1.236, so these tests document the actual wire format.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from claude_as_api import claude_code as backend_module
from claude_as_api.claude_code import ClaudeCodeBackend, child_env
from claude_as_api.core import (
    LLMAuthError,
    LLMRateLimitError,
    LLMSchemaError,
    LLMTimeoutError,
    LLMTransientError,
    LLMUnsupportedError,
)

CANNED: dict[str, Any] = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": '{"food_category":"non_food","confidence":0.96}',
    "structured_output": {"food_category": "non_food", "confidence": 0.96},
    "session_id": "823bb69a-e6bf-42eb-85bd-401dbf3f2566",
    "num_turns": 2,
    "stop_reason": "tool_use",
    "terminal_reason": "completed",
    "total_cost_usd": 0.005003,
    "duration_ms": 4826,
    "duration_api_ms": 5284,
    "api_error_status": None,
    "permission_denials": [],
    "usage": {
        "input_tokens": 2198,
        "output_tokens": 478,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "output_tokens_details": {"thinking_tokens": 371},
        "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
    },
}

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "food_category": {"type": "string"},
        "confidence": {"type": "number"},
    },
}


@pytest.fixture
def backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ClaudeCodeBackend:
    """A backend with the executable lookup stubbed out.

    Args:
        tmp_path: Scratch directory for the CLI's cwd.
        monkeypatch: pytest's attribute patcher.

    Returns:
        A ready backend.
    """
    monkeypatch.setattr(backend_module.shutil, "which", lambda _: "/usr/bin/claude")
    return ClaudeCodeBackend(scratch_dir=tmp_path)


def _fake_run(
    recorder: dict[str, Any], payload: dict[str, Any], returncode: int = 0
) -> Any:  # noqa: ANN401
    """Build a stand-in for ``subprocess.run`` that records its arguments.

    Args:
        recorder: Dict to write the observed call into.
        payload: JSON payload the fake CLI should print.
        returncode: Exit code the fake CLI should report.

    Returns:
        A callable suitable for monkeypatching.
    """

    def _run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:  # noqa: ANN401
        recorder["argv"] = argv
        recorder.update(kwargs)
        return subprocess.CompletedProcess(
            argv, returncode, stdout=json.dumps(payload), stderr=""
        )

    return _run


def _generate(
    backend: ClaudeCodeBackend,
    monkeypatch: pytest.MonkeyPatch,
    recorder: dict[str, Any],
    *,
    payload: dict[str, Any] = CANNED,
    returncode: int = 0,
    **overrides: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Run ``generate`` against a faked subprocess.

    Args:
        backend: The backend under test.
        monkeypatch: pytest's attribute patcher.
        recorder: Dict that receives the observed subprocess call.
        payload: JSON payload the fake CLI prints.
        returncode: Exit code the fake CLI reports.
        **overrides: Overrides for the generate() keyword arguments.

    Returns:
        The LLMResponse.
    """
    monkeypatch.setattr(
        backend_module.subprocess, "run", _fake_run(recorder, payload, returncode)
    )
    kwargs: dict[str, Any] = {
        "prompt": "PPR TWL 2PLY 30RL",
        "system": "Classify the line item.",
        "model": "opus",
        "schema": SCHEMA,
        "effort": "low",
        "max_tokens": 4096,
        "timeout_s": 30.0,
    }
    kwargs.update(overrides)
    return backend.generate(**kwargs)


def test_never_passes_bare(backend: ClaudeCodeBackend) -> None:
    """--bare would switch auth to API-key-only and break every Enterprise seat.

    Anthropic's own headless-CI docs recommend --bare, so this guard exists to
    stop a well-meaning future edit from adding it.
    """
    argv = backend.build_argv(model="opus", effort="low", system=None, schema=None)
    assert "--bare" not in argv


def test_disables_all_tools(backend: ClaudeCodeBackend) -> None:
    """An empty --tools value is what blocks web search and the filesystem."""
    argv = backend.build_argv(model="opus", effort="low", system=None, schema=None)
    assert argv[argv.index("--tools") + 1] == ""


def test_always_sends_explicit_effort(backend: ClaudeCodeBackend) -> None:
    """Reasoning depth is never left to the CLI default."""
    argv = backend.build_argv(model="opus", effort="medium", system=None, schema=None)
    assert argv[argv.index("--effort") + 1] == "medium"


def test_isolates_from_local_configuration(backend: ClaudeCodeBackend) -> None:
    """Settings files and MCP servers must not leak into results."""
    argv = backend.build_argv(model="opus", effort="low", system=None, schema=None)
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in argv


def test_schema_is_tightened_before_sending(backend: ClaudeCodeBackend) -> None:
    """The schema goes out with additionalProperties disabled."""
    argv = backend.build_argv(model="opus", effort="low", system=None, schema=SCHEMA)
    sent = json.loads(argv[argv.index("--json-schema") + 1])
    assert sent["additionalProperties"] is False
    assert sorted(sent["required"]) == ["confidence", "food_category"]


def test_prompt_goes_to_stdin_not_argv(
    backend: ClaudeCodeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Product descriptions contain quotes and $; argv would be a minefield."""
    recorder: dict[str, Any] = {}
    _generate(backend, monkeypatch, recorder, prompt='ORG "BRSSL" $5 #BAG')
    assert recorder["input"] == 'ORG "BRSSL" $5 #BAG'
    assert 'ORG "BRSSL" $5 #BAG' not in recorder["argv"]


def test_parses_a_successful_response(
    backend: ClaudeCodeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Structured output and the parity signals are all surfaced."""
    response = _generate(backend, monkeypatch, {})
    assert response.data == {"food_category": "non_food", "confidence": 0.96}
    assert response.thinking_tokens == 371
    assert response.num_turns == 2
    assert response.input_tokens == 2198
    assert response.cost_usd == pytest.approx(0.005003)
    assert response.effort == "low"


def test_rate_limit_is_classified(
    backend: ClaudeCodeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 429 must not be retried, so it needs its own type."""
    payload = {
        **CANNED,
        "is_error": True,
        "api_error_status": 429,
        "result": "rate limited",
    }
    with pytest.raises(LLMRateLimitError):
        _generate(backend, monkeypatch, {}, payload=payload)


def test_usage_limit_text_is_classified_as_rate_limit(
    backend: ClaudeCodeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Subscription exhaustion reports as text, not as a status code."""
    payload = {**CANNED, "is_error": True, "result": "Claude usage limit reached"}
    with pytest.raises(LLMRateLimitError):
        _generate(backend, monkeypatch, {}, payload=payload)


def test_auth_failure_is_classified(
    backend: ClaudeCodeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auth errors are never retried."""
    payload = {
        **CANNED,
        "is_error": True,
        "api_error_status": 401,
        "result": "not logged in",
    }
    with pytest.raises(LLMAuthError):
        _generate(backend, monkeypatch, {}, payload=payload)


def test_nonzero_exit_is_transient(
    backend: ClaudeCodeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unclassified crash is worth one more try."""
    with pytest.raises(LLMTransientError):
        _generate(backend, monkeypatch, {}, payload={"result": "boom"}, returncode=1)


def test_missing_structured_output_is_a_schema_error(
    backend: ClaudeCodeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A schema was requested but not honoured: a prompt bug, not a blip."""
    payload = {k: v for k, v in CANNED.items() if k != "structured_output"}
    with pytest.raises(LLMSchemaError):
        _generate(backend, monkeypatch, {}, payload=payload)


def test_unparseable_output_is_transient(
    backend: ClaudeCodeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Garbage on stdout should retry rather than crash the batch."""

    def _run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:  # noqa: ANN401, ARG001
        return subprocess.CompletedProcess(
            argv, 0, stdout="not json", stderr="segfault"
        )

    monkeypatch.setattr(backend_module.subprocess, "run", _run)
    with pytest.raises(LLMTransientError, match="segfault"):
        backend.generate(
            prompt="x",
            system=None,
            model="opus",
            schema=None,
            effort="low",
            max_tokens=10,
            timeout_s=1.0,
        )


def test_timeout_is_classified(
    backend: ClaudeCodeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hung CLI raises the retryable timeout type."""

    def _run(argv: list[str], **kwargs: Any) -> None:  # noqa: ANN401, ARG001
        raise subprocess.TimeoutExpired(argv, 1.0)

    monkeypatch.setattr(backend_module.subprocess, "run", _run)
    with pytest.raises(LLMTimeoutError):
        backend.generate(
            prompt="x",
            system=None,
            model="opus",
            schema=None,
            effort="low",
            max_tokens=10,
            timeout_s=1.0,
        )


def test_effort_none_is_rejected(backend: ClaudeCodeBackend) -> None:
    """The CLI has no way to disable reasoning; fail loudly rather than clamp."""
    with pytest.raises(LLMUnsupportedError):
        backend.generate(
            prompt="x",
            system=None,
            model="opus",
            schema=None,
            effort="none",
            max_tokens=10,
            timeout_s=1.0,
        )


def test_missing_cli_gives_an_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Better than a bare FileNotFoundError."""
    monkeypatch.setattr(backend_module.shutil, "which", lambda _: None)
    with pytest.raises(LLMAuthError, match="not on PATH"):
        ClaudeCodeBackend().build_argv(
            model="opus", effort="low", system=None, schema=None
        )


def test_child_env_strips_session_vars_but_keeps_the_oauth_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of the denylist.

    A parent Claude Code session exports CLAUDE_EFFORT and friends; inheriting
    them would make the pipeline behave differently when launched from inside
    Claude Code. But a blanket CLAUDE* wipe would delete the OAuth token and
    break Docker, so the two cases are tested together.
    """
    monkeypatch.setenv("CLAUDE_EFFORT", "max")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "abc")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-secret")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = child_env()

    assert "CLAUDE_EFFORT" not in env
    assert "CLAUDECODE" not in env
    assert "CLAUDE_CODE_SESSION_ID" not in env
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat-secret"
    assert env["PATH"] == "/usr/bin"


def test_generate_uses_the_scrubbed_env(
    backend: ClaudeCodeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards against a future 'simplification' back to os.environ.copy()."""
    monkeypatch.setenv("CLAUDE_EFFORT", "max")
    recorder: dict[str, Any] = {}
    _generate(backend, monkeypatch, recorder)
    assert "CLAUDE_EFFORT" not in recorder["env"]


def test_scratch_dir_defaults_under_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default scratch dir follows the caller's cwd, not this file.

    It must never be derived from ``__file__``: this package is installed - as
    a submodule, a path dependency, or into site-packages - so its own location
    says nothing about where the caller's project is.
    """
    monkeypatch.setattr(backend_module.shutil, "which", lambda _: "/usr/bin/claude")
    monkeypatch.chdir(tmp_path)
    resolved = ClaudeCodeBackend()._scratch()  # noqa: SLF001
    assert resolved == (tmp_path / ".llm_scratch").resolve()
    assert resolved.is_dir()
    assert Path(backend_module.__file__).parent not in resolved.parents


def test_scratch_dir_honours_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LLM_SCRATCH_DIR is the escape hatch for real isolation."""
    monkeypatch.setattr(backend_module.shutil, "which", lambda _: "/usr/bin/claude")
    target = tmp_path / "somewhere" / "else"
    monkeypatch.setenv("LLM_SCRATCH_DIR", str(target))
    assert ClaudeCodeBackend()._scratch() == target.resolve()  # noqa: SLF001


def test_explicit_scratch_dir_beats_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An injected directory wins, which is what the tests above rely on."""
    monkeypatch.setattr(backend_module.shutil, "which", lambda _: "/usr/bin/claude")
    monkeypatch.setenv("LLM_SCRATCH_DIR", str(tmp_path / "ignored"))
    explicit = tmp_path / "chosen"
    assert ClaudeCodeBackend(scratch_dir=explicit)._scratch() == explicit.resolve()  # noqa: SLF001
