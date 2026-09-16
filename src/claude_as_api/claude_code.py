"""Backend that drives the Claude Code CLI as a plain completion API.

This is the default backend, and the only one that works with a Claude
Enterprise (claude.ai) seat: an Enterprise subscription does not include
Anthropic API access, but the ``claude`` CLI authenticates against it directly.

The CLI is an agent, so this module works hard to make it behave like a bare
LLM - all tools off, settings and MCP servers ignored, an empty working
directory, and an explicit reasoning level on every call. See
the README for why that matters.

Limitations, all inherent to the CLI: no ``temperature``, no ``seed``, no
direct ``max_tokens``, and no way to disable reasoning entirely.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Final

from .core import (
    Effort,
    LLMAuthError,
    LLMRateLimitError,
    LLMResponse,
    LLMSchemaError,
    LLMTimeoutError,
    LLMTransientError,
    strict_schema,
    validate_effort,
)

_HTTP_UNAUTHORIZED: Final = 401
_HTTP_FORBIDDEN: Final = 403
_HTTP_TOO_MANY_REQUESTS: Final = 429
_HTTP_SERVER_ERROR: Final = 500

_RATE_LIMIT_RE: Final = re.compile(
    r"usage limit|rate limit|quota|too many requests", re.I
)
_AUTH_RE: Final = re.compile(
    r"not logged in|log ?in|authenticat|invalid api key|credential", re.I
)

#: Variables a parent Claude Code session exports into its children. Inheriting
#: them would make this pipeline behave differently depending on whether it was
#: launched from inside Claude Code or from a plain terminal - CLAUDE_EFFORT in
#: particular would silently override our own reasoning setting.
#:
#: This is a denylist and must stay one. A "CLAUDE*" prefix wipe would also
#: delete CLAUDE_CODE_OAUTH_TOKEN and break the Docker path entirely.
_SESSION_ENV: Final = frozenset(
    {
        "AI_AGENT",
        "CLAUDECODE",
        "CLAUDE_CODE_CHILD_SESSION",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_EXECPATH",
        "CLAUDE_CODE_MESSAGING_SOCKET",
        "CLAUDE_CODE_MESSAGING_TOKEN",
        "CLAUDE_CODE_SESSION_ID",
        "CLAUDE_EFFORT",
        "CLAUDE_PID",
    }
)


def child_env() -> dict[str, str]:
    """Build the environment for the CLI subprocess.

    Strips the parent Claude Code session's own variables while keeping
    everything else - PATH, HOME, CLAUDE_CODE_OAUTH_TOKEN and ANTHROPIC_* all
    survive. On macOS the Enterprise credential lives in the login keychain and
    needs the ambient environment, so this is a denylist, not an allowlist.

    Returns:
        A copy of the environment with session variables removed.
    """
    return {k: v for k, v in os.environ.items() if k not in _SESSION_ENV}


class ClaudeCodeBackend:
    """Runs ``claude -p`` as a subprocess and parses its JSON output."""

    name = "claude_code"
    #: The CLI rejects `--effort none` outright: "Valid values: low, medium,
    #: high, xhigh, max". `low` is the practical equivalent on Opus.
    supports_effort_none = False

    def __init__(
        self, claude_bin: str = "claude", scratch_dir: Path | None = None
    ) -> None:
        """Configure the backend.

        Args:
            claude_bin: Name or path of the CLI executable.
            scratch_dir: Empty directory to run the CLI in. Defaults to
                ``$LLM_SCRATCH_DIR``, or ``.llm_scratch`` under the current
                working directory.
        """
        self.claude_bin = claude_bin
        self._scratch_dir = scratch_dir

    def _resolve_bin(self) -> str:
        """Locate the CLI on PATH.

        Returns:
            An absolute path to the executable.

        Raises:
            LLMAuthError: If it is not installed.
        """
        found = shutil.which(self.claude_bin)
        if found is None:
            msg = (
                f"the {self.claude_bin!r} CLI is not on PATH. Install Claude Code "
                "(https://claude.com/claude-code) and run `claude` once to log in."
            )
            raise LLMAuthError(msg)
        return found

    def _scratch(self) -> Path:
        """Return an empty working directory for the CLI.

        The CLI is run from a directory of our choosing rather than the
        caller's cwd, so that a stray CLAUDE.md, .claude/ or project file next
        to the caller cannot change the answer.

        Note this must not be derived from ``__file__``: this package is
        installed - as a submodule, a path dependency or into site-packages -
        so its own location says nothing about where the caller's project is.

        ``.llm_scratch`` under the cwd is usually a project subdirectory, and
        Claude Code searches *parent* directories for CLAUDE.md, so this is
        cheap isolation rather than total isolation. For total isolation, point
        LLM_SCRATCH_DIR at a directory outside any project, e.g. a temp dir.

        Returns:
            A directory that exists.
        """
        directory = self._scratch_dir or Path(
            os.environ.get("LLM_SCRATCH_DIR") or ".llm_scratch"
        )
        directory = directory.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def build_argv(
        self,
        *,
        model: str,
        effort: Effort,
        system: str | None,
        schema: dict[str, Any] | None,
    ) -> list[str]:
        """Assemble the command line.

        Split out from :meth:`generate` so tests can assert on it without
        running anything.

        Args:
            model: Model alias or full id.
            effort: Reasoning depth; never "none" here.
            system: System prompt, or None.
            schema: JSON Schema for structured output, or None.

        Returns:
            The full argv, starting with the resolved executable.
        """
        # Never add --bare. In bare mode the CLI reads auth strictly from
        # ANTHROPIC_API_KEY or apiKeyHelper and never touches OAuth or the
        # keychain, which breaks every Claude Enterprise seat. Anthropic's own
        # headless-CI docs recommend it; that advice assumes an API key.
        argv = [self._resolve_bin(), "-p"]
        argv += ["--model", model]
        argv += ["--effort", effort]
        argv += ["--output-format", "json"]
        argv += ["--tools", ""]  # disables ALL tools, including web search
        argv += ["--setting-sources", ""]  # ignore user/project/local settings
        argv += ["--strict-mcp-config"]  # ignore any configured MCP servers
        argv += ["--disable-slash-commands", "--no-session-persistence"]
        if system is not None:
            # Replace rather than append: the default agent prompt is ~1-2k
            # input tokens even with tools off, and only replacing removes it.
            argv += ["--system-prompt", system]
        if schema is not None:
            argv += ["--json-schema", json.dumps(strict_schema(schema), sort_keys=True)]
        return argv

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
        """Run one completion through the CLI.

        Args:
            prompt: The user prompt, passed on stdin.
            system: System prompt, or None.
            model: Model alias or full id.
            schema: JSON Schema for structured output, or None.
            effort: Reasoning depth.
            max_tokens: Ignored - the CLI has no equivalent flag.
            timeout_s: Per-call timeout in seconds.

        Returns:
            The completion.

        Raises:
            LLMTimeoutError: If the CLI does not finish in time.
            LLMError: Or a subclass, on any other failure.
        """
        del max_tokens  # no CLI equivalent; documented in the module docstring
        effort = validate_effort(
            effort, provider=self.name, supports_none=self.supports_effort_none
        )
        argv = self.build_argv(model=model, effort=effort, system=system, schema=schema)

        try:
            # argv is built entirely from validated configuration above; no
            # caller data reaches it (the prompt goes to stdin).
            completed = subprocess.run(  # noqa: S603
                argv,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
                cwd=self._scratch(),
                env=child_env(),
            )
        except subprocess.TimeoutExpired as exc:
            msg = f"claude did not respond within {timeout_s}s"
            raise LLMTimeoutError(msg) from exc

        return self._parse(completed, model=model, effort=effort, schema=schema)

    def _parse(
        self,
        completed: subprocess.CompletedProcess[str],
        *,
        model: str,
        effort: Effort,
        schema: dict[str, Any] | None,
    ) -> LLMResponse:
        """Turn the CLI's JSON output into an :class:`LLMResponse`.

        Args:
            completed: The finished subprocess.
            model: Model that was requested.
            effort: Effort that was requested.
            schema: Schema that was requested, or None.

        Returns:
            The completion.

        Raises:
            LLMError: Or a subclass, if the CLI reported failure.
        """
        try:
            payload: dict[str, Any] = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            detail = (completed.stderr or completed.stdout or "").strip()[:500]
            msg = (
                f"could not parse claude output (exit {completed.returncode}): {detail}"
            )
            raise LLMTransientError(msg) from exc

        if payload.get("is_error") or completed.returncode != 0:
            raise self._classify(payload, completed.returncode)

        data = None
        if schema is not None:
            if "structured_output" not in payload:
                msg = (
                    "claude returned no structured_output for the requested schema. "
                    f"Raw result: {str(payload.get('result'))[:300]}"
                )
                raise LLMSchemaError(msg)
            data = payload["structured_output"]

        usage: dict[str, Any] = payload.get("usage") or {}
        details: dict[str, Any] = usage.get("output_tokens_details") or {}
        raw = {k: v for k, v in payload.items() if k != "result"}

        return LLMResponse(
            text=payload.get("result") or "",
            data=data,
            provider=self.name,
            model=model,
            effort=effort,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            thinking_tokens=details.get("thinking_tokens"),
            num_turns=payload.get("num_turns"),
            cost_usd=payload.get("total_cost_usd"),
            latency_ms=payload.get("duration_ms"),
            raw=raw,
        )

    @staticmethod
    def _classify(payload: dict[str, Any], returncode: int) -> Exception:
        """Map a failed CLI run onto the right exception type.

        Args:
            payload: The parsed JSON output.
            returncode: The process exit code.

        Returns:
            The exception to raise.
        """
        status = payload.get("api_error_status")
        text = " ".join(
            str(payload.get(key) or "")
            for key in ("result", "terminal_reason", "subtype")
        )
        detail = text.strip()[:300] or f"exit {returncode}"

        if status == _HTTP_TOO_MANY_REQUESTS or _RATE_LIMIT_RE.search(text):
            return LLMRateLimitError(
                f"Claude rate limit or plan quota reached: {detail}. "
                "This is not retryable; wait for the quota window to reset."
            )
        if status in (_HTTP_UNAUTHORIZED, _HTTP_FORBIDDEN) or _AUTH_RE.search(text):
            return LLMAuthError(
                f"Claude rejected the credentials: {detail}. "
                "Run `claude` once to log in, or set CLAUDE_CODE_OAUTH_TOKEN."
            )
        if isinstance(status, int) and status >= _HTTP_SERVER_ERROR:
            return LLMTransientError(f"Claude server error {status}: {detail}")
        return LLMTransientError(f"claude failed: {detail}")
