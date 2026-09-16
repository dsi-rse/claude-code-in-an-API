# claude-code-in-an-API

UChicago students have Claude Enterprise accounts, but sometimes they need an
API. This wraps Claude Code in one.

```python
from claude_as_api import complete_json

result = complete_json(
    "Product description: ORG BRSSL SPRTS 5# BAG\nClassify this line item.",
    schema={
        "type": "object",
        "properties": {
            "food_category": {"type": "string", "enum": ["produce", "meat", "non_food"]},
            "confidence": {"type": "number"},
        },
    },
)
# -> {"food_category": "produce", "confidence": 0.93}
```

No API key. Your code calls one interface — `claude_as_api` — and the provider
behind it is a setting in the environment. Write your code once now, switch
providers later.

> **What this is not:** an HTTP server. "API" here means a function-call API.
> There is no endpoint to `curl`; you `import` it and call it.

---

## 1. Which credential do you have?

| You have | Set `LLM_PROVIDER=` | Notes |
|---|---|---|
| A **UChicago Claude Enterprise seat** | `claude_code` (default) | Works today, no API key. This is what most people have. |
| A **console.anthropic.com API key** | `anthropic` | Also needs the `anthropic` extra installed. |
| An **OpenRouter API key** | `openrouter` | |

> **A Claude Enterprise seat is not Anthropic API access.** The Claude apps
> subscription and the developer platform are separate products with separate
> billing, and there is no Enterprise admin setting that hands out API keys.
> What an Enterprise seat *does* give you is the `claude` CLI, and the
> `claude_code` backend drives that CLI as a plain completion API.

## 2. Installing it

### As a submodule of your project (the usual case)

```bash
git submodule add https://github.com/dsi-rse/claude-code-in-an-API.git \
    external/claude-code-in-an-API
```

Then depend on it by path. With `uv`, in your project's `pyproject.toml`:

```toml
[project]
dependencies = ["claude-as-api"]

[project.optional-dependencies]
# optional: re-expose the anthropic backend to your own users
anthropic = ["claude-as-api[anthropic]"]

[tool.uv.sources]
claude-as-api = { path = "external/claude-code-in-an-API", editable = true }
```

`uv sync`, and `import claude_as_api` works.

Anyone cloning your project afterwards needs the submodule contents, which a
plain `git clone` does not fetch:

```bash
git submodule update --init --recursive
```

**To take updates**, pull inside the submodule and commit the new pointer in
your project. Everyone who updates then gets the same version:

```bash
git -C external/claude-code-in-an-API pull origin main
git add external/claude-code-in-an-API
git commit -m "bump claude-code-in-an-API"
```

That commit hash is the pin. Nothing changes under anyone's feet until someone
deliberately bumps it.

### Standalone

```bash
git clone https://github.com/dsi-rse/claude-code-in-an-API.git
cd claude-code-in-an-API
uv sync                       # add --extra anthropic if you need that backend
uv run python examples/classify_products.py
```

### Optional extras

| Extra | For |
|---|---|
| `anthropic` | `LLM_PROVIDER=anthropic`. The SDK is imported lazily, so it only matters if you select that backend. |
| `pandas` | `map_dataframe`. The library imports pandas only for type checking, so it is not a runtime requirement of the other functions. |

## 3. Setup on your own machine

```bash
claude auth status        # should report subscriptionType: enterprise
cp .env.example .env      # everything in it is optional
```

If `claude auth status` says you are not logged in, run `claude` once and
follow the browser prompt. No API key, no `ANTHROPIC_API_KEY`, nothing else.

## 4. Setup inside Docker

There is no browser in a container, so authenticate with a token instead.
**On the host**, once:

```bash
claude setup-token        # prints a token valid for one year
```

Pass it into the container as `CLAUDE_CODE_OAUTH_TOKEN`, and install the CLI in
your image:

```dockerfile
RUN curl -fsSL https://claude.ai/install.sh | bash \
 && ln -s /root/.local/bin/claude /usr/local/bin/claude
```

```yaml
environment:
  CLAUDE_CODE_OAUTH_TOKEN: ${CLAUDE_CODE_OAUTH_TOKEN:-}
  LLM_PROVIDER: ${LLM_PROVIDER:-claude_code}
  LLM_EFFORT: ${LLM_EFFORT:-low}
```

> That token is a real credential. `.env` is gitignored, but notebook output
> cells can print `os.environ` — do not commit notebook outputs.

## 5. The API

Everything comes from `claude_as_api`:

| Function | Use it for |
|---|---|
| `complete(prompt, ...)` | One call, returns an `LLMResponse`. |
| `complete_json(prompt, schema=...)` | One call, returns the parsed dict. |
| `map_prompts(prompts, ...)` | Many calls, concurrent, **results in input order**. |
| `map_dataframe(frame, template=..., schema=...)` | One call per row, results expanded into columns. |

`map_dataframe` is the one you usually want:

```python
import pandas as pd
from claude_as_api import map_dataframe

SCHEMA = {
    "type": "object",
    "properties": {
        "food_category": {"type": "string", "enum": ["produce", "meat", "non_food"]},
        "normalized_name": {"type": "string"},
        "confidence": {"type": "number"},
    },
}

labeled = map_dataframe(
    products,
    template="Product description: {product_description}\nSupplier: {supplier}\n\nClassify.",
    schema=SCHEMA,
    system="You classify food-procurement line items. JSON only.",
)
```

You get back a **copy** of the frame with one new column per schema property,
plus `llm_error`, `llm_thinking_tokens` and `llm_num_turns`.

Rows that fail do not sink the run: the error text lands in `llm_error` and the
other rows still come back. The one exception is hitting your plan quota, which
aborts the whole batch on purpose — see the pitfalls below.

For more control, build a client yourself:

```python
from claude_as_api import LLMClient, LLMConfig

client = LLMClient(config=LLMConfig(provider="openrouter", effort="none"))
```

`LLMConfig()` takes explicit values; `LLMConfig.from_env()` reads them from the
environment and `.env`. A bare `LLMClient()` uses `from_env()`.

## 6. Configuration

Every setting has an environment variable, read by `LLMConfig.from_env()`.
See [`.env.example`](.env.example) for the annotated list.

| Variable | Default | |
|---|---|---|
| `LLM_PROVIDER` | `claude_code` | `claude_code`, `anthropic`, `openrouter` |
| `LLM_MODEL` | per provider | Slug spellings differ — see the table in §8 |
| `LLM_EFFORT` | `low` | `none`…`max`; `none` unavailable on `claude_code` |
| `LLM_MAX_TOKENS` | `4096` | Ignored by `claude_code`, which has no such flag |
| `LLM_TIMEOUT_S` | `180` | Per call |
| `LLM_MAX_WORKERS` | `4` | Concurrency for `map_prompts` |
| `LLM_MAX_RETRIES` | `3` | Transient failures and timeouts only |
| `CLAUDE_BIN` | `claude` | Name or path of the CLI |
| `LLM_SCRATCH_DIR` | `./.llm_scratch` | Directory the CLI is run from |
| `LLM_HTTP_REFERER` / `LLM_APP_TITLE` | this repo | OpenRouter usage attribution — set to your own project |

## 7. Reasoning, tools, and whether your work will still run next term

This is the part worth reading twice.

The `claude_code` backend is an **agent** — Claude Code — pretending to be a
plain LLM. A bare model on OpenRouter is not an agent. If you build a workflow
that quietly leans on Claude's reasoning or its ability to look things up, it
will get quietly worse when you switch providers. The harness is built to stop
that from happening silently.

**Tools are off, on every backend.** No web search, no file access, no shell.
This is enforced, not requested: `claude -p --tools ""` really does refuse
("No web search tool is available in this session"), and the other two backends
never send a `tools` array. There is no flag to turn this on. If you ever need
an agent, that should be a deliberate decision someone makes, not something
that creeps into a preprocessing script.

**Reasoning depth is one setting, `LLM_EFFORT`, mapped onto all three backends:**

| `LLM_EFFORT` | `claude_code` | `anthropic` | `openrouter` |
|---|---|---|---|
| `none` | **not available** — raises `LLMUnsupportedError` | `thinking: disabled` | `reasoning: {enabled: false}` |
| `low` … `max` | `--effort X` | `output_config.effort` | `reasoning: {effort: X}` |

Two honest caveats:

- **You cannot turn reasoning off on `claude_code`.** The CLI rejects
  `--effort none` outright. `low` is the closest you can get.
- **`--effort` is a weak knob there.** Measured on Opus 5 with the same prompt
  four times per level, median thinking tokens were 190 (`low`), 150 (`high`),
  255 (`max`) — that is noise, not a trend. On Haiku 4.5 it does nothing at
  all, because that model has no effort parameter. Do not assume `low` buys you
  a non-reasoning model.

So don't trust the setting — **watch the number**. Every response carries
`thinking_tokens`, and `map_dataframe` puts it in a column:

```python
print(labeled["llm_thinking_tokens"].describe())
```

If that is consistently near zero, your prompts are doing the work and they
will port. If it is in the hundreds, Claude is reasoning its way to the answer
and a bare model may not.

**The rule of thumb:** a prompt that would fail if you handed it to a plain LLM
with no tools and no scratchpad is a prompt that will not survive the move to
another provider. Write prompts that are self-contained — put the categories,
the rules and the edge cases *in the prompt*, rather than expecting the model
to work them out.

**Before you actually switch providers**, check agreement on a labeled sample:

```python
from claude_as_api import LLMClient, LLMConfig

a = LLMClient(config=LLMConfig(provider="claude_code"))
b = LLMClient(config=LLMConfig(provider="openrouter"))

prompts = [TEMPLATE.format(**row) for row in sample.to_dict(orient="records")]
ra = a.map_prompts(prompts, schema=SCHEMA, system=SYSTEM, progress=False)
rb = b.map_prompts(prompts, schema=SCHEMA, system=SYSTEM, progress=False)

agree = sum(x.data == y.data for x, y in zip(ra, rb))
print(f"{agree}/{len(prompts)} agree")
```

## 8. Switching providers

One line in `.env`:

```
LLM_PROVIDER=openrouter
```

Model slugs are spelled differently by each provider, so if you also pin
`LLM_MODEL`, update it too:

| Provider | Opus 5 | Haiku 4.5 |
|---|---|---|
| `claude_code` | `opus` | `haiku` |
| `anthropic` | `claude-opus-5` | `claude-haiku-4-5` |
| `openrouter` | `anthropic/claude-opus-5` | `anthropic/claude-haiku-4.5` |

Opus 5 is the default. For a bulk run over thousands of rows, Haiku is much
cheaper and faster — set `LLM_MODEL` accordingly.

## 9. Pitfalls

- **Never add `--bare` to the CLI call.** It switches authentication to
  API-key-only and never reads OAuth or the keychain, which breaks every
  Enterprise seat. Anthropic's own headless-CI docs recommend it; that advice
  assumes you have an API key.
- **Never let the subprocess inherit the parent's `CLAUDE_*` variables.** A
  Claude Code session exports `CLAUDE_EFFORT` and friends, so a pipeline run
  from inside Claude Code would behave differently from one run in a plain
  terminal. `claude_code.child_env()` strips them — and deliberately keeps
  `CLAUDE_CODE_OAUTH_TOKEN`, which a blanket `CLAUDE*` wipe would destroy.
- **Never rely on a provider's default reasoning setting.** Thinking is *on* by
  default for Opus 5 on both the Anthropic API and OpenRouter. The harness
  always states it explicitly; keep it that way.
- **`map_prompts` returns results in input order.** Never re-derive order from
  completion time — you will mislabel rows and not notice.
- **`cost_usd` is a client-side estimate, and on a subscription it is not a
  bill.** Calls consume plan quota instead. Don't sum it and report "we spent
  $47".
- **Quota exhaustion is not retryable.** It lasts hours, so `map_prompts`
  deliberately aborts the batch rather than grinding through thousands of calls
  that will all fail. Do not wrap it in `while True`.
- **`claude_code` has no `temperature` and no `seed`.** Runs are not
  reproducible. Save your outputs.
- **`max_tokens` does nothing on `claude_code`.** The CLI has no equivalent
  flag, so `LLM_MAX_TOKENS` only takes effect on the other two backends.
- **The CLI runs in `.llm_scratch/`, not your project root.** That keeps a
  stray `CLAUDE.md` or `.claude/` next to your code from changing the answer.
  Claude Code still searches *parent* directories, so if you need real
  isolation, point `LLM_SCRATCH_DIR` somewhere outside any project.

## 10. Developing this repo

```bash
uv sync
uv run pytest                     # ~65 tests, no credentials and no network
uv run pre-commit run --all-files
```

The suite has two deliberate tripwires: `conftest.py` blocks `socket.socket`,
so an accidental live call fails loudly rather than passing quietly, and it
strips every variable `from_env()` reads, so your own `.env` cannot change what
the tests assert. If you add a setting, add it to `_CONFIG_VARS`.

`tests/test_claude_code.py` contains `CANNED`, a real `--output-format json`
payload captured from `claude` v2.1.236. It is the closest thing to a spec for
the CLI's output that exists here — if the CLI changes shape, that is the
fixture to update.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
