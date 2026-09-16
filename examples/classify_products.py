"""Example: label short product descriptions with a JSON schema.

Run it with::

    python examples/classify_products.py

It needs no API key if you are logged into Claude Code (``claude auth status``
should report your Enterprise seat). See the README for setup and for what the
printed parity signals mean.
"""

from typing import Any

import pandas as pd

from claude_as_api import LLMClient, map_dataframe

PRODUCT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["produce", "meat", "dairy", "beverage", "non_food", "unknown"],
        },
        "normalized_name": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["category", "normalized_name", "confidence"],
    "additionalProperties": False,
}

SYSTEM = (
    "You label abbreviated product descriptions. Answer only with the "
    "requested JSON. If a description is ambiguous, choose 'unknown' and give "
    "a low confidence."
)

TEMPLATE = "Product description: {product_description}\n\nLabel this item."

SAMPLE_ROWS = [
    {"product_description": "ORG BRSSL SPRTS 5# BAG"},
    {"product_description": "CHKN BRST BNLS SKNLS FRZ 10#"},
    {"product_description": "PPR TWL 2PLY 30RL"},
]


def summarize_parity(client: LLMClient, labeled: pd.DataFrame) -> str:
    """Describe how much agent machinery the run actually used.

    A workflow that burns hundreds of thinking tokens per row is leaning on
    Claude's reasoning and may not survive a move to a bare LLM. Printing this
    every run is what keeps that visible.

    Args:
        client: The client that produced the results.
        labeled: The frame returned by :func:`claude_as_api.map_dataframe`.

    Returns:
        A one-line summary.
    """
    thinking = labeled["llm_thinking_tokens"].dropna().tolist()
    turns = sorted(set(labeled["llm_num_turns"].dropna().tolist()))
    return (
        f"provider={client.config.provider} "
        f"model={client.config.resolve_model()} "
        f"effort={client.config.effort} | "
        f"thinking_tokens={thinking or 'n/a'} | "
        f"turns={turns or 'n/a'}"
    )


if __name__ == "__main__":
    products = pd.DataFrame(SAMPLE_ROWS)
    client = LLMClient()

    labeled = map_dataframe(
        products,
        template=TEMPLATE,
        schema=PRODUCT_SCHEMA,
        client=client,
        system=SYSTEM,
    )

    shown = ["product_description", "category", "normalized_name", "confidence"]
    print(labeled[shown].to_string(index=False))
    print()
    print(summarize_parity(client, labeled))

    failures = labeled["llm_error"].dropna()
    if not failures.empty:
        print(f"\n{len(failures)} row(s) failed:")
        print(failures.to_string())
