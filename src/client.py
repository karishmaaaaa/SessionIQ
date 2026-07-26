"""Thin Anthropic client: one structured-output call, with usage capture.

No orchestration lives here — retries and repair are the pipeline's job. This
module just turns a (system, user) prompt pair into a validated *shape* plus the
real token counts the API reports.

Key choices, and why:

* **Native structured output via forced tool use.** We define a single tool
  whose ``input_schema`` is ``SessionEvaluation.model_json_schema()`` and force
  it with ``tool_choice``. Shape consistency across runs comes from the schema,
  not from asking the prompt for JSON — which is exactly the consistency the
  brief calls for.
* **Non-strict tool use, on purpose.** The schema carries range/length bounds
  (``score`` 1–5, ``rationale`` ≤400, ``evidence`` 1–3). Strict / structured-
  output mode does not support those constraints, and we *want* them soft here:
  the model treats them as guidance, and ``validate.py`` enforces them as hard
  rules. That two-layer split is the whole point of the project.
* **No ``temperature``.** ``claude-sonnet-5`` (and the Opus 4.7+ generation)
  rejects ``temperature``/``top_p``/``top_k`` with a 400 — sampling parameters
  were removed on these models. ``config.TEMPERATURE`` is kept for reference but
  is deliberately not sent. Determinism is steered by ``thinking: disabled`` and
  the fixed schema instead.
* **Thinking disabled.** A fixed-schema extraction does not need adaptive
  thinking (which is on by default when the field is omitted on Sonnet 5). Off
  keeps latency and output-token cost predictable.
* **Real token counts.** Read straight off ``response.usage`` — never estimated
  with a tokenizer.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache

import anthropic
from dotenv import load_dotenv

from src.config import MAX_TOKENS, MODEL
from src.schema import evaluation_json_schema

load_dotenv()  # populate ANTHROPIC_API_KEY from a local .env if present

TOOL_NAME = "record_session_evaluation"

# Built once. The input_schema is the contract itself, so every call returns the
# same shape regardless of prompt wording.
_EVALUATION_TOOL: dict = {
    "name": TOOL_NAME,
    "description": (
        "Record the structured evaluation of the tutoring session. "
        "Call this exactly once with the completed evaluation."
    ),
    "input_schema": evaluation_json_schema(),
}


class StructuredOutputError(RuntimeError):
    """Raised when the model's response contains no usable tool call.

    Carries the response text so the pipeline can log what came back without
    paying for another call.
    """

    def __init__(self, message: str, *, text: str = "") -> None:
        super().__init__(message)
        self.text = text


@dataclass
class LLMResponse:
    """One structured-output call's result plus the metadata cost/logging needs."""

    raw: dict  # parsed tool-input dict — untrusted until validated
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: int
    stop_reason: str | None
    text: str = ""  # any stray text blocks, kept for failure logging


@lru_cache(maxsize=1)
def _get_client() -> anthropic.Anthropic:
    """Construct the SDK client once. Raises if no API key is available.

    The key is never logged. The SDK also resolves it from the environment, but
    we check explicitly to fail with a clear, actionable message.

    ``max_retries=0`` disables the SDK's built-in transient-error retrying so the
    pipeline's own retry/backoff is the single, visible retry mechanism.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return anthropic.Anthropic(max_retries=0)


def _extract_tool_input(message: anthropic.types.Message) -> tuple[dict, str]:
    """Pull the forced tool call's input dict out of the response.

    Also collects any text blocks so a malformed response (no tool call) can be
    logged. Raises :class:`StructuredOutputError` if the tool was not called.
    """
    tool_input: dict | None = None
    texts: list[str] = []
    for block in message.content:
        if block.type == "tool_use" and block.name == TOOL_NAME:
            tool_input = dict(block.input)
        elif block.type == "text":
            texts.append(block.text)
    text = "\n".join(texts)
    if tool_input is None:
        raise StructuredOutputError(
            f"model did not call {TOOL_NAME!r}", text=text
        )
    return tool_input, text


def call_model(
    *,
    system: str,
    user: str,
    model: str = MODEL,
    client: anthropic.Anthropic | None = None,
) -> LLMResponse:
    """Make one forced structured-output call and capture usage.

    Used for both the initial evaluation and repair attempts — the difference is
    only the ``user`` prompt. Transport errors (rate limit, overloaded, 5xx,
    timeout) propagate to the caller, which owns retry/backoff.
    """
    client = client or _get_client()
    started = time.perf_counter()
    message = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        thinking={"type": "disabled"},
        system=system,
        messages=[{"role": "user", "content": user}],
        tools=[_EVALUATION_TOOL],
        tool_choice={"type": "tool", "name": TOOL_NAME},
    )
    latency_ms = int((time.perf_counter() - started) * 1000)

    raw, text = _extract_tool_input(message)
    return LLMResponse(
        raw=raw,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        model=message.model,
        latency_ms=latency_ms,
        stop_reason=message.stop_reason,
        text=text,
    )
