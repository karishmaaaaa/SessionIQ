"""Repair loop for invalid model output.

When validation fails, we hand the model back its own output plus the *literal*
validation errors and ask for a corrected object. The repaired output is run
through the exact same validation — a repair is untrusted too. The loop is
bounded by ``MAX_REPAIR_ATTEMPTS`` (no unbounded retrying), and every repair
call is a separate API call whose usage is returned for cost logging.

The repair prompt embeds the full transcript (via ``build_user_prompt``) so the
model can fix a grounding error by picking a quote that actually exists, not just
reshuffle the object it already got wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from src.client import LLMResponse, call_model
from src.config import MAX_REPAIR_ATTEMPTS, MODEL
from src.prompts import REPAIR_PROMPT_TEMPLATE, SYSTEM_PROMPT, build_user_prompt
from src.transcript import Transcript
from src.validate import ValidationResult, validate

import anthropic


@dataclass
class RepairResult:
    """Outcome of the bounded repair loop."""

    validation: ValidationResult  # validation of the final repair attempt
    attempts: list[LLMResponse] = field(default_factory=list)  # one per API call, in order


def _build_repair_prompt(transcript: Transcript, previous_raw: dict, errors: list[str]) -> str:
    """Transcript + the previous invalid output + the exact errors to fix."""
    return "\n\n".join(
        [
            build_user_prompt(transcript),
            REPAIR_PROMPT_TEMPLATE.format(
                previous_output=json.dumps(previous_raw, indent=2, ensure_ascii=False),
                errors="\n".join(f"- {err}" for err in errors),
            ),
        ]
    )


def repair(
    *,
    raw: dict,
    errors: list[str],
    transcript: Transcript,
    model: str = MODEL,
    client: anthropic.Anthropic | None = None,
) -> RepairResult:
    """Attempt to fix invalid output, up to ``MAX_REPAIR_ATTEMPTS`` times.

    Each attempt re-validates and, if still invalid, feeds the *new* errors into
    the next attempt. Stops as soon as an attempt validates. Transport errors are
    not caught here — retry/backoff is the pipeline's responsibility.
    """
    current_raw = raw
    current_errors = errors
    attempts: list[LLMResponse] = []
    result: ValidationResult | None = None

    for _ in range(MAX_REPAIR_ATTEMPTS):
        user = _build_repair_prompt(transcript, current_raw, current_errors)
        response = call_model(system=SYSTEM_PROMPT, user=user, model=model, client=client)
        attempts.append(response)

        result = validate(response.raw, transcript)
        if result.ok:
            break
        current_raw, current_errors = response.raw, result.errors

    if result is None:  # MAX_REPAIR_ATTEMPTS <= 0: nothing attempted
        result = validate(raw, transcript)

    return RepairResult(validation=result, attempts=attempts)
