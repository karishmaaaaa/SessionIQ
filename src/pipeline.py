"""Orchestration: the reliability ladder for evaluating one transcript.

    1. Native structured output              -> usually valid
    2. Structural validation (Pydantic)      -> types, ranges, enums
    3. Semantic validation                   -> evidence must exist in transcript
    4. On validation failure: repair call    -> feed back exact error text
    5. On transport failure: retry + backoff -> 429, 5xx, timeout, overloaded
    6. Otherwise: graceful failure           -> record and continue

``evaluate_transcript`` never raises for an expected failure — a single bad
transcript records its failure and the batch moves on. It returns a
``PipelineResult`` carrying the status, the evaluation (if any), every per-call
usage record, and, on failure, the raw response so a human can debug without
paying for another call.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import anthropic

from src.client import LLMResponse, StructuredOutputError, call_model
from src.config import MAX_REPAIR_ATTEMPTS, MAX_RETRIES, MODEL
from src.prompts import SYSTEM_PROMPT, build_user_prompt
from src.repair import repair
from src.schema import SessionEvaluation
from src.transcript import Transcript
from src.validate import validate

logger = logging.getLogger(__name__)

# Transport errors worth retrying. Everything else (400, 401, 403, 404, a missing
# tool call) is terminal — retrying an identical request would just fail again.
_TRANSIENT_ERRORS = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,  # 5xx incl. 529 overloaded
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
)


@dataclass
class Attempt:
    """One physical API call: its role, and either the response or the error."""

    kind: str  # "initial" | "retry" | "repair"
    response: LLMResponse | None = None
    error: str | None = None  # transport/terminal error message, if the call failed


@dataclass
class PipelineResult:
    """Everything a run needs to log and persist for one transcript."""

    transcript_id: str
    status: str  # "valid" | "repaired" | "failed"
    evaluation: SessionEvaluation | None
    attempts: list[Attempt] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # validation errors, if failed
    last_error: str | None = None  # terminal transport error, if failed
    raw_response: dict | None = None  # last raw output, for debugging a failure

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def usages(self) -> list[LLMResponse]:
        """Per-call usage records — the ones that actually returned tokens."""
        return [a.response for a in self.attempts if a.response is not None]

    def to_output(self) -> dict:
        """The dict written to ``outputs/<transcript_id>.json``."""
        if self.status == "failed":
            return {
                "status": "failed",
                "transcript_id": self.transcript_id,
                "attempts": self.attempt_count,
                "last_error": self.last_error,
                "errors": self.errors,
                "raw_response": self.raw_response,
            }
        return self.evaluation.model_dump(mode="json")


def _describe(exc: Exception) -> str:
    """Compact error string for logs — no secrets, just type and message."""
    return f"{type(exc).__name__}: {exc}"


def _backoff_sleep(attempt: int) -> None:
    """Exponential backoff between transient retries (1s, 2s, 4s, ...)."""
    time.sleep(2**attempt)


def _call_with_retry(
    attempts: list[Attempt],
    *,
    system: str,
    user: str,
    model: str,
    client: anthropic.Anthropic | None,
    first_kind: str,
) -> LLMResponse:
    """Make the structured-output call, retrying transient errors with backoff.

    Records every physical attempt (first as ``first_kind``, re-tries as
    ``"retry"``). Transient errors are retried up to ``MAX_RETRIES`` times;
    non-transient errors are recorded and re-raised immediately.
    """
    for i in range(MAX_RETRIES + 1):
        kind = first_kind if i == 0 else "retry"
        try:
            response = call_model(system=system, user=user, model=model, client=client)
        except _TRANSIENT_ERRORS as exc:
            attempts.append(Attempt(kind=kind, error=_describe(exc)))
            if i < MAX_RETRIES:
                logger.warning("transient error, backing off (attempt %d): %s", i + 1, _describe(exc))
                _backoff_sleep(i)
                continue
            raise  # retries exhausted
        except (anthropic.APIError, StructuredOutputError) as exc:
            attempts.append(Attempt(kind=kind, error=_describe(exc)))
            raise  # non-retryable
        else:
            attempts.append(Attempt(kind=kind, response=response))
            return response
    raise AssertionError("unreachable")  # pragma: no cover


def _failed(
    transcript: Transcript,
    attempts: list[Attempt],
    *,
    last_error: str | None = None,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    raw: dict | None = None,
) -> PipelineResult:
    return PipelineResult(
        transcript_id=transcript.id,
        status="failed",
        evaluation=None,
        attempts=attempts,
        warnings=warnings or [],
        errors=errors or [],
        last_error=last_error,
        raw_response=raw,
    )


def evaluate_transcript(
    transcript: Transcript,
    *,
    model: str = MODEL,
    client: anthropic.Anthropic | None = None,
) -> PipelineResult:
    """Run the full ladder for one transcript. Never raises for an API/output
    failure — records it and returns a ``PipelineResult`` so the batch survives.
    """
    attempts: list[Attempt] = []
    user = build_user_prompt(transcript)

    # Steps 1 + 5: initial structured-output call, retrying transient errors.
    try:
        response = _call_with_retry(
            attempts, system=SYSTEM_PROMPT, user=user, model=model, client=client, first_kind="initial"
        )
    except (anthropic.APIError, StructuredOutputError) as exc:
        raw = {"text": exc.text} if isinstance(exc, StructuredOutputError) else None
        return _failed(transcript, attempts, last_error=_describe(exc), raw=raw)

    # Steps 2 + 3: structural then semantic validation.
    result = validate(response.raw, transcript)
    if result.ok:
        return PipelineResult(
            transcript_id=transcript.id,
            status="valid",
            evaluation=result.evaluation,
            attempts=attempts,
            warnings=result.warnings,
        )

    # Step 4: repair on validation failure. Never retries an identical call.
    if MAX_REPAIR_ATTEMPTS > 0:
        try:
            outcome = repair(
                raw=response.raw, errors=result.errors, transcript=transcript, model=model, client=client
            )
        except (anthropic.APIError, StructuredOutputError) as exc:
            return _failed(transcript, attempts, last_error=_describe(exc), raw=response.raw)

        attempts.extend(Attempt(kind="repair", response=r) for r in outcome.attempts)
        if outcome.validation.ok:
            return PipelineResult(
                transcript_id=transcript.id,
                status="repaired",
                evaluation=outcome.validation.evaluation,
                attempts=attempts,
                warnings=outcome.validation.warnings,
            )
        result = outcome.validation
        if outcome.attempts:
            response = outcome.attempts[-1]

    # Step 6: graceful failure — validation still failing after repair.
    return _failed(
        transcript,
        attempts,
        errors=result.errors,
        warnings=result.warnings,
        raw=response.raw,
    )
