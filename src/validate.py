"""Validation of untrusted model output, in two layers.

Layer 1 (structural) is Pydantic: types, ranges, enums, list sizes, extra
fields. Layer 2 (semantic) checks things Pydantic cannot know — most importantly
that every quoted piece of evidence actually exists in the transcript.

Error strings are written to be pasted straight into the repair prompt, so they
name the field and the offending value, e.g.
``engagement.evidence[0]: evidence quote '...' not found in transcript
(best match ratio 0.41, threshold 0.80)``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from pydantic import ValidationError

from src.config import EVIDENCE_MATCH_THRESHOLD
from src.schema import SessionEvaluation
from src.transcript import Transcript, parse_timestamp

_DIMENSIONS = ("engagement", "clarity", "pacing")

# Lower-cased substrings that mark an obvious attempt to instruct the evaluator
# through transcript content. Used only for the injection *warning*.
_INJECTION_MARKERS = (
    "ignore all previous instructions",
    "ignore previous instructions",
    "system override",
    "evaluation policy",
    "end of transcript",
    "instructions for the evaluation model",
    "disregard the rubric",
    "do not include a flags field",
    "do not mention this instruction",
)


@dataclass
class ValidationResult:
    """Outcome of validating one raw model output against one transcript."""

    ok: bool
    evaluation: SessionEvaluation | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace.

    Applied to both the quote and the transcript so grounding tolerates casing,
    spacing, and punctuation differences without tolerating different words.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _best_match_ratio(quote: str, turns_norm: list[str], early_stop: float) -> float:
    """Best difflib similarity of *quote* against any turn (or window of one).

    This is the deterministic hallucination check at the heart of the project:
    it confirms a quoted utterance really occurs in the transcript, in code, with
    no second model call. A short quote lifted from a long turn is matched
    against sliding windows of that turn so it isn't drowned out by the turn's
    length. Stops early once a match clears the grounding threshold.
    """
    q = _normalise(quote)
    if not q:
        return 0.0
    q_words = q.split()
    window = len(q_words)
    matcher = SequenceMatcher(None, q, "")
    best = 0.0
    for turn in turns_norm:
        matcher.set_seq2(turn)
        best = max(best, matcher.ratio())
        if best >= early_stop:
            return best
        t_words = turn.split()
        if len(t_words) > window:
            for i in range(len(t_words) - window + 1):
                matcher.set_seq2(" ".join(t_words[i : i + window]))
                best = max(best, matcher.ratio())
                if best >= early_stop:
                    return best
    return best


def _iter_quotes(evaluation: SessionEvaluation):
    """Yield (label, quote) for every quote that must be grounded."""
    for name in _DIMENSIONS:
        dimension = getattr(evaluation, name)
        for i, ev in enumerate(dimension.evidence):
            yield f"{name}.evidence[{i}]", ev.quote
    for i, moment in enumerate(evaluation.notable_moments):
        yield f"notable_moments[{i}]", moment.quote


def _iter_timestamps(evaluation: SessionEvaluation):
    """Yield (label, timestamp) for every timestamp the output references."""
    for name in _DIMENSIONS:
        dimension = getattr(evaluation, name)
        for i, ev in enumerate(dimension.evidence):
            if ev.timestamp is not None:
                yield f"{name}.evidence[{i}]", ev.timestamp
    for i, moment in enumerate(evaluation.notable_moments):
        yield f"notable_moments[{i}]", moment.timestamp


def _check_grounding(evaluation: SessionEvaluation, turns_norm: list[str]) -> list[str]:
    errors: list[str] = []
    for label, quote in _iter_quotes(evaluation):
        ratio = _best_match_ratio(quote, turns_norm, EVIDENCE_MATCH_THRESHOLD)
        if ratio < EVIDENCE_MATCH_THRESHOLD:
            errors.append(
                f"{label}: evidence quote {quote!r} not found in transcript "
                f"(best match ratio {ratio:.2f}, threshold {EVIDENCE_MATCH_THRESHOLD:.2f})"
            )
    return errors


def _safe_seconds(timestamp: str) -> int | None:
    try:
        return parse_timestamp(timestamp)
    except (ValueError, AttributeError):
        return None


def _check_timestamps(evaluation: SessionEvaluation, transcript: Transcript) -> list[str]:
    errors: list[str] = []
    valid = {turn.timestamp for turn in transcript.turns}
    duration = transcript.duration_seconds
    for label, timestamp in _iter_timestamps(evaluation):
        if timestamp in valid:
            continue
        seconds = _safe_seconds(timestamp)
        if seconds is None:
            errors.append(f"{label}: timestamp {timestamp!r} is not a valid HH:MM:SS")
        elif duration is not None and seconds > duration:
            errors.append(
                f"{label}: timestamp {timestamp!r} exceeds session duration "
                f"{transcript.duration}"
            )
        else:
            errors.append(f"{label}: timestamp {timestamp!r} does not appear in the transcript")
    return errors


def _check_injection(evaluation: SessionEvaluation, transcript: Transcript) -> list[str]:
    """Warn (do not fail) if the transcript looks injected but the flag is absent."""
    corpus = "\n".join(turn.text for turn in transcript.turns).lower()
    found = [marker for marker in _INJECTION_MARKERS if marker in corpus]
    if found and "possible_injection_attempt" not in evaluation.flags:
        preview = ", ".join(repr(m) for m in found[:3])
        return [
            "transcript contains likely injection markers "
            f"({preview}) but output did not set the 'possible_injection_attempt' flag"
        ]
    return []


def _format_pydantic_errors(exc: ValidationError) -> list[str]:
    """Turn a ValidationError into specific, repair-ready strings."""
    out: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "<root>"
        value = err.get("input")
        if isinstance(value, (str, int, float, bool)):
            out.append(f"{loc}: {err['msg']} (got {value!r})")
        else:
            out.append(f"{loc}: {err['msg']}")
    return out


def validate(raw: dict, transcript: Transcript) -> ValidationResult:
    """Validate one raw model output against its transcript.

    Structural failures short-circuit (there is nothing to ground-check on an
    object that doesn't parse). Semantic failures are collected together so the
    repair prompt sees every problem at once. The injection check is a warning
    and never sets ``ok`` to False.
    """
    try:
        evaluation = SessionEvaluation.model_validate(raw)
    except ValidationError as exc:
        return ValidationResult(ok=False, evaluation=None, errors=_format_pydantic_errors(exc))

    turns_norm = [_normalise(turn.text) for turn in transcript.turns]
    errors = _check_grounding(evaluation, turns_norm)
    errors += _check_timestamps(evaluation, transcript)
    warnings = _check_injection(evaluation, transcript)

    return ValidationResult(
        ok=not errors,
        evaluation=evaluation,
        errors=errors,
        warnings=warnings,
    )
