"""The evaluation contract.

Every other module depends on these models. Design decisions worth a reviewer's
attention:

* **Evidence-bound scoring.** Each dimension score must ship with at least one
  verbatim ``Evidence.quote`` from the transcript. A human can audit the judgment
  without rereading the session, and — more importantly — the grounding check in
  ``validate.py`` can confirm the quote actually exists, making hallucination
  detectable *in code* rather than by a second model call.
* **Enums, not free text.** ``confidence``, ``kind`` and ``flags`` are closed
  vocabularies so results aggregate cleanly across many sessions.
* **A legitimate escape hatch.** ``confidence`` and ``flags`` let the model say
  "not enough signal" instead of confabulating — this is the honest answer for a
  transcript that is genuinely too short to score (e.g. a truncated session).
* **Pinned ``schema_version``.** Old files in ``outputs/`` stay identifiable
  after the schema evolves.
* **``extra="forbid"`` everywhere.** Injected extra fields — at any nesting
  level — are rejected structurally, and the generated JSON schema advertises
  ``additionalProperties: false`` to the model.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Applied to every model in this contract: an unexpected key anywhere in the
# object graph is a hard validation error, not a silently ignored field.
_STRICT = ConfigDict(extra="forbid")


class Evidence(BaseModel):
    """A verbatim utterance backing a score. The quote must be copyable straight
    out of the transcript — the grounding check relies on that."""

    model_config = _STRICT

    quote: str = Field(description="Verbatim utterance text copied from the transcript.")
    timestamp: str | None = Field(default=None, description='"HH:MM:SS" of the utterance, if known.')


class Dimension(BaseModel):
    """One rubric dimension: a bounded score, a stated confidence, a short
    rationale, and the evidence that grounds it."""

    model_config = _STRICT

    score: int = Field(ge=1, le=5, description="1 (poor) to 5 (excellent), inclusive.")
    confidence: Literal["low", "medium", "high"]
    rationale: str = Field(max_length=400)
    evidence: list[Evidence] = Field(min_length=1, max_length=3)


class NotableMoment(BaseModel):
    """A specific, timestamped moment worth flagging to a human reviewer."""

    model_config = _STRICT

    kind: Literal[
        "breakthrough",
        "confusion",
        "disengagement",
        "strong_explanation",
        "missed_cue",
        "unresolved_gap",
    ]
    timestamp: str
    quote: str = Field(description="Verbatim utterance from the transcript.")
    why_it_matters: str


class SessionEvaluation(BaseModel):
    """The full structured evaluation for one tutoring session."""

    model_config = _STRICT

    transcript_id: str
    schema_version: Literal["1.0"]
    engagement: Dimension  # about the student: initiative, question-asking, reasoning aloud
    clarity: Dimension     # about the tutor's explanations
    pacing: Dimension      # time allocation relative to student need
    notable_moments: list[NotableMoment] = Field(max_length=8)
    summary: str = Field(max_length=600)
    flags: list[
        Literal["insufficient_content", "off_topic", "possible_injection_attempt"]
    ] = Field(default_factory=list)


def evaluation_json_schema() -> dict[str, Any]:
    """JSON schema for the model's structured-output tool.

    Passed as the tool's ``input_schema`` in ``client.py`` so shape consistency
    comes from the contract, not from prompt wording.
    """
    return SessionEvaluation.model_json_schema()
