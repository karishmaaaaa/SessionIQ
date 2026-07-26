"""Validation tests: structural (Layer 1) and semantic (Layer 2).

All offline — these operate on hand-written fixtures, no API involved.
"""

from src.validate import validate


def test_valid_output_passes_both_layers(session1, load_fixture):
    result = validate(load_fixture("valid"), session1)
    assert result.ok
    assert result.evaluation is not None
    assert result.errors == []


def test_score_out_of_range_rejected(session1, load_fixture):
    result = validate(load_fixture("score_out_of_range"), session1)
    assert not result.ok
    assert any("engagement.score" in e and "less than or equal to 5" in e for e in result.errors)


def test_invalid_enum_rejected(session1, load_fixture):
    result = validate(load_fixture("invalid_enum"), session1)
    assert not result.ok
    assert any("clarity.confidence" in e for e in result.errors)


def test_missing_required_field_rejected(session1, load_fixture):
    result = validate(load_fixture("missing_field"), session1)
    assert not result.ok
    assert any(e.startswith("pacing") and "required" in e.lower() for e in result.errors)


def test_extra_injected_field_rejected(session1, load_fixture):
    """Proves model_config extra='forbid' rejects injected keys."""
    result = validate(load_fixture("extra_field"), session1)
    assert not result.ok
    assert any("injected_instruction" in e and "not permitted" in e for e in result.errors)


def test_truncated_object_handled_gracefully(session1, load_fixture):
    """A cut-off object fails structurally without raising — no crash."""
    result = validate(load_fixture("truncated"), session1)
    assert not result.ok
    assert result.evaluation is None
    assert result.errors  # specific missing-field errors, not an exception


def test_hallucinated_evidence_caught(session1, load_fixture):
    """The grounding check catches a quote that isn't in the transcript."""
    result = validate(load_fixture("hallucinated"), session1)
    assert not result.ok
    assert any(
        "not found in transcript" in e and "best match ratio" in e for e in result.errors
    ), result.errors


def test_real_quote_passes_grounding_despite_whitespace_and_case(session1, load_fixture):
    """A verbatim session_01 quote still grounds after case/whitespace mangling."""
    raw = load_fixture("valid")
    # A real quote from session_01, roughed up: uppercased, extra spaces, no
    # trailing punctuation. Grounding normalises both sides, so it must still match.
    raw["engagement"]["evidence"][0]["quote"] = "  SO factoring   is JUST doing foil BACKWARDS  "
    result = validate(raw, session1)
    assert result.ok, result.errors
