"""Repair-loop tests, exercised end-to-end through the pipeline.

Offline: the client is a FakeClient replaying queued tool outputs.
"""

from src.pipeline import evaluate_transcript


def test_repair_fixes_invalid_output(session1, fake_client, load_fixture):
    """Bad-then-good: initial output fails grounding, the repair call fixes it."""
    bad = load_fixture("hallucinated")  # structurally valid, but a hallucinated quote
    good = load_fixture("valid")
    client = fake_client([bad, good])

    result = evaluate_transcript(session1, client=client)

    assert result.status == "repaired"
    assert result.evaluation is not None
    assert len(result.usages) == 2  # one initial + one repair call
    assert [a.kind for a in result.attempts] == ["initial", "repair"]


def test_repair_exhausted(session1, fake_client, load_fixture):
    """Bad-then-bad: repair is bounded and the transcript fails without raising."""
    bad = load_fixture("hallucinated")
    client = fake_client([bad, bad])

    result = evaluate_transcript(session1, client=client)  # must not raise

    assert result.status == "failed"
    assert result.evaluation is None
    assert result.errors  # the grounding error survives to the failure record
    assert client.calls == 2  # initial + one bounded repair (MAX_REPAIR_ATTEMPTS)
