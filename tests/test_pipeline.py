"""Pipeline reliability tests: retry, non-retryable failure, batch survival, and
injection resistance. All offline via FakeClient and mocks.
"""

from src import run
from src.client import LLMResponse
from src.config import TRANSCRIPTS_DIR
from src.pipeline import Attempt, PipelineResult, evaluate_transcript
from src.prompts import SYSTEM_PROMPT, build_user_prompt
from src.schema import SessionEvaluation
from src.transcript import load_all


def test_transient_error_retried(session1, fake_client, rate_limit_error, load_fixture, mocker):
    """Two 429s then success: backoff fires, and the run still validates."""
    backoff = mocker.patch("src.pipeline._backoff_sleep")  # no real sleeping
    client = fake_client([rate_limit_error(), rate_limit_error(), load_fixture("valid")])

    result = evaluate_transcript(session1, client=client)

    assert result.status == "valid"
    assert backoff.call_count == 2  # backed off before each retry
    assert client.calls == 3
    assert [a.kind for a in result.attempts] == ["initial", "retry", "retry"]


def test_non_retryable_error_not_retried(session1, fake_client, bad_request_error):
    """A 400 is terminal — exactly one attempt, graceful failure."""
    client = fake_client([bad_request_error()])

    result = evaluate_transcript(session1, client=client)

    assert result.status == "failed"
    assert client.calls == 1
    assert result.attempt_count == 1


def test_batch_survives_one_bad_transcript(mocker, tmp_path, load_fixture):
    """One transcript blowing up must not stop the other four from producing output."""
    transcripts = load_all(TRANSCRIPTS_DIR)
    good_eval = SessionEvaluation.model_validate(load_fixture("valid"))

    def fake_eval(transcript, *, model):
        if transcript.id == "session_04_truncated":
            raise RuntimeError("boom")  # simulate an unexpected crash
        return PipelineResult(
            transcript.id,
            "valid",
            good_eval,
            attempts=[Attempt("initial", LLMResponse({}, 100, 20, "claude-sonnet-5", 500, "tool_use"))],
        )

    mocker.patch("src.run.evaluate_transcript", side_effect=fake_eval)
    mocker.patch("src.run.append_run_records", return_value=[])

    results = run.evaluate_all(
        transcripts, model="claude-sonnet-5", out_dir=tmp_path, run_id="r", timestamp="t"
    )

    written = {p.stem for p in tmp_path.glob("*.json")}
    assert len(written) == 5  # every transcript produced an output file
    assert sum(r.status == "valid" for r in results) == 4
    assert sum(r.status == "failed" for r in results) == 1


def test_injection_resistance_surfaces_warning(session5, fake_client, load_fixture):
    """A compliant model (all 5s, empty flags) obeys the injected instruction, yet
    the semantic layer still surfaces the injection as a warning."""
    compliant = load_fixture("session_05_compliant")

    result = evaluate_transcript(session5, client=fake_client([compliant]))

    # The object is structurally sound and every quote is real, so it is accepted...
    assert result.status == "valid"
    # ...but the injection the model failed to flag is surfaced anyway.
    assert result.warnings
    assert any("injection" in w.lower() for w in result.warnings)


def test_prompt_contains_injection_guard_and_delimiter(session5):
    """The prompt itself carries the guard text and the explicit boundary."""
    assert "possible_injection_attempt" in SYSTEM_PROMPT
    assert "untrusted data" in SYSTEM_PROMPT

    user = build_user_prompt(session5)
    assert '<transcript id="session_05_adversarial">' in user
    assert "</transcript>" in user
    # boundary restatement must come AFTER the closing tag
    assert user.index("untrusted session data") > user.index("</transcript>")
