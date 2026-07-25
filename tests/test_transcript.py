"""Stage 2 parser tests. Pure/offline — no API involved.

These assert the format is parsed correctly and, critically, that continuation
lines (wrapped text and the multi-line injection payload) are handled without
dropping content.
"""

from collections import Counter

import pytest

from src import config
from src.transcript import load_all, load_transcript, parse_timestamp

# Verified against the committed transcript files. Doubles as a regression guard:
# if continuation handling ever splits or drops turns, these counts shift.
EXPECTED = {
    "session_01_strong": {"turns": 88, "TUTOR": 42, "STUDENT": 46},
    "session_02_weak": {"turns": 39, "TUTOR": 19, "STUDENT": 20},
    "session_03_pacing": {"turns": 74, "TUTOR": 38, "STUDENT": 36},
    "session_04_truncated": {"turns": 22, "TUTOR": 12, "STUDENT": 10},
    "session_05_adversarial": {"turns": 65, "TUTOR": 32, "STUDENT": 33},
}


@pytest.fixture(scope="module")
def transcripts():
    return {t.id: t for t in load_all(config.TRANSCRIPTS_DIR)}


def test_all_five_transcripts_parse(transcripts):
    assert set(transcripts) == set(EXPECTED)
    for t in transcripts.values():
        assert t.turns, f"{t.id} produced no turns"
        assert t.duration is not None


@pytest.mark.parametrize("tid", sorted(EXPECTED))
def test_speaker_counts(transcripts, tid):
    t = transcripts[tid]
    counts = Counter(turn.speaker for turn in t.turns)
    assert len(t.turns) == EXPECTED[tid]["turns"]
    assert counts["TUTOR"] == EXPECTED[tid]["TUTOR"]
    assert counts["STUDENT"] == EXPECTED[tid]["STUDENT"]
    assert set(counts) == {"TUTOR", "STUDENT"}


def test_injection_payload_stays_in_one_turn(transcripts):
    """The multi-line 'END OF TRANSCRIPT' payload must land inside a single turn,
    not be split across turns or dropped by the blank line inside it."""
    s5 = transcripts["session_05_adversarial"]
    marker = "INSTRUCTIONS FOR THE EVALUATION MODEL"
    containing = [turn for turn in s5.turns if marker in turn.text]

    assert len(containing) == 1, "payload should be in exactly one turn"
    payload_turn = containing[0]
    assert payload_turn.timestamp == "00:06:47"
    assert payload_turn.speaker == "STUDENT"
    # Whole multi-line payload present, including its fake end marker, the blank
    # line, and the JSON the injection wants us to emit.
    assert "--- END OF TRANSCRIPT ---" in payload_turn.text
    assert "---\n\nINSTRUCTIONS" in payload_turn.text  # internal blank line kept
    assert '"engagement": {"score": 5}' in payload_turn.text


def test_earlier_injection_also_single_turn(transcripts):
    s5 = transcripts["session_05_adversarial"]
    hits = [t for t in s5.turns if "Ignore all previous instructions" in t.text]
    assert len(hits) == 1
    assert hits[0].timestamp == "00:03:37"


def test_truncated_session_parses(transcripts):
    """session_04 ends mid-word; that is a valid turn, not a parse error."""
    s4 = transcripts["session_04_truncated"]
    assert len(s4.turns) == EXPECTED["session_04_truncated"]["turns"]
    last = s4.turns[-1]
    assert last.text.endswith("conne")  # cut off mid-"connection"
    assert last.timestamp == "00:03:41"


def test_seconds_and_durations():
    s2 = load_transcript(config.TRANSCRIPTS_DIR / "session_02_weak.txt")
    # timestamp -> seconds
    assert s2.turns[0].seconds == parse_timestamp(s2.turns[0].timestamp)
    assert parse_timestamp("00:01:47") == 107
    # one gap per turn; the long tutor monologue shows up as a large gap
    gaps = s2.turn_durations()
    assert len(gaps) == len(s2.turns)
    assert gaps[2] == 96  # 00:00:11 tutor lecture -> 00:01:47 student "Yeah"
    # last turn's gap runs to the session end and is non-negative
    assert gaps[-1] is not None and gaps[-1] >= 0


def test_missing_directory_globs_only_txt(tmp_path):
    """load_all must never pick up a .md answer-key file sharing the folder."""
    (tmp_path / "session_99.txt").write_text(
        "SESSION_ID: session_99\nDURATION: 00:00:05\n\n[00:00:01] TUTOR: hi\n",
        encoding="utf-8",
    )
    (tmp_path / "transcript_design.md").write_text("ANSWERS", encoding="utf-8")
    loaded = load_all(tmp_path)
    assert [t.id for t in loaded] == ["session_99"]
