"""Load and parse session transcripts into structured turns.

File format: a header of ``KEY: value`` lines, a blank line, then turns::

    [00:03:12] TUTOR: Okay, so what happens if we multiply both sides by x?

Two details drive the design:

* **Continuations.** A line that does not start with a ``[HH:MM:SS] SPEAKER:``
  marker — including a blank line — is treated as a continuation of the previous
  turn, not a parse error. ``session_05`` hides a multi-line prompt-injection
  payload this way; getting continuations right is what keeps that payload inside
  a single turn instead of silently dropping it.
* **Timestamps are the only pacing signal.** ``seconds`` and ``turn_durations()``
  are derived from the timestamps so the prompt can reason about long tutor
  monologues vs. short student replies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import re

# A turn line: "[HH:MM:SS] SPEAKER: text". Speaker is everything up to the first
# colon after the timestamp, so colons inside the utterance (e.g. "v2.4:") stay
# in the text.
_TURN_RE = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s+([^:]+):(.*)$")
_HEADER_RE = re.compile(r"^(SESSION_ID|SUBJECT|DURATION|PARTICIPANTS):\s*(.*)$")


def parse_timestamp(ts: str) -> int:
    """Convert an "HH:MM:SS" string to an integer number of seconds."""
    hours, minutes, seconds = (int(part) for part in ts.split(":"))
    return hours * 3600 + minutes * 60 + seconds


@dataclass
class Turn:
    """A single utterance. ``seconds`` is the timestamp in absolute seconds."""

    timestamp: str
    seconds: int
    speaker: str
    text: str


@dataclass
class Transcript:
    """A parsed session. ``id`` is the filename stem — the identifier used for
    output files and run logs, kept stable across the pipeline."""

    id: str
    subject: str | None
    duration: str | None
    path: Path
    turns: list[Turn] = field(default_factory=list)

    @property
    def duration_seconds(self) -> int | None:
        """Session length in seconds, or None if the header lacked a duration."""
        if not self.duration:
            return None
        try:
            return parse_timestamp(self.duration)
        except (ValueError, AttributeError):
            return None

    @property
    def raw_text(self) -> str:
        """Rebuild the turn lines as a single block for embedding in the prompt."""
        return "\n".join(f"[{t.timestamp}] {t.speaker}: {t.text}" for t in self.turns)

    def turn_durations(self) -> list[int | None]:
        """Gap in seconds from each turn to the next event.

        For every turn this is the time until the next turn begins; for the last
        turn it is the time until the session's stated end (or None if the
        duration header is missing). This is the primary pacing signal: a long
        gap after a tutor turn is a monologue, a short one after a student turn
        is a clipped reply.
        """
        gaps: list[int | None] = []
        end = self.duration_seconds
        for i, turn in enumerate(self.turns):
            if i + 1 < len(self.turns):
                gaps.append(self.turns[i + 1].seconds - turn.seconds)
            else:
                gaps.append(end - turn.seconds if end is not None else None)
        return gaps


def load_transcript(path: str | Path) -> Transcript:
    """Parse one transcript file into a :class:`Transcript`.

    Missing header fields default to None rather than raising. Lines that are not
    turn markers are folded into the preceding turn as continuations.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()

    # Header region: everything before the first turn line. Unknown or blank
    # lines here are simply ignored.
    header: dict[str, str] = {}
    body_start = len(lines)
    for i, line in enumerate(lines):
        if _TURN_RE.match(line):
            body_start = i
            break
        match = _HEADER_RE.match(line)
        if match:
            header[match.group(1)] = match.group(2).strip()

    turns: list[Turn] = []
    for line in lines[body_start:]:
        match = _TURN_RE.match(line)
        if match:
            hh, mm, ss, speaker, text = match.groups()
            timestamp = f"{hh}:{mm}:{ss}"
            turns.append(
                Turn(
                    timestamp=timestamp,
                    seconds=parse_timestamp(timestamp),
                    speaker=speaker.strip(),
                    text=text.strip(),
                )
            )
        elif turns:
            # Continuation of the previous turn (a wrapped line or a blank line
            # inside a multi-line payload). Accumulate raw so internal blank
            # lines survive; trailing whitespace is trimmed once below.
            turns[-1].text = f"{turns[-1].text}\n{line}"

    for turn in turns:
        turn.text = turn.text.rstrip()

    return Transcript(
        id=path.stem,
        subject=header.get("SUBJECT"),
        duration=header.get("DURATION"),
        path=path,
        turns=turns,
    )


def load_all(directory: str | Path) -> list[Transcript]:
    """Load every ``*.txt`` transcript in *directory*, sorted by filename.

    Globs ``*.txt`` only — never ``*.md`` — so the design/answer-key document is
    never picked up even if it sits alongside the transcripts.
    """
    paths = sorted(Path(directory).glob("*.txt"))
    return [load_transcript(p) for p in paths]
