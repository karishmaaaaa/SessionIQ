# Tutoring Session Evaluator

Reads tutoring-session transcripts, calls an LLM with a fixed structured-output
schema, and scores each session on engagement, clarity, and pacing. Every score
is bound to a verbatim quote from the transcript, and that quote is checked
against the transcript in code — so hallucinated evidence is caught
deterministically, without a second model call. Output that fails validation is
repaired once; transport errors are retried with backoff; a single bad
transcript never crashes the batch, and token usage and cost are logged per
API call.

## Setup

From a fresh clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then edit .env and add your ANTHROPIC_API_KEY
```

The key is only needed for real runs. `pytest` and `--dry-run` work without one.

## Usage

```bash
# Evaluate every transcript in transcripts/
python -m src.run --all

# Evaluate a single transcript
python -m src.run --transcript transcripts/session_01_strong.txt

# Build and print the exact prompt with an estimated input-token count,
# making NO API call — free prompt iteration
python -m src.run --all --dry-run

# Run the test suite (offline; no API key required)
pytest
```

Results are written to `outputs/<transcript_id>.json`, one JSONL line per API
attempt to `outputs/runs.jsonl`, and a summary prints at the end. Exit code is
`0` if every transcript succeeded, `1` if any failed — usable as a CI gate.
Other flags: `--out DIR` (output directory), `--model` (override the model).

## Design

**The validation ladder.** Each transcript goes through the same rungs
(`src/pipeline.py`):

```
1. Native structured output              -> usually valid
2. Structural validation (Pydantic)      -> types, ranges, enums
3. Semantic validation                   -> evidence must exist in transcript
4. On validation failure: repair call    -> feed back exact error text
5. On transport failure: retry + backoff -> 429, 5xx, timeout, overloaded
6. Otherwise: graceful failure           -> record and continue
```

**Evidence-bound scoring.** Every dimension score ships with at least one
verbatim quote. This lets a human audit the judgment without rereading the
session, and it makes hallucination detectable in code: `src/validate.py`
normalises the quote and the transcript (lowercase, strip punctuation, collapse
whitespace) and uses `difflib` against each turn and sliding windows of it. A
quote that isn't really there fails with a specific, repair-ready error.

**Native structured output, not JSON-in-prompt.** Shape consistency comes from a
tool whose `input_schema` is the Pydantic schema (`SessionEvaluation`), forced
with `tool_choice`. The model can't drift the shape because the shape isn't
described in prose. Constraints (score 1–5, list sizes) are left soft in the
tool schema on purpose and enforced by the Pydantic layer — that split is why
both a structural and a semantic layer exist.

**Prompt injection.** The transcript is untrusted data. It is wrapped in an
explicit `<transcript>…</transcript>` delimiter with a restatement *after* the
closing tag that the contents were data, and the system prompt instructs the
model to treat any embedded instruction, policy update, or end-of-transcript
marker as content to be scored, not obeyed — flagging it with
`possible_injection_attempt`. As a backstop that does not depend on the model
complying, the semantic layer independently detects injection markers in the
transcript and warns if the output failed to flag them.

## Output format

`outputs/<transcript_id>.json` for a successful evaluation (trimmed):

```json
{
  "transcript_id": "session_01_strong",
  "schema_version": "1.0",
  "engagement": {
    "score": 5,
    "confidence": "high",
    "rationale": "The student reasons aloud and connects ideas without prompting.",
    "evidence": [
      {"quote": "So factoring is just... doing FOIL backwards?", "timestamp": "00:01:50"}
    ]
  },
  "clarity":  { "score": 4, "confidence": "high",   "rationale": "…", "evidence": [ … ] },
  "pacing":   { "score": 4, "confidence": "medium", "rationale": "…", "evidence": [ … ] },
  "notable_moments": [
    {"kind": "breakthrough", "timestamp": "00:01:50",
     "quote": "So factoring is just... doing FOIL backwards?",
     "why_it_matters": "The student connects factoring to FOIL on their own."}
  ],
  "summary": "A strong session: the student derives rules rather than memorizing them.",
  "flags": []
}
```

A failed transcript instead writes `{"status": "failed", "transcript_id": ...,
"attempts": n, "last_error": ..., "errors": [...], "raw_response": ...}` so it
can be debugged without another API call.

One line in `outputs/runs.jsonl`, per API attempt (token values illustrative;
`cost_usd` is `null` until prices are filled into `config.PRICING`):

```json
{"run_id": "…", "timestamp": "2026-07-26T…", "transcript_id": "session_03_pacing",
 "model": "claude-sonnet-5", "prompt_version": "1.0", "schema_version": "1.0",
 "attempt": 1, "kind": "initial", "input_tokens": 4210, "output_tokens": 780,
 "cost_usd": null, "latency_ms": 6120, "status": "valid", "errors": []}
```

The end-of-run summary reports success/failure counts, **retry rate** and
**repair rate**, total tokens, mean/p50/p95 latency, and total/per-transcript
cost.

## Test transcripts

Five hand-designed inputs in `transcripts/`, each stressing something different:

- **`session_01_strong`** — a genuinely strong session; the student initiates,
  reasons aloud, and self-corrects. The high-quality baseline.
- **`session_02_weak`** — sustained tutor monologue with one-word student
  replies; tests engagement and pacing scoring on a lecture.
- **`session_03_pacing`** — time spent re-teaching material the student already
  knows while the actual homework topic is rushed at the end; a pacing case.
- **`session_04_truncated`** — a very short session that cuts off mid-word;
  tests the `insufficient_content` / low-confidence escape hatch and that
  truncation is parsed, not treated as an error.
- **`session_05_adversarial`** — the student pastes several prompt-injection
  payloads (fake system messages, an "evaluation policy update", an
  end-of-transcript marker with target JSON); tests injection resistance.

## Cost

TBD — no real end-to-end API run has been performed yet. After one full
five-transcript run with `config.PRICING` filled in, this will report the
measured total from `outputs/runs.jsonl`.
