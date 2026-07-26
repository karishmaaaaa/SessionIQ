# Tutoring Session Evaluator

Scores teaching-session transcripts on **engagement, clarity, and pacing** using an LLM with a fixed structured-output schema — and treats every byte the model returns as untrusted until proven otherwise.

The interesting part isn't the API call. It's what happens to the response afterwards: every score must cite a verbatim quote, every quote is checked against the source transcript in code, invalid output is repaired, transport errors are retried, and one bad transcript never takes down the batch. Token usage and cost are recorded per API call.

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your ANTHROPIC_API_KEY

python -m src.run --all       # evaluate all five transcripts
```

No API key? Two things still work and are worth trying first:

```bash
pytest                        # 26 tests, fully offline
python -m src.run --all --dry-run   # prints the exact prompt + token estimate, makes no call
```

---

## Layout

```
src/
  config.py       model id, pricing table, retry limits, paths
  schema.py       the contract — Pydantic models, extra="forbid"
  transcript.py   parser: turns, speakers, timestamps, turn durations
  prompts.py      versioned system/user/repair templates
  client.py       one forced-tool call + usage capture
  validate.py     structural (Pydantic) + semantic (evidence grounding)
  repair.py       bounded repair loop
  pipeline.py     the ladder: retry, repair, graceful failure
  cost.py         token accounting, runs.jsonl, run summary
  run.py          CLI
tests/            26 offline tests + JSON fixtures
transcripts/      five hand-designed test sessions
outputs/          per-transcript JSON + runs.jsonl
```

---

## Usage

| Command | What it does |
|---|---|
| `python -m src.run --all` | Evaluate every transcript in `transcripts/` |
| `python -m src.run --transcript PATH` | Evaluate one file |
| `python -m src.run --all --dry-run` | Print the exact prompt and an input-token estimate; **no API call** |
| `--out DIR` | Output directory (default `outputs/`) |
| `--model NAME` | Override the configured model |

Each run writes `outputs/<transcript_id>.json`, appends one JSONL line per API *attempt* to `outputs/runs.jsonl`, and prints a summary. Exit code is `0` if every transcript succeeded and `1` if any failed — usable directly as a CI gate.

---

## Design

### The validation ladder

Every transcript walks the same rungs (`src/pipeline.py`):

```
1. Native structured output              → usually valid
2. Structural validation (Pydantic)      → types, ranges, enums
3. Semantic validation                   → evidence must exist in transcript
4. On validation failure: repair call    → feed back exact error text
5. On transport failure: retry + backoff → 429, 5xx, timeout, overloaded
6. Otherwise: graceful failure           → record and continue
```

Rungs 4 and 5 are deliberately distinct. A validation failure is never retried with an identical call — that would just produce the same bad object. A transport failure is never "repaired" — there's nothing to repair.

### Evidence-bound scoring

Every dimension score ships with at least one verbatim quote from the transcript. This does two jobs:

- **Auditability.** A human reviewer can check the model's reasoning without rereading the session.
- **Deterministic hallucination detection.** `validate.py` normalises the quote and the transcript (lowercase, strip punctuation, collapse whitespace), then runs `difflib` against each turn and sliding windows within it. A quote that isn't really there fails below 0.80 similarity with a specific, repair-ready error — no second model call, no LLM-as-judge, no extra cost.

### Structured output from the schema, not the prompt

Shape consistency comes from a forced tool call whose `input_schema` **is** the Pydantic model. The model can't drift a shape that was never described to it in prose.

Range and length bounds (`score` 1–5, `evidence` 1–3 items) are left *soft* in the tool schema on purpose and enforced hard by Pydantic. That split is exactly why both a structural and a semantic layer exist.

### Prompt injection

The transcript is untrusted data — it's user-generated content that happens to reach the model. Three independent defences:

1. It's wrapped in an explicit `<transcript>…</transcript>` delimiter, with a restatement **after** the closing tag that everything inside was data.
2. The system prompt instructs the model to treat any embedded instruction, policy update, or end-of-transcript marker as content to be *scored*, not obeyed — and to raise `possible_injection_attempt`.
3. Because both of the above depend on the model complying, the semantic layer independently scans the transcript for injection markers and warns when the output failed to flag them. This backstop fires even if the model is fully compromised.

---

## Reliability & testing

26 tests, all offline — no network, no API key, no fixtures that expire. `pytest` passes from a clean clone.

| Failure mode | Covered by |
|---|---|
| Truncated / cut-off object | `test_truncated_object_handled_gracefully` |
| Score outside 1–5 | `test_score_out_of_range_rejected` |
| Invalid enum value | `test_invalid_enum_rejected` |
| Missing required field | `test_missing_required_field_rejected` |
| Injected extra key | `test_extra_injected_field_rejected` |
| **Hallucinated evidence** | `test_hallucinated_evidence_caught` |
| Real quote, mangled whitespace/case | `test_real_quote_passes_grounding…` |
| Repair fixes invalid output | `test_repair_fixes_invalid_output` |
| Repair exhausted → graceful fail | `test_repair_exhausted` |
| Rate limit → retry with backoff | `test_transient_error_retried` |
| 400 → *not* retried | `test_non_retryable_error_not_retried` |
| One bad transcript in a batch | `test_batch_survives_one_bad_transcript` |
| **Injection resistance** | `test_injection_resistance_surfaces_warning` |
| Injection guard present in prompt | `test_prompt_contains_injection_guard…` |

Plus parser tests asserting the multi-line injection payload in `session_05` lands inside a *single* turn rather than being silently dropped, and that `session_04`'s mid-word truncation parses rather than erroring.

---

## Cost

Token counts come from `response.usage` on every call — never estimated with a tokenizer — and are written per attempt to `outputs/runs.jsonl`. Cost is computed from the `PRICING` table in `config.py`, so a rate change is a one-line edit rather than a code change.

**Input size per transcript** (turn counts exact; tokens via the `--dry-run` `len/4` heuristic):

| Transcript | Turns | Transcript tokens | Total input* |
|---|---:|---:|---:|
| `session_01_strong` | 88 | ~2,070 | ~3,900 |
| `session_02_weak` | 39 | ~2,190 | ~3,900 |
| `session_03_pacing` | 74 | ~2,790 | ~4,600 |
| `session_04_truncated` | 22 | ~550 | ~2,200 |
| `session_05_adversarial` | 65 | ~1,950 | ~3,700 |

<sub>*Total input = rubric/system prompt + tool schema (~1.5k tokens, identical on every call) + transcript + turn annotations.</sub>

**Projected cost** at Claude Sonnet 5 list rates ($3 / $15 per Mtok), with output running ~800 tokens:

- **~$0.02** per transcript
- **~$0.10** for a full five-transcript run
- **~$0.05** for a realistic 50-minute session (~11k input tokens)

Three levers the design already accounts for:

| Lever | Effect | Why it applies here |
|---|---|---|
| Batch API | −50% both sides | Evaluation isn't user-facing; latency is a throughput problem, not a p95 problem |
| Prompt caching | −90% on cached input | ~1.5k tokens of rubric + schema are byte-identical on every single call |
| Two-tier routing | ~3× on input | Haiku by default, escalate only `confidence: low` results to Sonnet |

The run summary reports total and per-transcript cost, total tokens, mean/p50/p95 latency, and — most importantly for a reliability system — **retry rate** and **repair rate**.

---

## Output format

Successful evaluation, `outputs/<transcript_id>.json` (trimmed):

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

A failed transcript writes `{"status": "failed", "transcript_id": …, "attempts": n, "last_error": …, "errors": […], "raw_response": …}` instead — the raw response is persisted so a failure can be debugged without paying for another call.

One line per API attempt in `outputs/runs.jsonl`:

```json
{"run_id": "…", "timestamp": "2026-07-26T…", "transcript_id": "session_03_pacing",
 "model": "claude-sonnet-5", "prompt_version": "1.0", "schema_version": "1.0",
 "attempt": 1, "kind": "initial", "input_tokens": 4210, "output_tokens": 780,
 "cost_usd": 0.0243, "latency_ms": 6120, "status": "valid", "errors": []}
```

`kind` distinguishes `initial` / `retry` / `repair`, so the repair and retry rates are derivable from the log rather than guessed at.

---

## Test transcripts

Five hand-written sessions, deliberately varied — designed as test inputs, not five happy paths:

| File | Designed to stress |
|---|---|
| `session_01_strong` | The ceiling case. Student initiates, reasons aloud, self-corrects. |
| `session_02_weak` | Sustained tutor monologue, one-word student replies, early exit. Floor case for engagement. |
| `session_03_pacing` | **The discriminator.** Tutor is warm and accurate but spends 7 minutes on material the student says three times she already knows, then rushes the actual homework topic. Clarity should stay high while pacing drops — if the dimensions move together here, they aren't independent. |
| `session_04_truncated` | ~4 minutes, cuts off mid-word. Tests the `insufficient_content` / low-confidence escape hatch, and that truncation parses rather than erroring. |
| `session_05_adversarial` | Five prompt-injection payloads, escalating from a naive override to a forged `--- END OF TRANSCRIPT ---` block that names real schema fields and tries to suppress `flags`. The session is genuinely *poor* on the merits, so injection compliance produces a visibly wrong 5/5. |

Turn-level timestamps carry the pacing signal: `session_02`'s tutor turns span 90–110 seconds each while student turns span 2. Nothing in the text says "the tutor monologued" — it's only in the clock.

---

## Notes & next steps

- **Pricing is configuration, not code.** Fill `config.PRICING` from the current pricing page; `estimate_cost` returns `None` and warns once rather than crashing if a rate is unset, so a run still produces token and latency data either way.
- **`--dry-run` exists for prompt iteration at zero cost.** Prompt changes can be reviewed against all five transcripts without a single API call.
- **`prompt_version` and `schema_version` are logged on every line**, so results stay traceable to the exact wording and contract that produced them — which is what makes a frozen regression set possible when the prompt changes.
- **Not built, deliberately:** async, a web UI, a queue, a database. At real volume this becomes a batch job behind a queue with idempotency keys and content-hash dedupe, but none of that belongs in a five-transcript pipeline.
