"""Token accounting and the per-attempt run log.

Two responsibilities:

* ``estimate_cost`` turns real token counts into dollars using ``config.PRICING``.
  Prices are intentionally ``None`` in the repo, so this returns ``None`` (and
  warns once per model) rather than inventing a number.
* ``append_run_records`` writes **one JSON line per API attempt** to
  ``outputs/runs.jsonl`` — initial, retry, and repair calls each get their own
  line with their own token count — and ``summarise_run`` rolls those lines up.

Retry rate and repair rate are computed and surfaced prominently: they are what
show the pipeline is understood as a reliability system, not just a wrapper.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config import MODEL, PRICING, RUNS_LOG
from src.pipeline import PipelineResult
from src.prompts import PROMPT_VERSION

logger = logging.getLogger(__name__)

_MISSING_PRICE_WARNED: set[str] = set()  # models we've already warned about


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """USD cost for one call, or None if the model's price is not filled in.

    Warns once per model rather than crashing, so a run still produces token and
    latency numbers before pricing is configured.
    """
    prices = PRICING.get(model)
    if not prices or prices.get("input_per_mtok") is None or prices.get("output_per_mtok") is None:
        if model not in _MISSING_PRICE_WARNED:
            logger.warning("no pricing for model %r; cost_usd will be null (fill config.PRICING)", model)
            _MISSING_PRICE_WARNED.add(model)
        return None
    return (
        input_tokens / 1_000_000 * prices["input_per_mtok"]
        + output_tokens / 1_000_000 * prices["output_per_mtok"]
    )


def _records_for(
    result: PipelineResult,
    *,
    run_id: str,
    timestamp: str,
    model: str,
    prompt_version: str,
    schema_version: str,
) -> list[dict]:
    """One log record per API attempt in *result*."""
    records: list[dict] = []
    for i, attempt in enumerate(result.attempts, start=1):
        response = attempt.response
        if response is not None:
            input_tokens = response.input_tokens
            output_tokens = response.output_tokens
            record = {
                "model": response.model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": estimate_cost(response.model, input_tokens, output_tokens),
                "latency_ms": response.latency_ms,
                "errors": [],
            }
        else:
            # The call raised (transient/terminal); no tokens to report.
            record = {
                "model": model,
                "input_tokens": None,
                "output_tokens": None,
                "cost_usd": None,
                "latency_ms": None,
                "errors": [attempt.error] if attempt.error else [],
            }
        records.append(
            {
                "run_id": run_id,
                "timestamp": timestamp,
                "transcript_id": result.transcript_id,
                "prompt_version": prompt_version,
                "schema_version": schema_version,
                "attempt": i,
                "kind": attempt.kind,
                "status": result.status,  # transcript's final outcome, repeated per line
                **record,
            }
        )
    return records


def append_run_records(
    result: PipelineResult,
    *,
    run_id: str,
    timestamp: str,
    model: str = MODEL,
    prompt_version: str = PROMPT_VERSION,
    schema_version: str = "1.0",
    path: Path = RUNS_LOG,
) -> list[dict]:
    """Append one JSONL line per API attempt and return the records written."""
    records = _records_for(
        result,
        run_id=run_id,
        timestamp=timestamp,
        model=model,
        prompt_version=prompt_version,
        schema_version=schema_version,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return records


def _percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolated percentile of a sorted list (empty -> None)."""
    if not values:
        return None
    k = (len(values) - 1) * (pct / 100)
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    if lo == hi:
        return float(values[lo])
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def _read_rows(run_id: str, path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("run_id") == run_id:
            rows.append(row)
    return rows


def summarise_run(run_id: str, *, path: Path = RUNS_LOG) -> dict:
    """Roll up one run's log lines into headline reliability and cost numbers."""
    rows = _read_rows(run_id, path)

    by_transcript: dict[str, list[dict]] = {}
    for row in rows:
        by_transcript.setdefault(row["transcript_id"], []).append(row)
    n = len(by_transcript)

    statuses = [group[0]["status"] for group in by_transcript.values()]
    counts = {
        "valid": statuses.count("valid"),
        "repaired": statuses.count("repaired"),
        "failed": statuses.count("failed"),
    }
    succeeded = counts["valid"] + counts["repaired"]

    had_retry = sum(any(r["kind"] == "retry" for r in g) for g in by_transcript.values())
    had_repair = sum(any(r["kind"] == "repair" for r in g) for g in by_transcript.values())

    costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
    total_cost = sum(costs) if costs else None

    latencies = sorted(r["latency_ms"] for r in rows if r["latency_ms"] is not None)
    total_input = sum(r["input_tokens"] or 0 for r in rows)
    total_output = sum(r["output_tokens"] or 0 for r in rows)

    return {
        "run_id": run_id,
        "transcripts": n,
        "api_calls": len(rows),
        "status_counts": counts,
        "succeeded": succeeded,
        "failed": counts["failed"],
        "retry_rate": (had_retry / n) if n else 0.0,
        "repair_rate": (had_repair / n) if n else 0.0,
        "total_cost_usd": total_cost,
        "cost_per_transcript_usd": (total_cost / n) if (total_cost is not None and n) else None,
        "total_tokens": total_input + total_output,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "latency_ms": {
            "mean": (sum(latencies) / len(latencies)) if latencies else None,
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
        },
    }
