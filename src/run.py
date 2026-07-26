"""CLI entrypoint: evaluate transcripts and print a run summary.

Run as ``python -m src.run --all`` (or ``--transcript PATH``). One ``run_id`` per
invocation is stamped on every log line. Each transcript writes
``outputs/<transcript_id>.json``; a per-transcript line prints as it goes and the
full reliability/cost summary prints at the end. Exit code is 0 if every
transcript succeeded, 1 if any failed — usable as a CI gate.

This is the only module that prints; libraries use ``logging``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.config import MODEL, OUTPUTS_DIR, TRANSCRIPTS_DIR
from src.cost import append_run_records, summarise_run
from src.pipeline import PipelineResult, evaluate_transcript
from src.prompts import SYSTEM_PROMPT, build_user_prompt
from src.transcript import Transcript, load_all, load_transcript

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.run",
        description="Evaluate tutoring-session transcripts with the LLM pipeline.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--all", action="store_true", help="evaluate every transcript in transcripts/")
    target.add_argument("--transcript", metavar="PATH", help="evaluate a single transcript file")
    parser.add_argument("--out", default=str(OUTPUTS_DIR), help="output directory (default: outputs/)")
    parser.add_argument("--model", default=MODEL, help="override the model id (default: %(default)s)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse and print the exact prompt plus an estimated input-token count; "
        "makes NO API call (free prompt iteration)",
    )
    return parser.parse_args(argv)


def _load(args: argparse.Namespace) -> list[Transcript]:
    if args.all:
        return load_all(TRANSCRIPTS_DIR)
    return [load_transcript(args.transcript)]


def _estimate_tokens(text: str) -> int:
    """Rough, tokenizer-free input estimate for dry runs (~4 chars/token)."""
    return max(1, len(text) // 4)


def _dry_run(transcripts: list[Transcript]) -> int:
    for transcript in transcripts:
        user = build_user_prompt(transcript)
        estimate = _estimate_tokens(SYSTEM_PROMPT) + _estimate_tokens(user)
        print(f"\n===== {transcript.id} =====")
        print("----- SYSTEM -----")
        print(SYSTEM_PROMPT)
        print("----- USER -----")
        print(user)
        print(f"~estimated input tokens: {estimate} (rough len/4 heuristic; no API call made)")
    return 0


def _write_output(out_dir: Path, result: PipelineResult) -> None:
    path = out_dir / f"{result.transcript_id}.json"
    path.write_text(
        json.dumps(result.to_output(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _scores(result: PipelineResult) -> str:
    if result.evaluation is None:
        return "—"
    ev = result.evaluation
    return f"E{ev.engagement.score} C{ev.clarity.score} P{ev.pacing.score}"


def _print_transcript_line(result: PipelineResult, records: list[dict]) -> None:
    latency = sum(r["latency_ms"] for r in records if r["latency_ms"] is not None)
    costs = [r["cost_usd"] for r in records if r["cost_usd"] is not None]
    cost = f"${sum(costs):.4f}" if costs else "n/a"
    print(
        f"{result.transcript_id:<24} {result.status:<9} {_scores(result):<10} "
        f"{latency:>6}ms  {cost:>9}"
    )
    for warning in result.warnings:
        print(f"    ! {warning}")


def evaluate_all(
    transcripts: list[Transcript],
    *,
    model: str,
    out_dir: Path,
    run_id: str,
    timestamp: str,
) -> list[PipelineResult]:
    """Evaluate each transcript, logging and writing output as it goes.

    Wrapped in a broad ``except`` on purpose: a single transcript — even one that
    trips an unexpected bug — must never crash the batch.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[PipelineResult] = []
    for transcript in transcripts:
        try:
            result = evaluate_transcript(transcript, model=model)
        except Exception as exc:  # noqa: BLE001 — batch survival is the point
            logger.exception("unexpected error evaluating %s", transcript.id)
            result = PipelineResult(
                transcript_id=transcript.id,
                status="failed",
                evaluation=None,
                last_error=f"{type(exc).__name__}: {exc}",
            )
        records = append_run_records(result, run_id=run_id, timestamp=timestamp, model=model)
        _write_output(out_dir, result)
        _print_transcript_line(result, records)
        results.append(result)
    return results


def _fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def _print_summary(summary: dict) -> None:
    counts = summary["status_counts"]
    total_cost = summary["total_cost_usd"]
    per = summary["cost_per_transcript_usd"]
    latency = summary["latency_ms"]
    print("\n== run summary ==================================")
    print(f"run_id           {summary['run_id']}")
    print(f"transcripts      {summary['transcripts']}  (succeeded {summary['succeeded']}, failed {summary['failed']})")
    print(f"status           valid {counts['valid']}  repaired {counts['repaired']}  failed {counts['failed']}")
    print(f"API calls        {summary['api_calls']}")
    print(f"retry rate       {summary['retry_rate']:.0%}")
    print(f"repair rate      {summary['repair_rate']:.0%}")
    print(f"tokens           {summary['total_tokens']}  (in {summary['total_input_tokens']}, out {summary['total_output_tokens']})")
    print(f"latency ms       mean {_fmt_ms(latency['mean'])}  p50 {_fmt_ms(latency['p50'])}  p95 {_fmt_ms(latency['p95'])}")
    print(f"total cost       {'n/a (pricing not set)' if total_cost is None else f'${total_cost:.4f}'}")
    print(f"cost/transcript  {'n/a' if per is None else f'${per:.4f}'}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    transcripts = _load(args)

    if args.dry_run:
        return _dry_run(transcripts)

    run_id = uuid4().hex
    timestamp = datetime.now(timezone.utc).isoformat()
    results = evaluate_all(
        transcripts,
        model=args.model,
        out_dir=Path(args.out),
        run_id=run_id,
        timestamp=timestamp,
    )
    _print_summary(summarise_run(run_id))
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
