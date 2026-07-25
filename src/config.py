"""Central configuration: model id, pricing, retry limits, and paths.

Intentionally logic-free — just module-level constants imported elsewhere.
Paths are derived from this file's location so nothing hard-codes an absolute
local path into the repo.
"""

from pathlib import Path

# --- Model -------------------------------------------------------------------
# Pinned to an explicit, non-floating model string. Do not swap for an alias
# like "claude-sonnet-latest": a floating alias would silently change behaviour
# between runs and make the committed outputs/ irreproducible.
MODEL: str = "claude-sonnet-5"

# --- Pricing -----------------------------------------------------------------
# USD per million tokens, per model. Left as None on purpose: prices are not
# invented here. Fill these in before relying on cost figures; estimate_cost()
# returns None (and warns once) while a value is missing rather than crashing.
# TODO: fill from https://www.anthropic.com/pricing
PRICING: dict[str, dict[str, float | None]] = {
    "claude-sonnet-5": {
        "input_per_mtok": None,
        "output_per_mtok": None,
    },
}

# --- Generation --------------------------------------------------------------
MAX_TOKENS: int = 4096
TEMPERATURE: int = 0  # deterministic scoring; part of run reproducibility

# --- Reliability limits ------------------------------------------------------
MAX_RETRIES: int = 2          # transient transport errors only (429/5xx/timeout)
MAX_REPAIR_ATTEMPTS: int = 1  # follow-up calls to fix invalid model output

# --- Semantic validation -----------------------------------------------------
# Minimum difflib similarity for an evidence quote to count as grounded in the
# transcript. Below this, the quote is treated as hallucinated.
EVIDENCE_MATCH_THRESHOLD: float = 0.80

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR: Path = PROJECT_ROOT / "transcripts"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
RUNS_LOG: Path = OUTPUTS_DIR / "runs.jsonl"
