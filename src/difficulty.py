"""
difficulty.py — rule-based difficulty labeling, derived purely from
generation metadata (question_type + source chunk count), per spec
section 7. Never asks an LLM to self-report difficulty.

Rules (spec section 7):
  easy:   question_type = fact_extraction, single source chunk, answer is a
          direct quote/number.
  medium: question_type = numeric_calculation (one arithmetic step), OR
          fact_extraction requiring light synthesis across one paragraph.
  hard:   question_type = multi_step_reasoning or comparison, OR any
          question whose generation step used 2+ source chunks.

Note the overlap: fact_extraction can land in either easy or medium
depending on chunk count / synthesis, but our generation pipeline (per
config.yaml's multi_chunk_question_types) never sends fact_extraction
through the 2-chunk path, so in practice every fact_extraction candidate
we see is single-chunk. We still implement the full conditional rather than
assuming that's always true, so this module behaves correctly even if
generator.py's pairing config changes later.

The "hard if any question used 2+ source chunks" clause is checked FIRST
and overrides question_type, since the spec phrases it as an OR that can
promote an otherwise medium/easy type to hard.
"""

from __future__ import annotations

from config import CONFIG, setup_logging
from generator import GeneratedQA

logger = setup_logging(__name__)

VALID_LEVELS = {"easy", "medium", "hard"}


def label_difficulty(qa: GeneratedQA) -> str:
    """
    Returns one of "easy" / "medium" / "hard" for a single GeneratedQA,
    using only qa.question_type and len(qa.source_chunk_ids) — no LLM call,
    no use of qa.answer content, per the spec's "derive purely from
    generation metadata" instruction.
    """
    n_chunks = len(qa.source_chunk_ids)
    q_type = qa.question_type

    # Multi-chunk generation always promotes to hard, regardless of question_type,
    # per spec section 7's explicit OR clause.
    if n_chunks >= 2:
        return "hard"

    if q_type in ("multi_step_reasoning", "comparison"):
        return "hard"

    if q_type == "numeric_calculation":
        return "medium"

    if q_type == "fact_extraction":
        # Single chunk -> easy, per spec's explicit easy rule.
        # (The "light synthesis across one paragraph" medium case for
        # fact_extraction is metadata-indistinguishable from easy under a
        # single-chunk generation call, so without an additional signal we
        # default single-chunk fact_extraction to easy, matching the
        # spec's primary easy definition.)
        return "easy"

    # Unknown/unexpected question_type — log and fall back to medium rather
    # than silently mislabeling as easy or hard.
    logger.warning(f"Unrecognized question_type {q_type!r} for qa.id={qa.id}; defaulting to 'medium'")
    return "medium"


def label_all(qa_pairs: list[GeneratedQA]) -> dict[str, str]:
    """Returns a mapping of qa.id -> difficulty label for the whole list."""
    labels = {qa.id: label_difficulty(qa) for qa in qa_pairs}

    counts = {level: 0 for level in VALID_LEVELS}
    for level in labels.values():
        counts[level] += 1
    logger.info(f"Difficulty distribution: {counts}")

    return labels


if __name__ == "__main__":
    from chunker import load_chunks
    from generator import generate_all

    chunks = load_chunks()
    qa_pairs = generate_all(chunks)
    labels = label_all(qa_pairs)
    for qa in qa_pairs[:10]:
        print(f"[{labels[qa.id]:6s}] ({qa.question_type}, {len(qa.source_chunk_ids)} chunks) {qa.question}")
