"""
pipeline.py — orchestrates the full pipeline:
  fetch -> chunk (+ table extraction) -> generate -> verify -> dedup ->
  difficulty -> output (CSV/JSONL)

Run as:
    python src/pipeline.py
    python src/pipeline.py --force-refetch
    python src/pipeline.py --skip-verification   (debugging only — see warning below)

This module intentionally does the minimum amount of "thinking" itself —
each stage is a single call into the already-tested module that owns that
stage's logic (fetch_filing, chunker, table_extractor, generator, verifier,
dedup, difficulty). pipeline.py's job is sequencing, logging progress
between stages, and assembling the final output rows.
"""

from __future__ import annotations

import sys

# Printed BEFORE the heavy imports below (sentence-transformers pulls in
# torch/transformers, which can take 30-90s on a fresh environment with no
# GPU and no warm import cache). Without this, the terminal looks hung
# during that window with zero feedback.
print("Starting pipeline — loading models and dependencies (this can take 30-90s on first run)...", flush=True)

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from config import CONFIG, get_path, setup_logging
from fetch_filing import load_filing_html
from chunker import parse_filing_to_chunks, save_chunks
from table_extractor import extract_tables, save_tables, table_to_passage_text, ExtractedTable
from chunker import Chunk
from generator import generate_all, GeneratedQA, _build_client as build_groq_client
from verifier import verify_all, _build_hf_client, VerificationResult
from dedup import deduplicate
from difficulty import label_all

print("Dependencies loaded.", flush=True)

logger = setup_logging(__name__)


def _tables_as_pseudo_chunks(tables: list[ExtractedTable]) -> list[Chunk]:
    """
    Adapts ExtractedTable objects into the Chunk shape generator.py expects,
    so generation can run over tables with the same code path used for
    narrative chunks. Per spec, tables are the PRIMARY source for
    numeric_calculation and comparison questions, so they need to flow
    through generation just like narrative chunks do — table_extractor.py
    intentionally keeps them structured (not flattened) for traceability,
    but generation needs *some* text representation to pass to the LLM,
    which table_to_passage_text() provides.
    """
    pseudo_chunks = []
    for t in tables:
        text = table_to_passage_text(t)
        pseudo_chunks.append(
            Chunk(
                chunk_id=t.table_id,
                heading_path=t.heading_path,
                part=t.part,
                item=t.item,
                text=text,
                token_estimate=len(text.split()),
                order_index=1_000_000 + t.order_index,  # keep tables ordered after narrative chunks
            )
        )
    return pseudo_chunks


def _build_output_row(
    qa: GeneratedQA, verification: VerificationResult, difficulty: str
) -> dict:
    """Flattens a GeneratedQA + VerificationResult + difficulty label into
    one CSV-ready row, matching the output schema in spec section 8 (with
    nested fields flattened per the spec's explicit CSV column-naming
    instruction: source_item, source_section, context_recall_score, etc.)."""
    return {
        "id": qa.id,
        "question": qa.question,
        "answer": qa.answer,
        "source_passage": qa.source_passage,
        "source_item": qa.item or "",
        "source_section": qa.heading_path,
        # The spec's example schema includes a "page" field under
        # source_location; raw EDGAR .htm has no native page numbers (that's
        # a PDF-print artifact), so we leave it blank rather than fabricate one.
        "source_page": "",
        "question_type": qa.question_type,
        "difficulty": difficulty,
        "context_recall_score": round(verification.context_recall_score, 4),
        "faithfulness_pass": verification.faithfulness_pass,
        "numeric_check_pass": verification.numeric_check_pass,
        "verbatim_match_score": round(qa.verbatim_match_score, 4),
        "verifier_confidence": verification.verifier_confidence or "",
        "verifier_reason": verification.verifier_reason,
        "source_chunk_ids": ";".join(qa.source_chunk_ids),
    }


def write_output(rows: list[dict], out_path: Path) -> None:
    out_cfg = CONFIG["output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_cfg["format"] == "jsonl":
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    else:
        if not rows:
            logger.warning("No rows to write — writing empty CSV with no data rows.")
            out_path.write_text("", encoding="utf-8")
            return
        fieldnames = list(rows[0].keys())
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    logger.info(f"Wrote {len(rows)} rows to {out_path}")


def run_pipeline(force_refetch: bool = False, skip_verification: bool = False) -> Path:
    out_cfg = CONFIG["output"]

    # --- Stage 1: fetch ---
    logger.info("=== Stage 1/6: Fetch filing ===")
    html = load_filing_html(force_refetch=force_refetch)

    # --- Stage 2: chunk + extract tables ---
    logger.info("=== Stage 2/6: Chunk + extract tables ===")
    chunks = parse_filing_to_chunks(html)
    save_chunks(chunks)
    tables = extract_tables(html)
    save_tables(tables)
    table_chunks = _tables_as_pseudo_chunks(tables)
    all_chunks = chunks + table_chunks
    logger.info(f"Total generation units: {len(chunks)} narrative chunks + {len(table_chunks)} tables")

    # --- Stage 3: generate ---
    logger.info("=== Stage 3/6: Generate QA candidates (Groq llama-3.1-8b-instant) ===")
    groq_client = build_groq_client()
    candidates = generate_all(all_chunks, client=groq_client)
    if not candidates:
        logger.error("Generation produced zero candidates — aborting before verification.")
        sys.exit(1)

    # --- Stage 4: verify ---
    logger.info("=== Stage 4/6: Verify (HF Inference Providers gemma-3-27b-it + numeric_check) ===")
    if skip_verification:
        logger.warning(
            "--skip-verification was passed: marking all pairs as kept WITHOUT running "
            "the faithfulness/numeric checks. This is for pipeline-wiring debugging only — "
            "do NOT use the resulting dataset as a verified deliverable."
        )
        from verifier import VerificationResult as VR
        verified = [
            (qa, VR(faithfulness_pass=None, numeric_check_pass=None, context_recall_score=1.0,
                    verifier_confidence=None, verifier_reason="SKIPPED", keep=True))
            for qa in candidates
        ]
    else:
        hf_client = _build_hf_client()
        verified = verify_all(candidates, client=hf_client)

    kept_pairs = [(qa, v) for qa, v in verified if v.keep]
    logger.info(f"{len(kept_pairs)}/{len(verified)} candidates passed verification")

    if not kept_pairs:
        logger.error("No candidates survived verification — aborting before dedup.")
        sys.exit(1)

    # --- Stage 5: dedup ---
    logger.info("=== Stage 5/6: Deduplicate (bge-small-en-v1.5 cosine similarity) ===")
    qa_only = [qa for qa, _ in kept_pairs]
    verification_by_id = {qa.id: v for qa, v in kept_pairs}
    dedup_result = deduplicate(qa_only)

    # --- Stage 6: difficulty + output ---
    logger.info("=== Stage 6/6: Difficulty labeling + write output ===")
    difficulty_labels = label_all(dedup_result.kept)

    rows = [
        _build_output_row(qa, verification_by_id[qa.id], difficulty_labels[qa.id])
        for qa in dedup_result.kept
    ]

    out_path = get_path("output_dir") / out_cfg["file_name"]
    write_output(rows, out_path)

    if len(rows) < out_cfg["min_rows_required"]:
        logger.warning(
            f"Final dataset has {len(rows)} rows, below the target of "
            f"{out_cfg['min_rows_required']}. Consider: lowering "
            f"chunking.min_chunk_tokens to produce more chunks, raising "
            f"generation.questions_per_call, or running against additional filings "
            f"(see README scaling note)."
        )
    else:
        logger.info(f"Target met: {len(rows)} rows >= {out_cfg['min_rows_required']} required.")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="10-K Financial QA Generation Pipeline")
    parser.add_argument(
        "--force-refetch", action="store_true",
        help="Re-fetch the filing from EDGAR even if a cached copy exists.",
    )
    parser.add_argument(
        "--skip-verification", action="store_true",
        help="DEBUG ONLY: skip the verification stage entirely. Output will NOT be a "
             "verified deliverable if this flag is used.",
    )
    args = parser.parse_args()

    out_path = run_pipeline(
        force_refetch=args.force_refetch,
        skip_verification=args.skip_verification,
    )
    print(f"\nPipeline complete. Output written to: {out_path}")


if __name__ == "__main__":
    main()
