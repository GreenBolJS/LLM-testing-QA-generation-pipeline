"""
dedup.py — deduplicates surviving QA pairs via bge-small-en-v1.5 embeddings.

Per spec section 6: embed all surviving questions, compute pairwise cosine
similarity, and for any pair > 0.90 similarity, drop the later-generated
duplicate. The spec calls out that a meaningful drop rate here is expected,
since the same disclosure (e.g. a headline revenue growth %) often appears
in both an Overview bullet and the detailed segment narrative — so this is
NOT a bug if it removes a large fraction of pairs.

Implementation note: we dedup on the embedded *question* text (not the
answer or source_passage), per spec section 6's "Embed all surviving
questions" wording — two differently-phrased questions asking the same
underlying fact are the duplicates we're trying to catch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import CONFIG, setup_logging
from embeddings import embed_texts
from generator import GeneratedQA

logger = setup_logging(__name__)


@dataclass
class DedupResult:
    kept: list[GeneratedQA]
    dropped: list[GeneratedQA]
    dropped_reasons: dict[str, str]  # qa.id -> "duplicate of <other_id>"


def deduplicate(qa_pairs: list[GeneratedQA]) -> DedupResult:
    """
    qa_pairs is assumed to be in generation order (earlier items generated
    first). For any pair whose question-embedding cosine similarity exceeds
    embedding.dedup_similarity_threshold, the LATER item (higher index) is
    dropped, per spec section 6 ("drop the later-generated duplicate").
    """
    threshold = CONFIG["embedding"]["dedup_similarity_threshold"]

    if not qa_pairs:
        return DedupResult(kept=[], dropped=[], dropped_reasons={})

    questions = [qa.question for qa in qa_pairs]
    vectors = embed_texts(questions)  # already normalized -> dot product = cosine sim

    n = len(qa_pairs)
    dropped_mask = [False] * n
    dropped_reasons: dict[str, str] = {}

    # Pairwise comparison, O(n^2) — fine for the dataset sizes this pipeline
    # targets (hundreds, not millions, of candidate QA pairs per filing).
    sim_matrix = vectors @ vectors.T

    for i in range(n):
        if dropped_mask[i]:
            continue
        for j in range(i + 1, n):
            if dropped_mask[j]:
                continue
            sim = float(sim_matrix[i, j])
            if sim > threshold:
                dropped_mask[j] = True
                dropped_reasons[qa_pairs[j].id] = (
                    f"duplicate of id={qa_pairs[i].id} (cosine_sim={sim:.3f}): "
                    f"{qa_pairs[i].question!r}"
                )

    kept = [qa for qa, dropped in zip(qa_pairs, dropped_mask) if not dropped]
    dropped = [qa for qa, dropped in zip(qa_pairs, dropped_mask) if dropped]

    drop_rate = len(dropped) / n * 100 if n else 0.0
    logger.info(
        f"Dedup complete: kept {len(kept)}/{n} ({drop_rate:.1f}% dropped as near-duplicates, "
        f"threshold={threshold})"
    )

    return DedupResult(kept=kept, dropped=dropped, dropped_reasons=dropped_reasons)


if __name__ == "__main__":
    from chunker import load_chunks
    from generator import generate_all

    chunks = load_chunks()
    qa_pairs = generate_all(chunks)
    result = deduplicate(qa_pairs)
    print(f"Kept {len(result.kept)}, dropped {len(result.dropped)}")
    for qid, reason in list(result.dropped_reasons.items())[:5]:
        print(f"  {qid}: {reason}")
