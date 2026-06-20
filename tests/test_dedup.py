"""
test_dedup.py — deduplication logic, tested with a deterministic mocked
embedding function rather than the real bge-small-en-v1.5 model.

Why mock: dedup.py's own correctness (pairwise cosine similarity threshold,
"drop the later-generated duplicate", building dropped_reasons) is
independent of which embedding model produced the vectors. Loading the real
bge-small-en-v1.5 model requires a network call to Hugging Face the first
time it runs, which we don't want as a hard requirement for the unit test
suite. Tests that need the REAL model are marked `requires_model` and
skipped by default (see test_deduplicate_with_real_embeddings_smoke below).

The mock assigns each question a hand-picked unit vector based on simple
keyword matching, so we can construct exact, predictable cosine similarities
(0.0, ~0.95, 1.0, etc.) instead of relying on real semantic embeddings.
"""

from __future__ import annotations

import numpy as np
import pytest

import embeddings
from generator import GeneratedQA
from dedup import deduplicate


def _qa(id_, question, question_type="fact_extraction", chunk_ids=None):
    """Build a minimal GeneratedQA for dedup tests — fields irrelevant to
    dedup logic (answer, source_passage, heading_path, etc.) get simple
    placeholder values."""
    return GeneratedQA(
        id=id_,
        question=question,
        answer="placeholder answer",
        source_passage="placeholder source passage text",
        question_type=question_type,
        heading_path="Item 7 > OVERVIEW",
        part="PART I",
        item="Item 7",
        source_chunk_ids=chunk_ids or ["chunk_00000"],
        verbatim_match_score=1.0,
    )


@pytest.fixture
def mock_embeddings(monkeypatch):
    """
    Patches dedup.embed_texts (imported name inside dedup module) with a
    deterministic function: questions containing 'revenue' get vector
    [1, 0, 0]-ish, questions containing 'margin' get [0, 1, 0]-ish, and a
    'near_revenue' marker produces a vector at a controlled angle from the
    pure revenue vector so we can test threshold boundary behavior exactly.
    """
    def fake_embed_texts(texts):
        vecs = []
        for t in texts:
            t_lower = t.lower()
            if "near_revenue_dup" in t_lower:
                # cos_sim with pure revenue vector ~0.95 (above 0.90 threshold)
                vecs.append(np.array([0.95, np.sqrt(1 - 0.95**2), 0.0]))
            elif "barely_revenue_dup" in t_lower:
                # cos_sim with pure revenue vector ~0.85 (below 0.90 threshold)
                vecs.append(np.array([0.85, np.sqrt(1 - 0.85**2), 0.0]))
            elif "revenue" in t_lower:
                vecs.append(np.array([1.0, 0.0, 0.0]))
            elif "margin" in t_lower:
                vecs.append(np.array([0.0, 1.0, 0.0]))
            else:
                vecs.append(np.array([0.0, 0.0, 1.0]))
        arr = np.array(vecs)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / norms

    monkeypatch.setattr("dedup.embed_texts", fake_embed_texts)


# ---------------------------------------------------------------------------
# Core dedup behavior
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty_result(mock_embeddings):
    result = deduplicate([])
    assert result.kept == []
    assert result.dropped == []
    assert result.dropped_reasons == {}


def test_no_duplicates_keeps_everything(mock_embeddings):
    qas = [
        _qa("1", "What was total revenue?"),
        _qa("2", "What was operating margin?"),
    ]
    result = deduplicate(qas)
    assert [qa.id for qa in result.kept] == ["1", "2"]
    assert result.dropped == []


def test_near_identical_questions_drops_the_later_one(mock_embeddings):
    """Two questions both mapping to the 'pure revenue' vector are
    identical (cosine_sim=1.0, well above threshold) — the LATER one
    (id=2) must be dropped, per spec section 6."""
    qas = [
        _qa("1", "What was total revenue in fiscal 2025?"),
        _qa("2", "What was the company revenue for fiscal year 2025?"),
    ]
    result = deduplicate(qas)
    assert [qa.id for qa in result.kept] == ["1"]
    assert [qa.id for qa in result.dropped] == ["2"]
    assert "duplicate of id=1" in result.dropped_reasons["2"]


def test_distinct_topics_not_deduplicated(mock_embeddings):
    """Revenue and margin questions are orthogonal (cosine_sim=0.0) and
    must both survive."""
    qas = [
        _qa("1", "What was total revenue?"),
        _qa("2", "What was operating margin?"),
        _qa("3", "What were total expenses for the period reported?"),
    ]
    result = deduplicate(qas)
    assert len(result.kept) == 3
    assert result.dropped == []


def test_similarity_above_threshold_is_dropped(mock_embeddings):
    """cosine_sim ~0.95 > 0.90 threshold (from config.yaml) -> dropped."""
    qas = [
        _qa("1", "What was total revenue?"),
        _qa("2", "near_revenue_dup question about the same figure"),
    ]
    result = deduplicate(qas)
    assert [qa.id for qa in result.kept] == ["1"]
    assert [qa.id for qa in result.dropped] == ["2"]


def test_similarity_below_threshold_is_kept(mock_embeddings):
    """cosine_sim ~0.85 < 0.90 threshold -> both kept, even though they're
    fairly similar — this pins down the exact boundary behavior."""
    qas = [
        _qa("1", "What was total revenue?"),
        _qa("2", "barely_revenue_dup question that is only somewhat related"),
    ]
    result = deduplicate(qas)
    assert len(result.kept) == 2
    assert result.dropped == []


def test_three_way_duplicate_chain_keeps_only_first(mock_embeddings):
    """If three questions are all mutually near-identical, only the first
    (earliest-generated) should survive; the second and third should both
    be recorded as duplicates of the first (transitivity via the i<j loop:
    once dropped_mask[i] would be set, but here i=0 is never dropped, so
    both later items compare against item 0 and get dropped)."""
    qas = [
        _qa("1", "What was total revenue?"),
        _qa("2", "What was the company's total revenue?"),
        _qa("3", "What was revenue for the company overall?"),
    ]
    result = deduplicate(qas)
    assert [qa.id for qa in result.kept] == ["1"]
    assert {qa.id for qa in result.dropped} == {"2", "3"}


def test_dedup_preserves_kept_qa_objects_unmodified(mock_embeddings):
    """Kept GeneratedQA objects should be the same objects/values that went
    in — dedup shouldn't mutate fields like question_type or answer."""
    qa = _qa("1", "What was total revenue?", question_type="numeric_calculation")
    result = deduplicate([qa])
    assert result.kept[0] is qa
    assert result.kept[0].question_type == "numeric_calculation"


def test_dropped_reasons_only_contains_dropped_ids(mock_embeddings):
    qas = [
        _qa("1", "What was total revenue?"),
        _qa("2", "What was the company's total revenue?"),
        _qa("3", "What was operating margin?"),
    ]
    result = deduplicate(qas)
    assert set(result.dropped_reasons.keys()) == {qa.id for qa in result.dropped}
    assert "1" not in result.dropped_reasons
    assert "3" not in result.dropped_reasons


# ---------------------------------------------------------------------------
# Real-model smoke test (skipped by default — requires network access to
# download bge-small-en-v1.5 from Hugging Face on first run)
# ---------------------------------------------------------------------------

@pytest.mark.requires_model
def test_deduplicate_with_real_embeddings_smoke():
    """Sanity check against the actual bge-small-en-v1.5 model. Not run in
    default `pytest` invocations (see pytest.ini markers) since it requires
    downloading the model. Run explicitly with:
        pytest tests/test_dedup.py -m requires_model
    """
    qas = [
        _qa("1", "What was Microsoft's total revenue in fiscal year 2025?"),
        _qa("2", "What was Microsoft's total revenue for FY2025?"),
        _qa("3", "What was the operating margin reported for the segment?"),
    ]
    result = deduplicate(qas)
    # The two revenue-phrased questions should be recognized as near-duplicates
    # by a real semantic embedding model; the margin question should survive.
    assert len(result.kept) == 2
    kept_ids = {qa.id for qa in result.kept}
    assert "3" in kept_ids
