"""
test_generator.py — tests for generator.py's pure-logic helpers: JSON
response parsing, verbatim/table-aware match scoring, passage truncation,
and chunk-group sampling. Does NOT call the real Groq API — all tests run
against the module's internal functions directly with synthetic input.

The table-aware verbatim matching tests are regression tests for the bug
where table-derived passages (pipe-delimited text) were almost always
rejected by the narrative-style fuzzy verbatim check, because LLMs
naturally paraphrase "Segment | 50000 | 45000" into prose like "Segment
revenue was 50000" rather than quoting the literal pipe format — this was
the root cause of numeric_calculation/comparison being severely
underrepresented in the final dataset despite tables being the spec's
designated primary source for those question types.
"""

from __future__ import annotations

import pytest

from chunker import Chunk
from generator import (
    _strip_markdown_fences,
    _parse_generation_response,
    _verbatim_match_score,
    _numbers_in_text,
    _is_table_derived,
    _verbatim_or_table_match_score,
    _truncate_to_char_budget,
    _sample_groups,
    _split_template,
)


# ---------------------------------------------------------------------------
# JSON response parsing
# ---------------------------------------------------------------------------

def test_strip_markdown_fences_removes_json_wrapper():
    raw = '```json\n[{"a": 1}]\n```'
    assert _strip_markdown_fences(raw) == '[{"a": 1}]'


def test_strip_markdown_fences_unaffected_when_absent():
    raw = '[{"a": 1}]'
    assert _strip_markdown_fences(raw) == raw


def test_parse_generation_response_clean_json():
    raw = '[{"question": "Q?", "answer": "A", "source_passage": "P", "question_type": "fact_extraction"}]'
    result = _parse_generation_response(raw)
    assert len(result) == 1
    assert result[0]["question"] == "Q?"


def test_parse_generation_response_with_markdown_fences():
    raw = '```json\n[{"question": "Q?", "answer": "A", "source_passage": "P", "question_type": "fact_extraction"}]\n```'
    result = _parse_generation_response(raw)
    assert len(result) == 1


def test_parse_generation_response_single_object_not_list():
    raw = '{"question": "Q?", "answer": "A", "source_passage": "P", "question_type": "fact_extraction"}'
    result = _parse_generation_response(raw)
    assert len(result) == 1
    assert result[0]["question"] == "Q?"


def test_parse_generation_response_recovers_from_stray_preamble():
    raw = 'Sure, here is the JSON:\n[{"question": "Q?", "answer": "A", "source_passage": "P", "question_type": "fact_extraction"}]'
    result = _parse_generation_response(raw)
    assert len(result) == 1


def test_parse_generation_response_unparseable_returns_empty_list():
    raw = "This is not JSON at all, sorry."
    result = _parse_generation_response(raw)
    assert result == []


# ---------------------------------------------------------------------------
# Narrative verbatim matching (existing behavior, unchanged)
# ---------------------------------------------------------------------------

def test_verbatim_match_exact_quote_scores_high():
    chunk_text = "Revenue increased to 50000 million driven by cloud growth this fiscal year."
    score = _verbatim_match_score("Revenue increased to 50000 million", chunk_text)
    assert score == 1.0


def test_verbatim_match_paraphrase_scores_low():
    chunk_text = "Revenue increased to 50000 million driven by cloud growth this fiscal year."
    score = _verbatim_match_score("Revenue was about fifty billion dollars", chunk_text)
    assert score < 0.9


def test_verbatim_match_empty_passage_scores_zero():
    assert _verbatim_match_score("", "Some chunk text here.") == 0.0


# ---------------------------------------------------------------------------
# Table-derived chunk detection
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str, text: str = "placeholder text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, heading_path="Item 7 > SEGMENT", part="PART I",
        item="Item 7", text=text, token_estimate=10, order_index=0,
    )


def test_is_table_derived_true_for_table_prefixed_id():
    chunk = _make_chunk("table_0042")
    assert _is_table_derived(chunk) is True


def test_is_table_derived_false_for_narrative_chunk_id():
    chunk = _make_chunk("chunk_00042")
    assert _is_table_derived(chunk) is False


# ---------------------------------------------------------------------------
# Number extraction for table-aware matching
# ---------------------------------------------------------------------------

def test_numbers_in_text_extracts_and_normalizes():
    text = "Segment | $50,000 | 45000.5"
    nums = _numbers_in_text(text)
    assert "50000" in nums
    assert "45000.5" in nums


def test_numbers_in_text_empty_for_no_numbers():
    assert _numbers_in_text("No figures mentioned here at all.") == set()


# ---------------------------------------------------------------------------
# Table-aware verbatim matching (regression tests for the underrepresentation bug)
# ---------------------------------------------------------------------------

TABLE_TEXT = (
    "Segment | FY2025 | FY2024\n"
    "Productivity and Business Processes | 50000 | 45000\n"
    "Intelligent Cloud | 70000 | 60000"
)


def test_table_aware_match_passes_legitimate_paraphrase():
    """A paraphrased prose description of a table row, with all the same
    numbers, must score >= 0.90 under table-aware matching, even though it
    would fail the narrative fuzzy-match check on the literal pipe format."""
    paraphrased = (
        "Productivity and Business Processes segment revenue was $50,000 "
        "million in FY2025 compared to $45,000 million in FY2024"
    )
    narrative_score = _verbatim_or_table_match_score(paraphrased, TABLE_TEXT, is_table=False)
    table_score = _verbatim_or_table_match_score(paraphrased, TABLE_TEXT, is_table=True)

    assert narrative_score < 0.90  # would have been dropped under the old logic
    assert table_score >= 0.90     # correctly passes under table-aware logic


def test_table_aware_match_catches_fabricated_number():
    """A claimed source_passage citing a number that does NOT appear
    anywhere in the original table must still score low — table-aware
    matching relaxes the literal-string requirement, but must not let
    genuine hallucinated numbers through."""
    hallucinated = "Productivity and Business Processes segment revenue was $99,999 million"
    score = _verbatim_or_table_match_score(hallucinated, TABLE_TEXT, is_table=True)
    assert score < 0.90


def test_table_aware_match_falls_back_when_no_numbers_claimed():
    """If source_passage has no numbers at all (unusual for a table-sourced
    question, but possible), table-aware matching should fall back to the
    standard fuzzy check rather than vacuously scoring 1.0 with nothing to
    actually verify."""
    no_numbers = "The segment names are Productivity and Intelligent Cloud."
    score = _verbatim_or_table_match_score(no_numbers, TABLE_TEXT, is_table=True)
    # Falls back to fuzzy match against the literal table text; segment
    # names do appear in TABLE_TEXT, so this should score reasonably high
    # via the fallback path, not via number-matching (there are none).
    assert score == _verbatim_match_score(no_numbers, TABLE_TEXT)


def test_table_aware_match_unaffected_for_narrative_chunks():
    """is_table=False must produce IDENTICAL behavior to the original
    _verbatim_match_score — this fix should only change behavior for
    table-derived chunks, never for narrative ones."""
    narrative_text = "Revenue increased to 50000 million driven by cloud growth this year."
    claim = "Revenue increased to 50000 million"
    assert _verbatim_or_table_match_score(claim, narrative_text, is_table=False) == \
        _verbatim_match_score(claim, narrative_text)


# ---------------------------------------------------------------------------
# Passage truncation
# ---------------------------------------------------------------------------

def test_truncate_short_text_unaffected():
    text = "A short passage well within budget."
    assert _truncate_to_char_budget(text, 1000) == text


def test_truncate_long_text_cuts_on_line_boundary():
    text = "Line one of text here.\n" * 200
    truncated = _truncate_to_char_budget(text, 100)
    assert len(truncated) < len(text)
    assert truncated.endswith("[... truncated for length ...]")
    # Should not cut mid-word — the line before the marker should be a
    # complete line from the original text or empty.
    body = truncated.rsplit("\n[... truncated", 1)[0]
    assert body == "" or text.startswith(body)


def test_truncate_falls_back_to_hard_cut_when_no_good_boundary():
    """If there's no newline within the first half of the budget, fall
    back to a hard character cut rather than failing to truncate at all."""
    text = "a" * 500  # no newlines anywhere
    truncated = _truncate_to_char_budget(text, 100)
    assert len(truncated) <= 100 + len("\n[... truncated for length ...]")


# ---------------------------------------------------------------------------
# Chunk-group sampling
# ---------------------------------------------------------------------------

def test_sample_groups_under_cap_returns_all():
    groups = [(_make_chunk(f"chunk_{i:05d}"),) for i in range(10)]
    sampled = _sample_groups(groups, max_groups=40)
    assert len(sampled) == 10


def test_sample_groups_over_cap_spans_full_range():
    groups = [(_make_chunk(f"chunk_{i:05d}"),) for i in range(500)]
    sampled = _sample_groups(groups, max_groups=40)
    assert len(sampled) == 40
    first_idx = int(sampled[0][0].chunk_id.split("_")[1])
    last_idx = int(sampled[-1][0].chunk_id.split("_")[1])
    assert first_idx < 20
    assert last_idx > 450


def test_sample_groups_includes_table_derived_chunks_when_present():
    """Regression check related to the numeric_calculation underrepresentation
    bug: when narrative chunks and table chunks are concatenated (narrative
    first, tables appended after), even sampling should still pick up some
    table-derived groups rather than only ever sampling from the narrative
    range."""
    narrative = [(_make_chunk(f"chunk_{i:05d}"),) for i in range(399)]
    tables = [(_make_chunk(f"table_{i:04d}"),) for i in range(85)]
    combined = narrative + tables
    sampled = _sample_groups(combined, max_groups=40)
    table_groups_sampled = [g for g in sampled if g[0].chunk_id.startswith("table_")]
    assert len(table_groups_sampled) > 0


# ---------------------------------------------------------------------------
# Prompt template splitting
# ---------------------------------------------------------------------------

def test_split_template_separates_system_and_user():
    template = "SYSTEM:\nYou are a helper.\n\nUSER:\nDo the thing: {x}"
    system_p, user_p = _split_template(template)
    assert system_p == "You are a helper."
    assert user_p == "Do the thing: {x}"


def test_split_template_raises_on_missing_user_marker():
    with pytest.raises(ValueError):
        _split_template("SYSTEM:\nNo user marker here at all.")
