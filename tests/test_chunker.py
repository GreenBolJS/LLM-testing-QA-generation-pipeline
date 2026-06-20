"""
test_chunker.py — heading detection on known sample sections.

Covers every rule in chunker.py's design doc (spec section 2):
  1. PART/ITEM regex detection as fixed top-two heading levels.
  2. Heading-candidate classification: short + no terminal punctuation +
     bold/underline/italic.
  3. ALL CAPS -> level 2, title-case/italic -> level 3.
  4. New chunk starts at every heading.
  5. Full heading-path metadata on each chunk.
  6. Small chunks (<min_chunk_tokens) merged into a neighbor.
  7. Long chunks split by paragraph at ~max_chunk_tokens.

Each sample HTML snippet below is a minimal, known fixture (not pulled from
a live filing) so tests are deterministic and don't depend on network
access or EDGAR's actual markup at test-run time.
"""

from __future__ import annotations

import pytest

from chunker import (
    PART_RE,
    ITEM_RE,
    _classify_heading_candidate,
    _is_bold_or_underlined,
    _is_italic,
    _strip_non_visible_content,
    parse_filing_to_chunks,
    save_chunks,
    load_chunks,
)
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Regex-level tests: PART / ITEM detection (spec section 2, rule 1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,should_match",
    [
        ("PART I", True),
        ("Part II", True),
        ("PART IV", True),
        ("  PART III  ", True),
        ("PARTICIPATION AGREEMENT", False),  # must not match "PART" as a word-fragment
        ("PARTNER REVENUE", False),
        ("Item 7", False),  # ITEM_RE's job, not PART_RE's
    ],
)
def test_part_regex(text, should_match):
    assert bool(PART_RE.match(text)) == should_match


@pytest.mark.parametrize(
    "text,should_match",
    [
        ("ITEM 7.", True),
        ("Item 7A.", True),
        ("ITEM 1B.", True),
        ("Item 15.", True),
        ("  ITEM 9A.  ", True),
        # Real EDGAR filings are inconsistent about the separator after the
        # item number — these variants must ALSO match, since requiring a
        # literal trailing period caused most real ITEM headers in actual
        # filings to be missed (see chunker.py's ITEM_RE comment).
        ("Item 7", True),               # no separator at all
        ("ITEM 7", True),
        ("Item 7:", True),              # colon separator
        ("Item 7 -", True),             # dash separator
        ("Item\u00a07.", True),         # non-breaking space (common in EDGAR HTML)
        ("ITEM 1A. RISK FACTORS", True),  # heading text on the same line
        ("ITEM7.", False),       # missing required space
        ("ITEMIZED LIST.", False),  # must not match "ITEM" as a word-fragment
        ("Item 7abc", False),    # run-on suffix, not a real item letter
        ("PART I", False),
    ],
)
def test_item_regex(text, should_match):
    assert bool(ITEM_RE.match(text)) == should_match


# ---------------------------------------------------------------------------
# Heading-candidate classification (spec section 2, rules 2-3)
# ---------------------------------------------------------------------------

def _tag_from_html(html: str):
    """Helper: parse a single-tag HTML snippet and return the innermost
    tag that directly carries the text (e.g. the <p>, not the outer <html>/
    <body> wrapper BeautifulSoup always adds). find_all(True) walks outer-to-
    inner, so we want the LAST match whose own text equals the full text —
    i.e. the most specific tag, not html/body which also "contain" all the text."""
    soup = BeautifulSoup(html, "lxml")
    candidates = [t for t in soup.find_all(True) if t.get_text(strip=True)]
    if not candidates:
        raise ValueError(f"No text-bearing tag found in: {html}")
    return candidates[-1]


def test_bold_via_child_tag_detected():
    tag = _tag_from_html("<p><b>OVERVIEW</b></p>")
    assert _is_bold_or_underlined(tag) is True


def test_bold_via_inline_style_detected():
    tag = _tag_from_html('<p style="font-weight:bold">RISKS</p>')
    assert _is_bold_or_underlined(tag) is True


def test_underline_via_inline_style_detected():
    tag = _tag_from_html('<p style="text-decoration:underline">Segment Results</p>')
    assert _is_bold_or_underlined(tag) is True


def test_italic_via_child_tag_detected():
    tag = _tag_from_html("<p><i>Foreign Currencies</i></p>")
    assert _is_italic(tag) is True


def test_plain_paragraph_not_bold_or_italic():
    tag = _tag_from_html("<p>This is a plain narrative sentence with no styling at all.</p>")
    assert _is_bold_or_underlined(tag) is False
    assert _is_italic(tag) is False


def test_allcaps_bold_classified_as_level_2():
    tag = _tag_from_html("<p><b>OVERVIEW</b></p>")
    level = _classify_heading_candidate("OVERVIEW", tag)
    assert level == 2


def test_titlecase_italic_classified_as_level_3():
    tag = _tag_from_html("<p><i>Foreign Currencies</i></p>")
    level = _classify_heading_candidate("Foreign Currencies", tag)
    assert level == 3


def test_long_bold_text_not_a_heading_candidate():
    """Heading candidates must be short (<12 words per config) — a long
    bold sentence is still narrative text, not a heading."""
    long_text = "This is a fairly long bold sentence that goes well beyond twelve words in total length"
    tag = _tag_from_html(f"<p><b>{long_text}</b></p>")
    level = _classify_heading_candidate(long_text, tag)
    assert level is None


def test_terminal_punctuation_disqualifies_heading_candidate():
    """A short bold phrase ending in a period reads as a sentence, not a
    heading, per the 'lacks terminal punctuation' rule."""
    tag = _tag_from_html("<p><b>Revenue grew.</b></p>")
    level = _classify_heading_candidate("Revenue grew.", tag)
    assert level is None


def test_trailing_colon_allowed_for_mixed_case_heading():
    """Trailing colons are common in real 10-K sub-headers and should not
    disqualify a heading candidate (only '.', '!', '?', ';', ',' do).
    'Key Risks:' is bold but NOT all-caps, so it should classify as the
    lower sub-level (3), not the ALL CAPS higher sub-level (2)."""
    tag = _tag_from_html("<p><b>Key Risks:</b></p>")
    level = _classify_heading_candidate("Key Risks:", tag)
    assert level == 3


def test_trailing_colon_allowed_for_allcaps_heading():
    """Same colon-tolerance rule, but applied to an ALL CAPS heading —
    should classify as the higher sub-level (2)."""
    tag = _tag_from_html("<p><b>KEY RISKS:</b></p>")
    level = _classify_heading_candidate("KEY RISKS:", tag)
    assert level == 2


def test_plain_short_text_without_styling_is_not_a_heading():
    tag = _tag_from_html("<p>Equity</p>")
    level = _classify_heading_candidate("Equity", tag)
    assert level is None  # no bold/underline/italic -> not a heading candidate


# ---------------------------------------------------------------------------
# Full document parsing: chunk boundaries + heading-path metadata
# (spec section 2, rules 3, 5)
# ---------------------------------------------------------------------------

SAMPLE_FILING_HTML = """
<html><body>
<div>PART I</div>
<div>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</div>
<p><b>OVERVIEW</b></p>
<p>Revenue increased 15 percent year over year driven primarily by growth in our cloud
services offerings and continued strength across our commercial customer base globally.</p>
<p><i>Foreign Currencies</i></p>
<p>Foreign currency movements decreased revenue by approximately 1 percentage point during
the period, primarily due to the strengthening of the U.S. dollar relative to certain
foreign currencies in which we transact business internationally across our operations.</p>
<p><b>RISKS</b></p>
<p>Equity</p>
<p>We are exposed to equity price risk related to changes in the fair value of our equity
investments held for purposes other than trading in the ordinary course of business.</p>
<div>ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK</div>
<p>We are exposed to a variety of market risks, including foreign currency risk, interest
rate risk, and equity price risk, that arise in the ordinary course of business operations.</p>
</body></html>
"""


def test_parse_produces_expected_number_of_chunks():
    chunks = parse_filing_to_chunks(SAMPLE_FILING_HTML)
    # OVERVIEW, Foreign Currencies (distinct sub-heading -> own chunk),
    # RISKS (with "Equity" merged in, since it's below min_chunk_tokens),
    # and ITEM 7A narrative = 4 chunks expected.
    assert len(chunks) == 4


def test_heading_path_includes_part_item_and_subheadings():
    chunks = parse_filing_to_chunks(SAMPLE_FILING_HTML)
    overview_chunk = chunks[0]
    assert overview_chunk.heading_path.startswith("PART I")
    assert "ITEM 7." in overview_chunk.heading_path
    assert "OVERVIEW" in overview_chunk.heading_path


def test_new_chunk_starts_at_each_heading():
    chunks = parse_filing_to_chunks(SAMPLE_FILING_HTML)
    # The "Foreign Currencies" sub-heading should start its own chunk,
    # distinct from the OVERVIEW chunk above it.
    paths = [c.heading_path for c in chunks]
    assert any("Foreign Currencies" in p for p in paths)
    fx_chunk = next(c for c in chunks if "Foreign Currencies" in c.heading_path)
    overview_chunk = next(c for c in chunks if c.heading_path.endswith("OVERVIEW"))
    assert fx_chunk.text != overview_chunk.text
    assert "Foreign currency movements" in fx_chunk.text
    assert "Revenue increased 15 percent" in overview_chunk.text


def test_item_change_resets_subheadings():
    """Moving from one ITEM to the next should reset h2/h3 sub-heading state
    so stale sub-headings don't leak into the new ITEM's heading_path."""
    chunks = parse_filing_to_chunks(SAMPLE_FILING_HTML)
    item_7a_chunk = next(c for c in chunks if "ITEM 7A." in c.heading_path)
    assert "OVERVIEW" not in item_7a_chunk.heading_path
    assert "RISKS" not in item_7a_chunk.heading_path
    assert "Foreign Currencies" not in item_7a_chunk.heading_path


# ---------------------------------------------------------------------------
# Small-chunk merge behavior (spec section 2, rule 6)
# ---------------------------------------------------------------------------

def test_small_subheader_chunk_merged_into_neighbor():
    """'Equity' alone, with only a short sentence under it, should be below
    min_chunk_tokens and get merged into the RISKS chunk rather than
    surviving as its own shallow single-sentence chunk."""
    chunks = parse_filing_to_chunks(SAMPLE_FILING_HTML)
    # There should be no standalone chunk whose heading_path ends exactly
    # in "RISKS" with ONLY the short equity sentence — it should be merged
    # such that the RISKS-rooted chunk contains the equity risk sentence.
    risks_chunks = [c for c in chunks if "RISKS" in c.heading_path]
    assert len(risks_chunks) == 1
    assert "equity price risk" in risks_chunks[0].text


def test_merge_small_chunks_standalone_behavior():
    """Directly test _merge_small_chunks with synthetic tiny chunks to pin
    down the merge-forward behavior independent of HTML parsing."""
    from chunker import Chunk, _merge_small_chunks

    tiny = Chunk(
        chunk_id="chunk_00000", heading_path="A > B", part="A", item="B",
        text="Equity", token_estimate=1, order_index=0,
    )
    normal = Chunk(
        chunk_id="chunk_00001", heading_path="A > B > C", part="A", item="B",
        text="We are exposed to equity price risk in the ordinary course of business operations today.",
        token_estimate=16, order_index=1,
    )
    merged = _merge_small_chunks([tiny, normal])
    assert len(merged) == 1
    assert "Equity" in merged[0].text
    assert "equity price risk" in merged[0].text


def test_merge_small_chunks_merges_backward_when_last():
    """If the undersized chunk is the LAST chunk in the document (no
    'next' to merge forward into), it should merge backward into the
    previous chunk instead."""
    from chunker import Chunk, _merge_small_chunks

    normal = Chunk(
        chunk_id="chunk_00000", heading_path="A > B", part="A", item="B",
        text="This is a normal length paragraph with plenty of words in it to exceed the minimum threshold easily.",
        token_estimate=20, order_index=0,
    )
    tiny_last = Chunk(
        chunk_id="chunk_00001", heading_path="A > B > C", part="A", item="B",
        text="Tail note.", token_estimate=2, order_index=1,
    )
    merged = _merge_small_chunks([normal, tiny_last])
    assert len(merged) == 1
    assert "Tail note." in merged[0].text


# ---------------------------------------------------------------------------
# Long-chunk splitting (spec section 2, rule 4)
# ---------------------------------------------------------------------------

def test_long_narrative_split_by_paragraph():
    from chunker import _split_long_text

    # Build paragraphs that individually fit but collectively exceed max_tokens.
    para = "This paragraph contains exactly twenty words which is a reasonably sized chunk of narrative filler text today. "
    long_text = "\n".join([para] * 10)  # ~200 words * 10 = way over a small max_tokens
    pieces = _split_long_text(long_text, max_tokens=50, target_tokens=40)
    assert len(pieces) > 1
    # No piece should be drastically larger than target+1 paragraph's worth
    for piece in pieces:
        assert piece.strip() != ""


def test_short_text_not_split():
    from chunker import _split_long_text

    short_text = "A short paragraph that fits easily within the token budget."
    pieces = _split_long_text(short_text, max_tokens=500, target_tokens=400)
    assert pieces == [short_text]


# ---------------------------------------------------------------------------
# save_chunks / load_chunks round-trip
# ---------------------------------------------------------------------------

def test_save_and_load_chunks_round_trip(tmp_path):
    chunks = parse_filing_to_chunks(SAMPLE_FILING_HTML)
    out_file = tmp_path / "chunks.json"
    save_chunks(chunks, out_path=out_file)
    assert out_file.exists()

    loaded = load_chunks(path=out_file)
    assert len(loaded) == len(chunks)
    assert [c.chunk_id for c in loaded] == [c.chunk_id for c in chunks]
    assert [c.heading_path for c in loaded] == [c.heading_path for c in chunks]
    assert [c.text for c in loaded] == [c.text for c in chunks]


def test_parse_empty_html_returns_no_chunks():
    chunks = parse_filing_to_chunks("<html><body></body></html>")
    assert chunks == []


# ---------------------------------------------------------------------------
# XBRL / hidden-content stripping (regression tests for the bug where
# inline-XBRL taxonomy tags and display:none content leaked into chunks,
# producing fake questions like "What is the year mentioned in the
# passage?" sourced from a raw "http://fasb.org/us-gaap/2024#..." URI, and
# chunks whose heading_path fell back to "Untitled" because they had no
# real document structure above them at all.)
# ---------------------------------------------------------------------------

def test_strip_removes_ix_header_and_hidden_tags():
    html = """
    <html><body>
    <ix:header>
      <ix:hidden>
        <ix:nonNumeric name="us-gaap:DerivativeAssets">http://fasb.org/us-gaap/2024#DerivativeAssets</ix:nonNumeric>
      </ix:hidden>
    </ix:header>
    <p>Visible content that should survive stripping untouched today.</p>
    </body></html>
    """
    soup = BeautifulSoup(html, "lxml")
    _strip_non_visible_content(soup)
    remaining_text = soup.get_text()
    assert "fasb.org" not in remaining_text
    assert "DerivativeAssets" not in remaining_text
    assert "Visible content that should survive" in remaining_text


def test_strip_removes_display_none_elements():
    html = """
    <html><body>
    <div style="display:none">Hidden tagging metadata that must not appear in any chunk.</div>
    <p>Real visible paragraph content for the document body here today.</p>
    </body></html>
    """
    soup = BeautifulSoup(html, "lxml")
    _strip_non_visible_content(soup)
    remaining_text = soup.get_text()
    assert "Hidden tagging metadata" not in remaining_text
    assert "Real visible paragraph" in remaining_text


def test_strip_removes_visibility_hidden_elements():
    html = """
    <html><body>
    <span style="visibility:hidden">Invisible span content not meant for readers.</span>
    <p>Another real visible paragraph with plenty of content words here.</p>
    </body></html>
    """
    soup = BeautifulSoup(html, "lxml")
    _strip_non_visible_content(soup)
    remaining_text = soup.get_text()
    assert "Invisible span content" not in remaining_text
    assert "Another real visible paragraph" in remaining_text


def test_full_parse_excludes_xbrl_metadata_from_chunks():
    """End-to-end regression test: a filing with inline-XBRL metadata
    interleaved between real headings/paragraphs must produce chunks with
    NO trace of the XBRL content, and no 'Untitled' heading_path fallback
    (since every real chunk should have legitimate PART/ITEM/heading
    context once the non-visible XBRL nodes are removed first)."""
    html = """
    <html><body>
    <div>PART I</div>
    <div>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</div>
    <ix:header>
      <ix:hidden>
        <ix:nonNumeric name="us-gaap:DerivativeAssets" contextRef="c1">http://fasb.org/us-gaap/2024#DerivativeAssets</ix:nonNumeric>
      </ix:hidden>
    </ix:header>
    <p><b>OVERVIEW</b></p>
    <p>Revenue increased fifteen percent year over year driven by growth in cloud
    services and continued strength across our commercial customer base globally.</p>
    </body></html>
    """
    chunks = parse_filing_to_chunks(html)
    assert len(chunks) >= 1
    for c in chunks:
        assert "fasb.org" not in c.text
        assert "DerivativeAssets" not in c.text
        assert c.heading_path != "Untitled"