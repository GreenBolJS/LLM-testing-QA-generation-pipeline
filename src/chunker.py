"""
chunker.py — heading-hierarchy based chunking of a raw EDGAR 10-K .htm file.

Design (per spec section 2):
  1. "PART [IVX]+" and "ITEM \\d+[A-Z]?." are regex-matched as fixed top-two
     heading levels — they're always present and consistently formatted
     across 10-Ks, so we don't rely on style detection for them.
  2. All other block-level elements are heading *candidates* if: short text
     (<12 words), no terminal punctuation, and bold/underlined (tag or
     inline style). ALL CAPS -> higher sub-level; title-case/italic ->
     lower sub-level.
  3. A new chunk starts at every heading (any level) and ends at the next
     heading.
  4. Narrative chunks are capped at ~500-600 tokens (config-driven), split
     by paragraph if longer.
  5. Every chunk gets a full heading-path string in its metadata, e.g.
     "Item 7A > RISKS > Foreign Currencies".
  6. Chunks under ~30 tokens get merged into the neighboring chunk.
  7. <table> elements are NOT handled here — table_extractor.py pulls them
     separately via pandas.read_html(), tagged with the same heading-path
     metadata so generation can treat narrative and tabular content
     consistently.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from config import CONFIG, get_path, setup_logging

logger = setup_logging(__name__)

# EDGAR iXBRL filings declare an "xmlns:ix" namespace at the document root,
# which makes bs4/lxml suspect the document might actually be XML and emit
# XMLParsedAsHTMLWarning. It isn't — the visible structure (p/div/table/etc.)
# is genuine HTML5; the ix: tags are just XBRL data interleaved into it. We
# deliberately keep the HTML parser (not "lxml-xml") because the heading/
# block-walking logic below depends on HTML-mode tag semantics. Suppressing
# this specific, expected warning rather than switching parsers.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

PART_RE = re.compile(r"^\s*PART\s+[IVXLC]+\b", re.IGNORECASE)
# Matches "ITEM 7.", "Item 7", "ITEM 7:", "Item 7 -", "Item 1A." etc.
# Real EDGAR filings are inconsistent about the separator after the item
# number — some use a period, some a colon or dash, some nothing at all if
# the heading text follows directly on the same line/tag. The original
# regex required a literal trailing period, which caused most real ITEM
# headers to fall through to the generic heading-candidate heuristic
# instead, silently losing the item-level heading-path/metadata for the
# majority of chunks (see: source_item populated in only ~6% of rows).
# \b after the number ensures we don't match "ITEM 7A" as wrongly stopping
# at "7" before the letter suffix; the optional separator group is
# non-capturing and tolerant of period/colon/dash/nothing.
ITEM_RE = re.compile(r"^\s*ITEM\s+\d+[A-Z]?(?:\.|:|\s+-|\b)", re.IGNORECASE)

# Block-level tags we walk looking for headings / paragraph text.
# <table> is deliberately excluded — table_extractor.py owns tables.
BLOCK_TAGS = {"p", "div", "span", "li", "h1", "h2", "h3", "h4", "h5", "h6"}


@dataclass
class Chunk:
    chunk_id: str
    heading_path: str          # e.g. "Item 7A > RISKS > Foreign Currencies"
    part: str | None
    item: str | None
    text: str
    token_estimate: int
    order_index: int           # position in document, for stable sort / multi-chunk pairing


@dataclass
class HeadingLevel:
    text: str
    level: int                 # 0 = PART, 1 = ITEM, 2 = ALLCAPS sub-heading, 3 = title-case/italic sub-heading


def _approx_tokens(text: str) -> int:
    words = len(text.split())
    return int(words * CONFIG["chunking"]["approx_tokens_per_word"])


def _is_bold_or_underlined(tag: Tag) -> bool:
    if tag.find(["b", "strong", "u"]) is not None:
        return True
    style = (tag.get("style") or "").lower()
    if "font-weight:bold" in style.replace(" ", "") or "font-weight:700" in style.replace(" ", ""):
        return True
    if "text-decoration:underline" in style.replace(" ", ""):
        return True
    # Tag itself wrapped in b/strong/u (not just containing one)
    if tag.name in ("b", "strong", "u"):
        return True
    return False


def _is_italic(tag: Tag) -> bool:
    if tag.find(["i", "em"]) is not None:
        return True
    style = (tag.get("style") or "").lower()
    if "font-style:italic" in style.replace(" ", ""):
        return True
    if tag.name in ("i", "em"):
        return True
    return False


def _classify_heading_candidate(text: str, tag: Tag) -> int | None:
    """
    Returns a heading level (2=ALLCAPS sub-heading, 3=title-case/italic
    sub-heading) if `text`/`tag` looks like a heading candidate per the
    spec's rule (short, no terminal punctuation, bold/underlined). Returns
    None if it doesn't qualify, i.e. it's ordinary paragraph text.
    """
    stripped = text.strip()
    if not stripped:
        return None

    word_count = len(stripped.split())
    if word_count >= CONFIG["chunking"]["min_heading_words"]:
        return None

    # No terminal punctuation (allow trailing colon, which is common in headers)
    if stripped[-1] in ".!?;,":
        return None

    bold_ul = _is_bold_or_underlined(tag)
    italic = _is_italic(tag)

    if not (bold_ul or italic):
        return None

    # ALL CAPS check: ignore non-alpha chars (numbers, punctuation, &) when deciding
    letters = [c for c in stripped if c.isalpha()]
    is_all_caps = bool(letters) and all(c.isupper() for c in letters)

    if is_all_caps and bold_ul:
        return 2  # higher sub-level, e.g. "OVERVIEW", "RISKS"
    if italic or bold_ul:
        return 3  # lower sub-level, e.g. "Foreign Currencies"
    return None


def _split_long_text(text: str, max_tokens: int, target_tokens: int) -> list[str]:
    """Split narrative text exceeding max_tokens into paragraph-bounded pieces
    targeting ~target_tokens each. Never splits mid-paragraph."""
    if _approx_tokens(text) <= max_tokens:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return [text]

    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _approx_tokens(para)
        if current and current_tokens + para_tokens > target_tokens:
            pieces.append("\n".join(current))
            current = [para]
            current_tokens = para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        pieces.append("\n".join(current))

    return pieces if pieces else [text]


def _walk_blocks(soup: BeautifulSoup):
    """
    Yield (tag, text) for top-level-ish block elements in document order,
    skipping nested duplicates (e.g. a <p> inside a <div> we already yielded
    text for) and skipping <table> entirely (handled by table_extractor.py).
    """
    seen_ids: set[int] = set()

    for tag in soup.find_all(BLOCK_TAGS):
        if tag.find_parent("table") is not None:
            continue  # tables handled separately
        # Skip if an ancestor of this tag is already one we'll yield with the same text
        # (avoids double-counting e.g. <div><p>text</p></div>)
        text = tag.get_text(separator=" ", strip=True)
        if not text:
            continue
        if id(tag) in seen_ids:
            continue

        # If a direct block-tag descendant carries the same full text, skip the parent
        # to prefer the more specific tag's styling info.
        descendant_block = tag.find(BLOCK_TAGS)
        if descendant_block is not None:
            descendant_text = descendant_block.get_text(separator=" ", strip=True)
            if descendant_text == text:
                continue

        seen_ids.add(id(tag))
        yield tag, text


def _safe_decompose(tag: Tag) -> None:
    """
    Decompose a tag, but skip it if it's already been torn down.

    decompose() recursively wipes the internal state (incl. `.attrs`) of
    EVERY descendant of the tag it's called on, not just the tag itself.
    All three passes below collect their matches eagerly (soup.find_all
    returns a full list before the loop body runs anything), so a single
    pass can easily contain both an outer element AND one of its own
    descendants as separate list entries (e.g. an outer
    <div style="display:none"> wrapping an inner <span style="...">).
    Decomposing the outer entry first silently nulls the inner entry's
    `.attrs` to None; if we then call `.get()` on that already-gutted
    descendant later in the same loop, bs4 raises
    "'NoneType' object has no attribute 'get'" — which is exactly the
    crash this guard prevents, in all three passes, not just the one that
    happened to trip it first.
    """
    if getattr(tag, "decomposed", False):
        return
    tag.decompose()


def _strip_non_visible_content(soup: BeautifulSoup) -> None:
    """
    Removes inline-XBRL tagging and other non-visible/hidden nodes from the
    parsed document IN PLACE, before any heading/chunk walking begins.

    Modern EDGAR filings use Inline XBRL (iXBRL): the visible HTML is
    interleaved with machine-readable tags like <ix:header>, <ix:hidden>,
    <ix:nonNumeric>, <ix:nonFraction> that carry structured financial data
    (e.g. raw taxonomy URIs like "http://fasb.org/us-gaap/2024#..."). Some
    of these wrap visible numbers (legitimate, should stay), but <ix:header>
    and <ix:hidden> specifically are NEVER rendered to a human reader —
    they're pure machine metadata. Without stripping these, the chunker can
    produce "chunks" made entirely of taxonomy URIs and tag soup with no
    real heading context, which both pollutes generation (the LLM invents a
    trivial question from a URI) and explains chunks whose heading_path
    falls back to "Untitled" (no real PART/ITEM/heading ever preceded them,
    because they aren't part of the visible document flow at all).

    We also strip any element whose inline style hides it from a reader
    (display:none or visibility:hidden) — a common iXBRL pattern wraps
    hidden data in a styled <span> or <div> rather than an <ix:hidden> tag.

    Every decompose call below goes through _safe_decompose, and every
    direct attribute read goes through a `tag.attrs` truthiness check
    first — see _safe_decompose's docstring for why both guards are needed.
    """
    # Inline-XBRL namespace tags that are never rendered to a human reader.
    # (ix:nonFraction / ix:nonNumeric are deliberately NOT included here —
    # those wrap VISIBLE numbers/text the filing actually displays; only
    # header/hidden are pure non-visible metadata.)
    for tag in soup.find_all(["ix:header", "ix:hidden"]):
        _safe_decompose(tag)

    # Some parsers normalize the "ix:" namespace prefix differently
    # depending on how the document declares its namespaces — fall back to
    # a name-based match in case the colon prefix didn't survive parsing.
    # IMPORTANT: only match tags whose name actually starts with "ix" (e.g.
    # "ix:header", "ixheader") or whose class list contains an "ix"-prefixed
    # token — a bare "header"/"hidden" tag name with no XBRL signal at all
    # (e.g. a plain HTML5 <header> landmark) must NOT match here, or we'd
    # delete legitimate visible content.
    def _is_xbrl_header_or_hidden(t) -> bool:
        if getattr(t, "decomposed", False):
            return False
        if not t.name:
            return False
        local_name = t.name.split(":")[-1]
        if local_name not in ("header", "hidden"):
            return False
        if t.name.startswith("ix"):
            return True
        classes = t.attrs.get("class") if t.attrs else None
        if not classes:
            return False
        if not isinstance(classes, list):
            classes = [classes]
        return any("ix" in str(c) for c in classes)

    for tag in soup.find_all(_is_xbrl_header_or_hidden):
        _safe_decompose(tag)

    # display:none / visibility:hidden inline styles — common pattern for
    # hiding iXBRL-tagged values from visual rendering while keeping them
    # machine-readable.
    #
    # This is the pass that actually crashed: a style-bearing element can be
    # a descendant of another style-bearing element decomposed earlier in
    # this same eagerly-collected list (see _safe_decompose's docstring).
    # We check both `decomposed` and `tag.attrs` truthiness before touching
    # `.get()` at all — `tag.get("style", "")` is NOT safe here even with a
    # default, because the crash is `self.attrs` itself being None, not the
    # missing-key case the default argument is meant to handle.
    for tag in soup.find_all(style=True):
        if getattr(tag, "decomposed", False) or not tag.attrs:
            continue
        style = (tag.attrs.get("style") or "").lower().replace(" ", "")
        if "display:none" in style or "visibility:hidden" in style:
            _safe_decompose(tag)


def parse_filing_to_chunks(html: str) -> list[Chunk]:
    """
    Main entry point: parse raw 10-K HTML into a flat, ordered list of Chunk
    objects, each carrying a full heading-path string.
    """
    soup = BeautifulSoup(html, "lxml")
    _strip_non_visible_content(soup)

    chunks: list[Chunk] = []
    order_index = 0

    current_part: str | None = None
    current_item: str | None = None
    current_h2: str | None = None   # ALLCAPS sub-heading
    current_h3: str | None = None   # title-case/italic sub-heading

    buffer_text: list[str] = []

    def heading_path() -> str:
        parts = [p for p in (current_part, current_item, current_h2, current_h3) if p]
        return " > ".join(parts) if parts else "Untitled"

    def flush_buffer():
        nonlocal order_index, buffer_text
        text = "\n".join(buffer_text).strip()
        buffer_text = []
        if not text:
            return
        for piece in _split_long_text(
            text,
            CONFIG["chunking"]["max_chunk_tokens"],
            CONFIG["chunking"]["target_chunk_tokens"],
        ):
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{order_index:05d}",
                    heading_path=heading_path(),
                    part=current_part,
                    item=current_item,
                    text=piece,
                    token_estimate=_approx_tokens(piece),
                    order_index=order_index,
                )
            )
            order_index += 1

    for tag, text in _walk_blocks(soup):
        if PART_RE.match(text):
            flush_buffer()
            current_part = text.strip()
            current_item = None
            current_h2 = None
            current_h3 = None
            continue

        if ITEM_RE.match(text):
            flush_buffer()
            current_item = text.strip()
            current_h2 = None
            current_h3 = None
            continue

        level = _classify_heading_candidate(text, tag)
        if level == 2:
            flush_buffer()
            current_h2 = text.strip()
            current_h3 = None
            continue
        if level == 3:
            flush_buffer()
            current_h3 = text.strip()
            continue

        # Ordinary paragraph text — accumulate into the current chunk buffer.
        buffer_text.append(text)

    flush_buffer()  # flush trailing content

    merged = _merge_small_chunks(chunks)
    logger.info(f"Parsed {len(merged)} chunks from filing ({len(chunks)} before small-chunk merge)")
    return merged


def _merge_small_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """
    Merge any chunk under min_chunk_tokens into its neighboring chunk
    (prefer merging forward into the next chunk, falling back to merging
    backward into the previous one if it's the last chunk in the list).
    This prevents shallow single-sentence fact questions from trivial
    sub-headers like "Equity".
    """
    min_tokens = CONFIG["chunking"]["min_chunk_tokens"]
    if not chunks:
        return chunks

    merged: list[Chunk] = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        if chunk.token_estimate < min_tokens and i + 1 < len(chunks):
            nxt = chunks[i + 1]
            combined_text = chunk.text + "\n" + nxt.text
            merged_chunk = Chunk(
                chunk_id=nxt.chunk_id,
                # Keep the more specific (later) heading path — the small
                # chunk was likely a shallow sub-header anyway.
                heading_path=nxt.heading_path,
                part=nxt.part,
                item=nxt.item,
                text=combined_text,
                token_estimate=_approx_tokens(combined_text),
                order_index=chunk.order_index,
            )
            chunks[i + 1] = merged_chunk
            i += 1
            continue
        elif chunk.token_estimate < min_tokens and merged:
            prev = merged[-1]
            combined_text = prev.text + "\n" + chunk.text
            merged[-1] = Chunk(
                chunk_id=prev.chunk_id,
                heading_path=prev.heading_path,
                part=prev.part,
                item=prev.item,
                text=combined_text,
                token_estimate=_approx_tokens(combined_text),
                order_index=prev.order_index,
            )
            i += 1
            continue
        else:
            merged.append(chunk)
            i += 1

    return merged


def save_chunks(chunks: list[Chunk], out_path: Path | None = None) -> Path:
    """Persist chunks as intermediate JSON for debugging/inspection."""
    if out_path is None:
        chunks_dir = get_path("chunks_dir")
        chunks_dir.mkdir(parents=True, exist_ok=True)
        out_path = chunks_dir / "chunks.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2)

    logger.info(f"Saved {len(chunks)} chunks to {out_path}")
    return out_path


def load_chunks(path: Path | None = None) -> list[Chunk]:
    if path is None:
        path = get_path("chunks_dir") / "chunks.json"
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Chunk(**c) for c in raw]


if __name__ == "__main__":
    from fetch_filing import load_filing_html

    html = load_filing_html()
    chunks = parse_filing_to_chunks(html)
    save_chunks(chunks)
    print(f"Produced {len(chunks)} chunks.")
    for c in chunks[:5]:
        print(f"  [{c.chunk_id}] ({c.token_estimate} tok) {c.heading_path}")