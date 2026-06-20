"""
table_extractor.py — pulls <table> elements from the raw 10-K HTML via
pandas.read_html(), tagged with the same heading-path metadata used by
chunker.py. Tables are kept as structured data (DataFrame + a row-major
"cells" representation), never flattened to a single text blob, since the
spec calls them the primary source for numeric_calculation and comparison
questions and flattening would destroy row/column alignment that the
generator needs.

We re-walk the document ourselves (rather than just calling
pd.read_html(html) once) so each table can be associated with the nearest
preceding heading, mirroring chunker.py's heading-tracking logic.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Tag

from config import get_path, setup_logging
from chunker import PART_RE, ITEM_RE, _classify_heading_candidate, _walk_blocks, _strip_non_visible_content

logger = setup_logging(__name__)


@dataclass
class ExtractedTable:
    table_id: str
    heading_path: str
    part: str | None
    item: str | None
    order_index: int
    n_rows: int
    n_cols: int
    columns: list[str]
    rows: list[list[str]]   # row-major cell values, all stringified
    raw_html: str            # original <table>...</table> for traceability / source_passage use


def _table_to_records(table_tag: Tag) -> pd.DataFrame | None:
    """Parse a single <table> tag with pandas.read_html. Returns None if
    pandas can't parse it (e.g. empty/decorative table) rather than raising,
    since 10-Ks contain plenty of layout-only tables we should skip quietly."""
    try:
        dfs = pd.read_html(io.StringIO(str(table_tag)))
    except (ValueError, ImportError) as e:
        logger.debug(f"Skipping unparseable table: {e}")
        return None
    if not dfs:
        return None
    df = dfs[0]
    # Drop fully-empty rows/cols that pandas sometimes produces from
    # decorative spacer cells in EDGAR's table markup.
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty or df.shape[0] < 1:
        return None
    return df


def extract_tables(html: str) -> list[ExtractedTable]:
    """
    Walk the document in order, tracking heading state the same way
    chunker.py does, and emit one ExtractedTable per <table> found —
    skipping tables pandas can't meaningfully parse (decorative/empty ones).
    """
    soup = BeautifulSoup(html, "lxml")
    _strip_non_visible_content(soup)

    current_part: str | None = None
    current_item: str | None = None
    current_h2: str | None = None
    current_h3: str | None = None

    def heading_path() -> str:
        parts = [p for p in (current_part, current_item, current_h2, current_h3) if p]
        return " > ".join(parts) if parts else "Untitled"

    tables: list[ExtractedTable] = []
    order_index = 0

    # Walk ALL relevant top-level nodes in document order: headings (from
    # block tags) AND <table> tags, interleaved, so heading state is correct
    # at the moment we encounter each table.
    all_nodes = soup.find_all(["p", "div", "span", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table"])

    seen_table_ids: set[int] = set()

    for tag in all_nodes:
        if tag.name == "table":
            if id(tag) in seen_table_ids:
                continue
            # Skip nested tables-within-tables to avoid double extraction
            if tag.find_parent("table") is not None:
                continue
            seen_table_ids.add(id(tag))

            df = _table_to_records(tag)
            if df is None:
                continue

            columns = [str(c) for c in df.columns]
            # IMPORTANT: df.astype(str) does NOT convert NaN cells to the
            # string "nan" in all pandas versions/dtypes — NaN can survive
            # as a literal float even in a string-dtype column, since pandas
            # treats it as missing data exempt from the cast. Financial
            # tables commonly have blank/merged cells (spacer columns,
            # missing year data), so we explicitly fillna BEFORE casting to
            # guarantee every cell is a real str before " | ".join() runs
            # on it downstream in table_to_passage_text().
            rows = df.fillna("").astype(str).values.tolist()

            tables.append(
                ExtractedTable(
                    table_id=f"table_{order_index:04d}",
                    heading_path=heading_path(),
                    part=current_part,
                    item=current_item,
                    order_index=order_index,
                    n_rows=len(rows),
                    n_cols=len(columns),
                    columns=columns,
                    rows=rows,
                    raw_html=str(tag),
                )
            )
            order_index += 1
            continue

        # Skip nodes that live inside a table — already covered when we hit
        # the table itself, and walking them again would corrupt heading state.
        if tag.find_parent("table") is not None:
            continue

        text = tag.get_text(separator=" ", strip=True)
        if not text:
            continue

        if PART_RE.match(text):
            current_part = text.strip()
            current_item = None
            current_h2 = None
            current_h3 = None
            continue
        if ITEM_RE.match(text):
            current_item = text.strip()
            current_h2 = None
            current_h3 = None
            continue

        level = _classify_heading_candidate(text, tag)
        if level == 2:
            current_h2 = text.strip()
            current_h3 = None
        elif level == 3:
            current_h3 = text.strip()

    logger.info(f"Extracted {len(tables)} usable tables from filing")
    return tables


def table_to_passage_text(table: ExtractedTable) -> str:
    """
    Render a table as a compact pipe-delimited text block for use as a
    source_passage / generation context — preserves row/column structure
    (unlike a flattened prose summary) while still being plain text the
    Groq/HF chat APIs can consume.

    Defensively str()-coerces every cell here too (on top of the fillna+
    astype(str) done at extraction time in extract_tables) so a single
    malformed/NaN cell degrades to an empty string instead of crashing the
    whole pipeline run with a TypeError deep into a multi-table batch.
    """
    def _cell_str(v) -> str:
        if v is None:
            return ""
        s = str(v)
        return "" if s.lower() == "nan" else s

    lines = [" | ".join(_cell_str(c) for c in table.columns)]
    for row in table.rows:
        lines.append(" | ".join(_cell_str(c) for c in row))
    return "\n".join(lines)


def save_tables(tables: list[ExtractedTable], out_path: Path | None = None) -> Path:
    if out_path is None:
        chunks_dir = get_path("chunks_dir")
        chunks_dir.mkdir(parents=True, exist_ok=True)
        out_path = chunks_dir / "tables.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in tables], f, indent=2)
    logger.info(f"Saved {len(tables)} tables to {out_path}")
    return out_path


def load_tables(path: Path | None = None) -> list[ExtractedTable]:
    if path is None:
        path = get_path("chunks_dir") / "tables.json"
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [ExtractedTable(**t) for t in raw]


if __name__ == "__main__":
    from fetch_filing import load_filing_html

    html = load_filing_html()
    tables = extract_tables(html)
    save_tables(tables)
    print(f"Extracted {len(tables)} tables.")
    for t in tables[:3]:
        print(f"  [{t.table_id}] {t.n_rows}x{t.n_cols} | {t.heading_path}")