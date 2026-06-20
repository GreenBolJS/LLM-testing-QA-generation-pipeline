"""
fetch_filing.py — pulls the raw EDGAR .htm filing for a given CIK/accession
number, with local caching so repeated pipeline runs don't re-hit EDGAR.

Deliberately fetches the *raw* .htm document, never a PDF print. The
chunker (chunker.py) depends on preserved <b>/<u>/<i>/inline-style/<table>
tags for heading detection, and PDF-print versions of 10-Ks typically lose
or flatten that markup.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

from config import CONFIG, get_path, setup_logging

logger = setup_logging(__name__)


class FilingFetchError(RuntimeError):
    """Raised when the filing cannot be retrieved after all retries."""


def _cache_path(filing_cfg: dict) -> Path:
    raw_dir = get_path("raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{filing_cfg['ticker'].lower()}_{filing_cfg['accession_number']}.htm"
    return raw_dir / fname


def fetch_filing(force_refetch: bool = False) -> Path:
    """
    Fetch the configured filing's raw HTML, using the on-disk cache unless
    force_refetch=True. Returns the path to the cached .htm file.
    """
    filing_cfg = CONFIG["filing"]
    http_cfg = CONFIG["http"]

    cache_file = _cache_path(filing_cfg)

    if cache_file.exists() and not force_refetch:
        logger.info(f"Using cached filing at {cache_file}")
        return cache_file

    url = filing_cfg["url"]
    headers = {"User-Agent": http_cfg["user_agent"]}

    if "replace-me@example.com" in headers["User-Agent"]:
        logger.warning(
            "http.user_agent in config.yaml still contains the placeholder email. "
            "SEC EDGAR requires a real, descriptive User-Agent (e.g. 'CompanyName contact@domain.com'); "
            "requests with placeholder/generic UAs may be throttled or blocked."
        )

    last_exc: Exception | None = None
    for attempt in range(1, http_cfg["max_retries"] + 1):
        try:
            logger.info(f"Fetching filing from EDGAR (attempt {attempt}): {url}")
            resp = requests.get(url, headers=headers, timeout=http_cfg["request_timeout_s"])
            resp.raise_for_status()
            cache_file.write_text(resp.text, encoding="utf-8")
            logger.info(f"Fetched and cached filing to {cache_file} ({len(resp.text):,} chars)")
            return cache_file
        except requests.RequestException as e:
            last_exc = e
            backoff = http_cfg["retry_backoff_base_s"] * attempt
            logger.warning(f"Fetch failed ({e}); retrying in {backoff}s")
            time.sleep(backoff)

    raise FilingFetchError(
        f"Failed to fetch filing from {url} after {http_cfg['max_retries']} attempts: {last_exc}"
    )


def load_filing_html(force_refetch: bool = False) -> str:
    """Convenience wrapper: fetch (or load from cache) and return the raw HTML text."""
    path = fetch_filing(force_refetch=force_refetch)
    return path.read_text(encoding="utf-8")


if __name__ == "__main__":
    p = fetch_filing()
    print(f"Filing cached at: {p}")
