"""
numeric_check.py — deterministic verification for numeric_calculation
questions, per spec section 5: these questions SKIP the LLM verifier
entirely and go through this module instead.
 
Approach: extract all numbers present in source_passage via regex, then try
every relationship we deterministically know how to check (difference,
percent change, sum, ratio) between pairs of extracted numbers against the
number(s) found in the claimed `answer`. If any relationship matches within
tolerance, the check passes. This is intentionally conservative — we are not
re-deriving arbitrary arithmetic via an LLM (free, but unreliable for math);
we are confirming the claimed answer's numeric value is *reachable* from the
passage's numbers via a simple, named operation.
 
This will not catch every valid-but-complex calculation a human could do
(e.g. three-number weighted averages), so cases with no matching
relationship are reported as numeric_check_pass=False with a "reason" field
rather than silently passing — callers can decide to discard or
keep-with-flag based on the other verification signals.
"""
 
from __future__ import annotations
 
import re
from dataclasses import dataclass
 
from config import CONFIG, setup_logging
 
logger = setup_logging(__name__)
 
# Matches numbers with optional $ prefix, thousands separators, decimals,
# parens for negatives (common in financial statements), and trailing % .
NUMBER_RE = re.compile(
    r"\(?-?\$?\d[\d,]*\.?\d*\)?%?"
)
 
 
@dataclass
class NumericCheckResult:
    numeric_check_pass: bool
    extracted_numbers: list[float]
    claimed_value: float | None
    matched_relationship: str | None
    reason: str
 
 
def _clean_number_token(token: str) -> float | None:
    """Convert a regex-matched token like '(1,234.5)%' or '$45,000' into a
    float. Parenthesized numbers are treated as negative (standard financial
    notation). Returns None if the token has no actual digits."""
    is_negative = token.startswith("(") and token.endswith(")")
    is_negative = is_negative or token.startswith("-")
    cleaned = token.strip("()%$").replace(",", "").lstrip("-")
    if not cleaned or not any(ch.isdigit() for ch in cleaned):
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if is_negative else value
 
 
def extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from text, in order of appearance,
    de-duplicating exact repeats but preserving order of first occurrence."""
    tokens = NUMBER_RE.findall(text)
    numbers: list[float] = []
    for tok in tokens:
        val = _clean_number_token(tok)
        if val is None:
            continue
        numbers.append(val)
    return numbers
 
 
def _within_tolerance(a: float, b: float, tolerance_pct: float) -> bool:
    if b == 0:
        return abs(a - b) < 1e-6
    return abs(a - b) / abs(b) * 100.0 <= tolerance_pct
 
 
def _claimed_value_from_answer(answer: str) -> float | None:
    """
    Pull the PRIMARY/FINAL numeric value out of the claimed answer string.
 
    The generation prompt asks for a single computed value, but free-tier
    models sometimes write out the full equation instead, e.g.
    "281724.0 - 211915.0 = 69609.0" rather than just "69609.0". In that case
    the FIRST number is an input operand, not the claimed result — taking
    it naively would mean we never actually check the number the question
    was asking about, and could rubber-stamp a wrong calculation by
    coincidentally matching an unrelated operand instead.
 
    Strategy: if the answer contains an "=" sign, take the LAST number that
    appears AFTER the last "=" (the stated final result). Otherwise (no
    equation, just a bare value like "$50,000 million" or "11.1%"), fall
    back to the first number found, as before.
    """
    if "=" in answer:
        after_last_equals = answer.rsplit("=", 1)[1]
        nums_after = extract_numbers(after_last_equals)
        if nums_after:
            return nums_after[0]
        # Equals sign present but nothing numeric after it (e.g. answer ends
        # mid-sentence) — fall through to the whole-string search below.
 
    nums = extract_numbers(answer)
    return nums[0] if nums else None
 
 
def check_numeric_answer(source_passage: str, answer: str) -> NumericCheckResult:
    """
    Main entry point: given the source_passage and the generator's claimed
    answer for a numeric_calculation question, deterministically check
    whether the claimed value is reachable from the passage's numbers via
    a simple named operation (difference, abs difference, percent change,
    sum, ratio, product).
    """
    tolerance = CONFIG["verification"]["numeric_tolerance_pct"]
 
    passage_numbers = extract_numbers(source_passage)
    claimed_value = _claimed_value_from_answer(answer)
 
    if claimed_value is None:
        return NumericCheckResult(
            numeric_check_pass=False,
            extracted_numbers=passage_numbers,
            claimed_value=None,
            matched_relationship=None,
            reason="Could not extract a numeric value from the claimed answer.",
        )
 
    if len(passage_numbers) < 2:
        # A single-number passage can still trivially "match" if the answer
        # just restates that number (e.g. a fact-like numeric question) —
        # check that case before giving up.
        if len(passage_numbers) == 1 and _within_tolerance(claimed_value, passage_numbers[0], tolerance):
            return NumericCheckResult(
                numeric_check_pass=True,
                extracted_numbers=passage_numbers,
                claimed_value=claimed_value,
                matched_relationship="direct_match",
                reason="Claimed answer matches the single number found in the passage.",
            )
        return NumericCheckResult(
            numeric_check_pass=False,
            extracted_numbers=passage_numbers,
            claimed_value=claimed_value,
            matched_relationship=None,
            reason=f"Fewer than 2 numbers extracted from passage ({len(passage_numbers)}); cannot verify a calculation.",
        )
 
    # Try every unordered pair of extracted numbers against every known
    # relationship; stop at first match within tolerance.
    for i in range(len(passage_numbers)):
        for j in range(len(passage_numbers)):
            if i == j:
                continue
            a, b = passage_numbers[i], passage_numbers[j]
 
            candidates = {
                "difference": a - b,
                "abs_difference": abs(a - b),
                "sum": a + b,
            }
            if b != 0:
                candidates["percent_change"] = (a - b) / abs(b) * 100.0
                candidates["ratio"] = a / b
            candidates["product"] = a * b
 
            for rel_name, computed in candidates.items():
                if _within_tolerance(claimed_value, computed, tolerance):
                    return NumericCheckResult(
                        numeric_check_pass=True,
                        extracted_numbers=passage_numbers,
                        claimed_value=claimed_value,
                        matched_relationship=rel_name,
                        reason=(
                            f"Claimed value {claimed_value} matches {rel_name} of "
                            f"{a} and {b} (computed={computed:.4f}) within {tolerance}% tolerance."
                        ),
                    )
 
    return NumericCheckResult(
        numeric_check_pass=False,
        extracted_numbers=passage_numbers,
        claimed_value=claimed_value,
        matched_relationship=None,
        reason=(
            f"Claimed value {claimed_value} did not match any simple relationship "
            f"(difference/percent_change/sum/ratio/product) between numbers extracted "
            f"from the passage: {passage_numbers}."
        ),
    )
 
 
if __name__ == "__main__":
    # Quick smoke examples
    examples = [
        ("Revenue grew from $45,000 million to $50,000 million.", "11.1%"),
        ("Revenue grew from $45,000 million to $50,000 million.", "5000"),
        ("Segment A had $120 million and Segment B had $80 million in revenue.", "40"),
        ("Operating margin was 35% compared to 30% last year.", "5 percentage points"),
    ]
    for passage, answer in examples:
        result = check_numeric_answer(passage, answer)
        print(f"Passage: {passage}\nAnswer: {answer}\n-> {result}\n")
 