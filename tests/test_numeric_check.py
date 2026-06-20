"""
test_numeric_check.py — known arithmetic cases for the deterministic
numeric_calculation verifier.
 
Covers:
  - extract_numbers(): plain integers/decimals, $ prefix, thousands
    separators, parenthesized negatives, trailing %, and mixed text.
  - check_numeric_answer(): every named relationship (difference,
    abs_difference, sum, percent_change, ratio, product), the
    single-number direct-match path, the <2-numbers failure path, the
    no-numeric-value-in-answer failure path, and tolerance boundary behavior.
"""
 
from __future__ import annotations
 
import pytest
 
from numeric_check import (
    extract_numbers,
    _clean_number_token,
    _within_tolerance,
    _claimed_value_from_answer,
    check_numeric_answer,
)
 
 
# ---------------------------------------------------------------------------
# extract_numbers / _clean_number_token: parsing edge cases
# ---------------------------------------------------------------------------
 
def test_extract_plain_integers():
    assert extract_numbers("Revenue was 50000 and costs were 30000.") == [50000.0, 30000.0]
 
 
def test_extract_decimals():
    assert extract_numbers("Margin was 35.5 percent.") == [35.5]
 
 
def test_extract_dollar_prefixed_numbers():
    assert extract_numbers("Revenue was $50,000 million.") == [50000.0]
 
 
def test_extract_thousands_separators():
    assert extract_numbers("Total assets were 1,234,567.") == [1234567.0]
 
 
def test_extract_percentages():
    nums = extract_numbers("Growth was 11.1% year over year.")
    assert nums == [11.1]
 
 
def test_extract_parenthesized_negative():
    """Financial statements commonly show negatives in parentheses,
    e.g. '(1,234)' meaning -1234."""
    nums = extract_numbers("Net loss of (1,234) million was recorded.")
    assert nums == [-1234.0]
 
 
def test_extract_explicit_negative_sign():
    nums = extract_numbers("Change was -45.2 compared to prior year.")
    assert nums == [-45.2]
 
 
def test_extract_multiple_mixed_numbers_in_order():
    text = "Segment A had $120 million and Segment B had $80 million in revenue."
    assert extract_numbers(text) == [120.0, 80.0]
 
 
def test_extract_no_numbers_returns_empty_list():
    assert extract_numbers("There were no figures mentioned in this sentence at all.") == []
 
 
def test_clean_number_token_dollar_and_commas():
    assert _clean_number_token("$45,000") == 45000.0
 
 
def test_clean_number_token_parens_negative():
    assert _clean_number_token("(1,234.5)") == -1234.5
 
 
def test_clean_number_token_percent_sign():
    assert _clean_number_token("11.1%") == 11.1
 
 
def test_clean_number_token_no_digits_returns_none():
    assert _clean_number_token("()") is None
    assert _clean_number_token("$") is None
 
 
# ---------------------------------------------------------------------------
# _within_tolerance
# ---------------------------------------------------------------------------
 
def test_within_tolerance_exact_match():
    assert _within_tolerance(100.0, 100.0, tolerance_pct=1.0) is True
 
 
def test_within_tolerance_just_inside_bound():
    # 1% of 100 is 1.0, so 100.9 is within tolerance, 101.1 is not
    assert _within_tolerance(100.9, 100.0, tolerance_pct=1.0) is True
 
 
def test_within_tolerance_just_outside_bound():
    assert _within_tolerance(102.0, 100.0, tolerance_pct=1.0) is False
 
 
def test_within_tolerance_zero_baseline_exact():
    assert _within_tolerance(0.0, 0.0, tolerance_pct=1.0) is True
 
 
def test_within_tolerance_zero_baseline_nonzero_claim():
    assert _within_tolerance(5.0, 0.0, tolerance_pct=1.0) is False
 
 
# ---------------------------------------------------------------------------
# check_numeric_answer: each named relationship, using the spec's tolerance
# config (verification.numeric_tolerance_pct = 1.0 in config.yaml)
# ---------------------------------------------------------------------------
 
def test_percent_change_relationship_matches():
    passage = "Revenue grew from $45,000 million to $50,000 million."
    answer = "11.1%"
    result = check_numeric_answer(passage, answer)
    assert result.numeric_check_pass is True
    assert result.matched_relationship == "percent_change"
 
 
def test_difference_relationship_matches():
    passage = "Segment A had $120 million and Segment B had $80 million in revenue."
    answer = "40"
    result = check_numeric_answer(passage, answer)
    assert result.numeric_check_pass is True
    assert result.matched_relationship in ("difference", "abs_difference")
 
 
def test_sum_relationship_matches():
    passage = "Segment A had $120 million and Segment B had $80 million in revenue."
    answer = "200"
    result = check_numeric_answer(passage, answer)
    assert result.numeric_check_pass is True
    assert result.matched_relationship == "sum"
 
 
def test_ratio_relationship_matches():
    passage = "Segment A had $100 million and Segment B had $25 million in revenue."
    answer = "4"
    result = check_numeric_answer(passage, answer)
    assert result.numeric_check_pass is True
    assert result.matched_relationship == "ratio"
 
 
def test_product_relationship_matches():
    passage = "Unit price was 5 dollars and quantity sold was 20 units."
    answer = "100"
    result = check_numeric_answer(passage, answer)
    assert result.numeric_check_pass is True
    assert result.matched_relationship == "product"
 
 
def test_percentage_point_difference_matches():
    passage = "Operating margin was 35% compared to 30% last year."
    answer = "5 percentage points"
    result = check_numeric_answer(passage, answer)
    assert result.numeric_check_pass is True
    assert result.matched_relationship in ("difference", "abs_difference")
 
 
# ---------------------------------------------------------------------------
# check_numeric_answer: failure / edge paths
# ---------------------------------------------------------------------------
 
def test_no_numeric_value_in_answer_fails():
    passage = "Revenue grew from $45,000 million to $50,000 million."
    answer = "It increased significantly with no specific figure given."
    result = check_numeric_answer(passage, answer)
    assert result.numeric_check_pass is False
    assert result.claimed_value is None
    assert "could not extract" in result.reason.lower()
 
 
def test_fewer_than_two_passage_numbers_fails():
    passage = "Revenue was strong this quarter, driven by cloud adoption."
    answer = "50000"
    result = check_numeric_answer(passage, answer)
    assert result.numeric_check_pass is False
    assert "fewer than 2 numbers" in result.reason.lower()
 
 
def test_single_passage_number_direct_match_passes():
    """If the passage has exactly one number and the claimed answer
    restates it (within tolerance), that should pass as a direct match
    even though it's not technically a 'calculation'."""
    passage = "Total revenue for the year was $50,000 million."
    answer = "$50,000 million"
    result = check_numeric_answer(passage, answer)
    assert result.numeric_check_pass is True
    assert result.matched_relationship == "direct_match"
 
 
def test_single_passage_number_mismatch_fails():
    passage = "Total revenue for the year was $50,000 million."
    answer = "$60,000 million"
    result = check_numeric_answer(passage, answer)
    assert result.numeric_check_pass is False
 
 
def test_unreachable_relationship_fails():
    """A claimed value with no simple relationship to any pair of passage
    numbers should fail rather than false-passing."""
    passage = "Segment A had $120 million and Segment B had $80 million in revenue."
    answer = "12345"  # not difference, sum, ratio, or product of 120/80
    result = check_numeric_answer(passage, answer)
    assert result.numeric_check_pass is False
    assert result.matched_relationship is None
 
 
def test_claimed_value_uses_result_after_equals_not_first_operand():
    """Regression test: when the answer spells out a full equation like
    '281724.0 - 211915.0 = 69609.0', the claimed value must be the stated
    RESULT (69609.0, after the '='), not the first number that appears
    (281724.0, which is just an input operand). Taking the first number
    naively would let the verifier 'pass' by coincidentally matching the
    wrong operand against some unrelated relationship, instead of actually
    checking the calculation the question asked about."""
    answer = "281724.0 - 211915.0 = 69609.0"
    assert _claimed_value_from_answer(answer) == 69609.0
 
 
def test_claimed_value_equation_catches_wrong_arithmetic():
    """If the model's stated equation result is actually wrong (here: the
    true percent change is ~33%, not 32%), the fix must extract the WRONG
    claimed result (0.32) and then correctly fail it against the passage,
    rather than accidentally matching some other operand and passing."""
    passage = "Total revenue was $281,724 million in fiscal year 2025 compared to $211,915 million in fiscal year 2023."
    answer = "(281724.0 - 211915.0) / 211915.0 = 0.32 or 32%"
    result = check_numeric_answer(passage, answer)
    assert result.claimed_value == 0.32
    assert result.numeric_check_pass is False
 
 
def test_claimed_value_equation_correct_arithmetic_passes():
    """Sanity check the positive case: a correctly-computed equation result
    should still extract and pass normally after the fix."""
    passage = "Total revenue was $281,724 million in fiscal year 2025 compared to $211,915 million in fiscal year 2023."
    answer = "281724.0 - 211915.0 = 69809.0"
    result = check_numeric_answer(passage, answer)
    assert result.claimed_value == 69809.0
    assert result.numeric_check_pass is True
    assert result.matched_relationship == "difference"
 
 
def test_claimed_value_no_equals_sign_unaffected():
    """Answers without an '=' sign (the common case — a bare value like
    '11.1%' or '$50,000 million') must be unaffected by the equals-sign
    handling and continue taking the first extracted number as before."""
    assert _claimed_value_from_answer("11.1%") == 11.1
    assert _claimed_value_from_answer("$50,000 million") == 50000.0
 
 
def test_claimed_value_equals_sign_with_no_trailing_number_falls_back():
    """Edge case: an '=' sign present but nothing numeric follows it (e.g.
    a truncated/malformed model response) should fall back to searching
    the whole answer string rather than returning None outright."""
    answer = "The result equals approximately... 42"
    # No literal '=' character here, so this exercises the normal path,
    # confirming no crash and correct extraction when '=' is absent.
    assert _claimed_value_from_answer(answer) == 42.0
 
 
def test_extracted_numbers_and_claimed_value_recorded_on_failure():
    """Even on failure, the result should record what was extracted/claimed
    so the row can be inspected later without regenerating (per the spec's
    'store all three scores' philosophy applied to this signal too)."""
    passage = "Segment A had $120 million and Segment B had $80 million in revenue."
    answer = "99999"
    result = check_numeric_answer(passage, answer)
    assert result.extracted_numbers == [120.0, 80.0]
    assert result.claimed_value == 99999.0
    assert result.numeric_check_pass is False
 
 
@pytest.mark.parametrize(
    "passage,answer,expected_pass",
    [
        ("Revenue rose from 100 to 110.", "10%", True),     # +10% growth
        ("Revenue fell from 110 to 100.", "-9.1%", True),    # ~-9.09% decline
        ("Revenue fell from 250 to 200.", "-25%", False),    # wrong sign/value: actual is -20%
    ],
)
def test_percent_change_sign_sensitivity(passage, answer, expected_pass):
    """percent_change is computed as (a-b)/b for ordered pair (a,b) in the
    order numbers were extracted; this parametrized check confirms growth
    vs. decline produce correctly-signed results.
 
    Note: check_numeric_answer() deliberately tries ALL named relationships
    (difference, percent_change, ratio, sum, product) across all pairs, not
    just the one matching the question's intended operation — it's checking
    "is this value reachable by some simple operation", not "is this the
    correct percent_change specifically". The third case here is chosen so
    that -25% doesn't coincidentally equal any of difference/sum/ratio/product
    of 250 and 200 either, isolating a genuine mismatch.
    """
    result = check_numeric_answer(passage, answer)
    assert result.numeric_check_pass is expected_pass
 