"""
verifier.py — independent hallucination-check verification layer.

Per spec section 5:
  - Uses Hugging Face **Inference Providers** router (provider="auto" or a
    named partner with free-tier capacity), NOT the legacy hf-inference
    serverless endpoint — that path is mostly CPU-bound for small/legacy
    models and unreliable for a 27B generative model like gemma-3-27b-it.
  - Model: google/gemma-3-27b-it — a different model family from the Llama
    generator, so it isn't grading its own homework.
  - Retry-with-backoff (2-3 retries) around every HF call to absorb cold
    starts / transient 503s. This is a known limitation, called out in the
    README, not silently hidden.
  - numeric_calculation questions SKIP this verifier entirely (handled by
    numeric_check.py instead) — see verify_qa_pair() below, which dispatches
    accordingly rather than ever sending numeric questions to Gemma.
  - Also computes a context-recall embedding similarity score (answer vs.
    source_passage) via bge-small-en-v1.5, as a second independent signal.

All three signals (faithfulness/numeric_check, context_recall, and the
generation-time verbatim_match already computed in generator.py) are
returned together rather than collapsed into one pass/fail, per spec
section 5's instruction to store all three and allow later strictness
filtering without regenerating.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from huggingface_hub import InferenceClient

from config import CONFIG, get_path, require_env, setup_logging
from generator import GeneratedQA
from numeric_check import check_numeric_answer
from embeddings import cosine_similarity, embed_texts

logger = setup_logging(__name__)


@dataclass
class VerificationResult:
    faithfulness_pass: bool | None        # None for numeric_calculation (not applicable)
    numeric_check_pass: bool | None       # None for non-numeric question types
    context_recall_score: float
    verifier_confidence: str | None       # "high"/"medium"/"low" from Gemma, None if numeric
    verifier_reason: str
    keep: bool                            # final keep/discard decision


def _build_hf_client() -> InferenceClient:
    token = require_env("HF_TOKEN")
    ver_cfg = CONFIG["verification"]
    return InferenceClient(provider=ver_cfg["hf_provider"], token=token)


def _load_faithfulness_prompt() -> str:
    prompts_dir = get_path("prompts_dir")
    return (prompts_dir / "verification_faithfulness.txt").read_text(encoding="utf-8")


def _split_template(template: str) -> tuple[str, str]:
    if "USER:" not in template:
        raise ValueError("Prompt template missing 'USER:' marker")
    system_part, user_part = template.split("USER:", 1)
    system_part = system_part.replace("SYSTEM:", "", 1).strip()
    return system_part, user_part.strip()


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_verifier_response(raw_text: str) -> dict | None:
    cleaned = _strip_markdown_fences(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            logger.warning(f"Could not parse verifier response as JSON: {raw_text[:200]}")
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning(f"Could not recover JSON from verifier response: {raw_text[:200]}")
            return None


def _call_gemma_with_retry(
    client: InferenceClient, system_prompt: str, user_prompt: str
) -> str | None:
    """Calls gemma-3-27b-it via HF Inference Providers with retry-with-backoff
    to absorb cold starts / transient 503s. Returns None (not raises) if all
    retries are exhausted, so a single flaky call doesn't crash the whole
    verification pass over the dataset — callers treat None as
    "verification unavailable" and fall back to the other two signals."""
    ver_cfg = CONFIG["verification"]
    last_exc: Exception | None = None

    for attempt in range(1, ver_cfg["max_retries"] + 1):
        try:
            resp = client.chat.completions.create(
                model=ver_cfg["model_name"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=ver_cfg["temperature"],
                max_tokens=ver_cfg["max_tokens"],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_exc = e
            backoff = ver_cfg["retry_backoff_base_s"] * attempt
            logger.warning(
                f"HF Inference Providers call failed (attempt {attempt}/{ver_cfg['max_retries']}): "
                f"{e}; retrying in {backoff}s. NOTE: cold starts / transient 503s on free-tier "
                f"partner providers are a known limitation (see README)."
            )
            time.sleep(backoff)

    logger.error(f"HF verification call failed after {ver_cfg['max_retries']} attempts: {last_exc}")
    return None


def _faithfulness_check(
    client: InferenceClient, source_passage: str, question: str, answer: str
) -> tuple[bool | None, str | None, str]:
    """Returns (matches_claimed_answer, confidence, reason). All None/empty-
    reason if the HF call failed outright after retries."""
    template = _load_faithfulness_prompt()
    system_prompt, user_template = _split_template(template)
    user_prompt = user_template.format(
        source_passage=source_passage, question=question, answer=answer
    )

    raw = _call_gemma_with_retry(client, system_prompt, user_prompt)
    if raw is None:
        return None, None, "HF Inference Providers call failed after all retries."

    parsed = _parse_verifier_response(raw)
    if parsed is None:
        return None, None, "Could not parse verifier JSON response."

    matches = parsed.get("matches_claimed_answer")
    if isinstance(matches, str):
        matches = matches.strip().lower() == "true"

    return (
        bool(matches) if matches is not None else None,
        parsed.get("confidence"),
        parsed.get("reason", ""),
    )


def _context_recall_score(answer: str, source_passage: str) -> float:
    """bge-small-en-v1.5 cosine similarity between the answer and its
    source_passage — an independent embedding-based signal alongside the
    LLM faithfulness check, per spec section 3(b)."""
    vecs = embed_texts([answer, source_passage])
    return cosine_similarity(vecs[0], vecs[1])


def verify_qa_pair(client: InferenceClient, qa: GeneratedQA) -> VerificationResult:
    """
    Dispatches verification based on question_type:
      - numeric_calculation -> numeric_check.py only, Gemma is skipped entirely.
      - all other types -> Gemma faithfulness re-derivation.
    In both branches, context_recall_score is always computed via bge-small.
    """
    ver_cfg = CONFIG["verification"]
    context_recall = _context_recall_score(qa.answer, qa.source_passage)

    if qa.question_type == "numeric_calculation":
        numeric_result = check_numeric_answer(qa.source_passage, qa.answer)
        keep = numeric_result.numeric_check_pass and context_recall >= ver_cfg["min_context_recall_score"]
        return VerificationResult(
            faithfulness_pass=None,
            numeric_check_pass=numeric_result.numeric_check_pass,
            context_recall_score=context_recall,
            verifier_confidence=None,
            verifier_reason=numeric_result.reason,
            keep=keep,
        )

    matches, confidence, reason = _faithfulness_check(
        client, qa.source_passage, qa.question, qa.answer
    )
    keep = bool(matches) and context_recall >= ver_cfg["min_context_recall_score"]
    return VerificationResult(
        faithfulness_pass=matches,
        numeric_check_pass=None,
        context_recall_score=context_recall,
        verifier_confidence=confidence,
        verifier_reason=reason or "",
        keep=keep,
    )


def verify_all(qa_pairs: list[GeneratedQA], client: InferenceClient | None = None) -> list[tuple[GeneratedQA, VerificationResult]]:
    if client is None:
        client = _build_hf_client()

    results: list[tuple[GeneratedQA, VerificationResult]] = []
    for i, qa in enumerate(qa_pairs, 1):
        logger.info(f"Verifying {i}/{len(qa_pairs)}: [{qa.question_type}] {qa.question[:60]!r}")
        result = verify_qa_pair(client, qa)
        results.append((qa, result))

    kept = sum(1 for _, r in results if r.keep)
    logger.info(f"Verification complete: {kept}/{len(results)} pairs kept")
    return results


if __name__ == "__main__":
    from chunker import load_chunks
    from generator import generate_all

    chunks = load_chunks()
    qa_pairs = generate_all(chunks)
    verified = verify_all(qa_pairs)
    for qa, result in verified[:5]:
        print(f"[{qa.question_type}] keep={result.keep} recall={result.context_recall_score:.2f} | {qa.question}")
