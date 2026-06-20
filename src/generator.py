"""
generator.py — generates QA pairs from chunks via Groq's llama-3.1-8b-instant,
one call per (question_type, chunk-or-chunk-pair).

Per spec section 4:
  - comparison and multi_step_reasoning prompts occasionally receive TWO
    related chunks (same section, adjacent fiscal years) rather than always
    a single chunk.
  - Output is strict JSON: question, answer, source_passage (verbatim),
    question_type.
  - After generation, fuzzy-match source_passage against the original chunk
    text; if similarity < verbatim_match_threshold, discard or regenerate.
    This is an early hallucination signal (catches paraphrasing-instead-of-
    quoting) before the verification stage even runs.
"""

from __future__ import annotations

import json
import random
import re
import uuid
from dataclasses import dataclass, field

from openai import OpenAI
from rapidfuzz import fuzz

from config import CONFIG, get_path, require_env, setup_logging
from chunker import Chunk

logger = setup_logging(__name__)

PROMPT_FILES = {
    "fact_extraction": "generation_fact.txt",
    "numeric_calculation": "generation_numeric.txt",
    "comparison": "generation_comparison.txt",
    "multi_step_reasoning": "generation_multistep.txt",
}


@dataclass
class GeneratedQA:
    id: str
    question: str
    answer: str
    source_passage: str
    question_type: str
    heading_path: str
    part: str | None
    item: str | None
    source_chunk_ids: list[str]
    verbatim_match_score: float


def _load_prompt_template(question_type: str) -> str:
    prompts_dir = get_path("prompts_dir")
    fname = PROMPT_FILES[question_type]
    path = prompts_dir / fname
    return path.read_text(encoding="utf-8")


def _build_client() -> OpenAI:
    """Groq exposes an OpenAI-compatible API, so we reuse the openai client
    pointed at Groq's base URL rather than writing a bespoke HTTP client."""
    api_key = require_env("GROQ_API_KEY")
    gen_cfg = CONFIG["generation"]
    return OpenAI(api_key=api_key, base_url=gen_cfg["api_base"])


def _strip_markdown_fences(text: str) -> str:
    """Defensive cleanup: even though the prompt says 'no markdown fences',
    instruction-following on a small free-tier model isn't perfect, so we
    strip ```json ... ``` wrappers if the model adds them anyway."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_generation_response(raw_text: str) -> list[dict]:
    cleaned = _strip_markdown_fences(raw_text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Last-resort recovery: grab the first [...] block in case the model
        # added stray preamble text despite instructions.
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            logger.warning(f"Could not parse generation response as JSON: {raw_text[:200]}")
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning(f"Could not recover JSON from generation response: {raw_text[:200]}")
            return []

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    return parsed


def _coerce_str(value) -> str:
    """
    JSON-mode generation isn't guaranteed to return strings for fields we
    treat as text. The clearest example: a numeric_calculation `answer`
    like 281724 or 0.32 frequently comes back as a bare JSON number rather
    than a quoted string, which crashes a direct `.strip()` call (the
    previous `(cand.get("answer") or "").strip()` pattern only guards
    against None/empty-string, not against the value being truthy but
    non-string). The same exposure exists for `question`, `source_passage`,
    and `question_type` — e.g. a model could hand back source_passage as a
    list if it ever bundles multiple spans — so every field pulled out of a
    generation candidate goes through this coercion, not just the one that
    happened to trip first.

    If the model returns a list (e.g. multiple passage spans), join them
    with a space rather than stringifying the Python list repr, so the
    result still reads like a real passage instead of "['a', 'b']".
    """
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_coerce_str(v) for v in value).strip()
    return str(value).strip()


def _call_groq(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    gen_cfg = CONFIG["generation"]
    resp = client.chat.completions.create(
        model=gen_cfg["model_name"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=gen_cfg["temperature"],
        max_tokens=gen_cfg["max_tokens"],
    )
    return resp.choices[0].message.content or ""


def _split_template(template: str) -> tuple[str, str]:
    """Templates are authored as 'SYSTEM:\\n...\\n\\nUSER:\\n...' — split into
    the two halves for the chat API."""
    if "USER:" not in template:
        raise ValueError("Prompt template missing 'USER:' marker")
    system_part, user_part = template.split("USER:", 1)
    system_part = system_part.replace("SYSTEM:", "", 1).strip()
    return system_part, user_part.strip()


def _verbatim_match_score(source_passage: str, chunk_text: str) -> float:
    """
    Fuzzy-match the LLM's claimed source_passage against the original chunk
    text it was generated from. rapidfuzz partial_ratio is used (rather than
    plain ratio) because source_passage is expected to be a SPAN within
    chunk_text, not the whole chunk — plain ratio would unfairly penalize
    short verbatim quotes pulled from a long chunk.
    """
    if not source_passage.strip():
        return 0.0
    score = fuzz.partial_ratio(source_passage, chunk_text)
    return score / 100.0


def _maybe_pair_chunks(
    chunks: list[Chunk], question_type: str
) -> list[tuple[Chunk, ...]]:
    """
    Build the list of chunk-groups to generate against for this question_type.
    fact_extraction/numeric_calculation always use single chunks (per spec,
    these question types operate on what's directly stated in one passage).
    comparison/multi_step_reasoning occasionally pair two chunks from the
    same heading_path (proxy for "same section, adjacent fiscal years") with
    probability multi_chunk_probability.
    """
    gen_cfg = CONFIG["generation"]
    if question_type not in gen_cfg["multi_chunk_question_types"]:
        groups = [(c,) for c in chunks]
    else:
        groups = []
        used_as_second: set[str] = set()

        # Group chunks by their top-level heading (Part>Item) as a proxy for
        # "same section" pairing eligibility.
        by_section: dict[str, list[Chunk]] = {}
        for c in chunks:
            key = f"{c.part}|{c.item}"
            by_section.setdefault(key, []).append(c)

        for c in chunks:
            key = f"{c.part}|{c.item}"
            siblings = [s for s in by_section.get(key, []) if s.chunk_id != c.chunk_id]
            if siblings and random.random() < gen_cfg["multi_chunk_probability"]:
                partner = random.choice(siblings)
                if partner.chunk_id not in used_as_second:
                    groups.append((c, partner))
                    used_as_second.add(partner.chunk_id)
                    continue
            groups.append((c,))

    return _sample_groups(groups, gen_cfg["max_chunk_groups_per_type"])


def _sample_groups(
    groups: list[tuple[Chunk, ...]], max_groups: int
) -> list[tuple[Chunk, ...]]:
    """
    Caps the number of chunk-groups generation will run against for a given
    question_type, per config.yaml's generation.max_chunk_groups_per_type.
    The spec's deliverable only needs >=100 final verified rows — running
    every question type against every chunk in a 300-500+ chunk filing
    produces thousands of unneeded candidates and burns through free-tier
    API quota far faster than necessary.

    Samples EVENLY across document order (every Nth group) rather than
    randomly, so the kept sample still spans the whole filing (early items,
    MD&A, risk factors, financial statements, etc.) instead of randomly
    clustering in one section by chance.
    """
    if len(groups) <= max_groups:
        return groups

    step = len(groups) / max_groups
    indices = [int(i * step) for i in range(max_groups)]
    return [groups[i] for i in indices]


def _truncate_to_char_budget(text: str, max_chars: int) -> str:
    """
    Truncates text to at most max_chars, cutting on the last newline before
    the limit (never mid-word/mid-sentence) so a giant table or chunk still
    sends complete rows/lines to the LLM rather than a garbled partial line.
    Appends a marker so the model (and anyone auditing output later) knows
    the passage was cut, rather than silently looking complete.
    """
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n", 0, max_chars)
    if cut == -1 or cut < max_chars * 0.5:
        # No good line boundary found (or it would cut away more than half
        # the budget) — fall back to a hard character cut.
        cut = max_chars
    return text[:cut] + "\n[... truncated for length ...]"


def _combined_passage_text(chunk_group: tuple[Chunk, ...]) -> str:
    gen_cfg = CONFIG["generation"]
    max_chars = gen_cfg["max_passage_chars"]

    if len(chunk_group) == 1:
        text = chunk_group[0].text
    else:
        text = f"{chunk_group[0].text}\n--- NEXT PASSAGE ---\n{chunk_group[1].text}"

    return _truncate_to_char_budget(text, max_chars)


def _is_table_derived(chunk: Chunk) -> bool:
    """Table-derived pseudo-chunks are tagged with chunk_id='table_NNNN' by
    pipeline.py's _tables_as_pseudo_chunks(). Used to apply a different,
    table-appropriate verbatim check (see _verbatim_or_table_match_score)."""
    return chunk.chunk_id.startswith("table_")


def _numbers_in_text(text: str) -> set[str]:
    """Extracts a set of numeric substrings from text, normalized (commas/
    currency symbols stripped) for comparing whether numbers mentioned in a
    claimed source_passage actually appear in the original table text."""
    import re as _re
    raw = _re.findall(r"-?\d[\d,]*\.?\d*", text)
    return {tok.replace(",", "") for tok in raw if any(c.isdigit() for c in tok)}


def _verbatim_or_table_match_score(
    source_passage: str, passage_text: str, is_table: bool
) -> float:
    """
    For narrative chunks: unchanged — fuzzy partial_ratio of source_passage
    against the original chunk text (see _verbatim_match_score).

    For table-derived chunks: a literal pipe-delimited quote is an unnatural
    thing to ask an LLM to reproduce verbatim — models naturally convert
    "Productivity | 50000 | 45000" into prose like "Productivity segment
    revenue was 50000". Requiring a high fuzzy-match on the literal table
    string was silently dropping most table-sourced candidates (this is why
    numeric_calculation/comparison questions were severely underrepresented
    despite tables being explicitly the primary source for them per spec).

    Instead, for tables we check that every number mentioned in the claimed
    source_passage actually appears somewhere in the original table text.
    This still catches real hallucination (a fabricated number that isn't
    in the table at all) while not penalizing the LLM for paraphrasing the
    surrounding words/labels.
    """
    if not is_table:
        return _verbatim_match_score(source_passage, passage_text)

    claimed_numbers = _numbers_in_text(source_passage)
    if not claimed_numbers:
        # No numbers claimed at all from a table passage — fall back to the
        # standard fuzzy check rather than vacuously passing.
        return _verbatim_match_score(source_passage, passage_text)

    table_numbers = _numbers_in_text(passage_text)
    matched = claimed_numbers & table_numbers
    return len(matched) / len(claimed_numbers)


def generate_for_chunk_group(
    client: OpenAI, chunk_group: tuple[Chunk, ...], question_type: str
) -> list[GeneratedQA]:
    """Run one generation call for a single chunk-or-chunk-pair + question_type,
    returning verbatim-checked GeneratedQA objects (failures are dropped, not
    raised, so one bad chunk doesn't kill the whole pipeline run)."""
    gen_cfg = CONFIG["generation"]
    template = _load_prompt_template(question_type)
    system_prompt, user_template = _split_template(template)

    passage_text = _combined_passage_text(chunk_group)
    primary = chunk_group[0]
    is_table = _is_table_derived(primary)

    user_prompt = user_template.format(
        heading_path=primary.heading_path,
        chunk_text=passage_text,
        n=gen_cfg["questions_per_call"],
    )

    try:
        raw = _call_groq(client, system_prompt, user_prompt)
    except Exception as e:
        logger.warning(f"Groq call failed for {primary.chunk_id} / {question_type}: {e}")
        return []

    candidates = _parse_generation_response(raw)
    results: list[GeneratedQA] = []

    for cand in candidates:
        # All four fields go through _coerce_str — JSON-mode output isn't
        # guaranteed to give us strings (see _coerce_str's docstring). This
        # was previously `(cand.get("answer") or "").strip()`, which only
        # guards against None/"" and crashes on a truthy non-string like a
        # bare JSON number.
        question = _coerce_str(cand.get("question"))
        answer = _coerce_str(cand.get("answer"))
        source_passage = _coerce_str(cand.get("source_passage"))
        q_type = _coerce_str(cand.get("question_type")) or question_type

        if not question or not answer or not source_passage:
            logger.debug(f"Dropping incomplete candidate from {primary.chunk_id}: {cand}")
            continue

        score = _verbatim_or_table_match_score(source_passage, passage_text, is_table)
        if score < gen_cfg["verbatim_match_threshold"]:
            logger.info(
                f"Dropping candidate (verbatim_match={score:.2f} < "
                f"{gen_cfg['verbatim_match_threshold']}) from {primary.chunk_id}: "
                f"{question[:60]!r}"
            )
            continue

        results.append(
            GeneratedQA(
                id=str(uuid.uuid4()),
                question=question,
                answer=answer,
                source_passage=source_passage,
                question_type=q_type,
                heading_path=primary.heading_path,
                part=primary.part,
                item=primary.item,
                source_chunk_ids=[c.chunk_id for c in chunk_group],
                verbatim_match_score=score,
            )
        )

    return results


def generate_all(chunks: list[Chunk], client: OpenAI | None = None) -> list[GeneratedQA]:
    """
    Top-level entry point: for every question_type, build chunk groups
    (pairing where applicable) and run one generation call per group,
    aggregating all surviving GeneratedQA across the whole filing.
    """
    if client is None:
        client = _build_client()

    gen_cfg = CONFIG["generation"]
    all_results: list[GeneratedQA] = []

    for question_type in gen_cfg["question_types"]:
        groups = _maybe_pair_chunks(chunks, question_type)
        logger.info(f"Generating '{question_type}' over {len(groups)} chunk-group(s)")
        for group in groups:
            results = generate_for_chunk_group(client, group, question_type)
            all_results.extend(results)

    logger.info(f"Generation complete: {len(all_results)} candidate QA pairs survived verbatim check")
    return all_results


if __name__ == "__main__":
    from chunker import load_chunks

    chunks = load_chunks()
    qa_pairs = generate_all(chunks)
    print(f"Generated {len(qa_pairs)} candidate QA pairs.")
    for qa in qa_pairs[:3]:
        print(f"  [{qa.question_type}] {qa.question}")