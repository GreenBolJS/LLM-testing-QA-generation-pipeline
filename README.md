# 10-K Financial QA Generation Pipeline

Generates a verified dataset of ≥100 question–answer pairs from a single SEC
10-K filing, with hallucination-checked answers, source traceability,
question-type tagging, and difficulty labels — using only free-tier model
APIs (Groq for generation, Hugging Face Inference Providers for
verification, a local embedding model for dedup/recall scoring).

Default target filing: Microsoft's FY2025 10-K (configurable in
`config.yaml`).

---

## Quickstart

```bash
git clone https://github.com/GreenBolJS/LLM-testing-QA-generation-pipeline.git
cd LLM-testing-QA-generation-pipeline
pip install -r requirements.txt

cp env.example .env
# edit .env and fill in GROQ_API_KEY and HF_TOKEN

export $(grep -v '^#' .env | xargs)   # or use python-dotenv / direnv

python src/pipeline.py
```

Output lands at `output/msft_10k_2025_qa_dataset.csv` (or whatever
`output.file_name` is set to in `config.yaml`).

Useful flags:

```bash
python src/pipeline.py --force-refetch       # bypass the cached .htm, re-pull from EDGAR
python src/pipeline.py --skip-verification   # DEBUG ONLY — see warning in "Known Limitations"
```

Run tests:

```bash
pytest                        # fast unit tests (~80 tests), requires_model tests excluded by default via pytest.ini
pytest -m requires_model      # additionally runs the real-bge-small dedup smoke test (needs network access)
```

---

## Pipeline Stages

```
fetch → chunk (+ extract tables) → generate → verify → dedup → difficulty → output
```

| Stage | Module | What it does |
|---|---|---|
| 1. Fetch | `src/fetch_filing.py` | Pulls the raw EDGAR `.htm` (never a PDF print) with a descriptive `User-Agent`, caches to `data/raw/` |
| 2. Chunk | `src/chunker.py`, `src/table_extractor.py` | Heading-hierarchy chunking of narrative text; tables pulled separately via `pandas.read_html()` and kept structured |
| 3. Generate | `src/generator.py` | Groq `llama-3.1-8b-instant`, one call per (question_type, chunk-or-chunk-pair); fuzzy verbatim-match check on `source_passage` |
| 4. Verify | `src/verifier.py`, `src/numeric_check.py` | HF Inference Providers `gemma-3-27b-it` faithfulness re-derivation for non-numeric types; deterministic arithmetic recompute for `numeric_calculation` |
| 5. Dedup | `src/dedup.py` | `bge-small-en-v1.5` embeddings, pairwise cosine similarity > 0.90 → drop the later duplicate |
| 6. Difficulty | `src/difficulty.py` | Rule-based labeling from generation metadata only (question_type + chunk count) — never LLM self-report |
| Output | `src/pipeline.py` | Flattens everything into one CSV (or JSONL) row per verified pair |

---

## Design Decisions

### Why heading-hierarchy chunking, not fixed-size windows
10-K filings have a very regular, predictable structure (`PART I`, `ITEM 7.`,
etc.) plus locally-styled sub-headings (bold ALL CAPS section headers,
italic sub-headers). Anchoring chunk boundaries to this structure means
every chunk comes with real section context for free — `"Item 7A > RISKS >
Foreign Currencies"` tells the generator (and later, anyone auditing the
output) exactly where a fact came from, which a sliding window over raw
text can't reliably reconstruct.

`PART`/`ITEM` are regex-matched directly since their formatting is
consistent across virtually all 10-Ks. Everything below that is classified
by a short heuristic (short text, no terminal punctuation, bold/underline/
italic styling) rather than a fixed tag whitelist, because EDGAR filers use
wildly inconsistent HTML for sub-headers — sometimes a real `<h3>`,
sometimes just a bolded `<p>` or a `<span style="font-weight:bold">`.

### Why tables are extracted separately, not flattened into chunk text
Numeric and comparison questions need rows/columns to stay aligned —
flattening a table into prose loses the structure that makes "compare
segment A's FY25 revenue to FY24" answerable at all. `table_extractor.py`
keeps each table as structured data (DataFrame-derived rows/columns) tagged
with the same heading-path metadata chunks get, then `pipeline.py` renders
each table to a compact pipe-delimited text block only at the point of
calling the generation LLM (which needs *some* text representation), via
`table_to_passage_text()`. The structured form is what gets saved to
`data/chunks/tables.json` for inspection/debugging.

### Why generation and verification use different model families
`llama-3.1-8b-instant` (Groq) generates; `gemma-3-27b-it` (HF) independently
re-derives the answer from the passage and checks it against the
generator's claim. Using the same model for both would mean the model could
share the same blind spots or biases for both producing and grading an
answer. Different model families closes that loop.

### Why numeric_calculation skips the LLM verifier entirely
Free-tier LLMs (especially 8-9B models) are not reliable at arithmetic.
Asking Gemma to "check the math" would just substitute one unreliable
calculator for another. `numeric_check.py` instead extracts every number in
the source passage via regex and tries every simple named relationship
(difference, absolute difference, sum, percent change, ratio, product)
against the claimed answer — if any relationship matches within tolerance,
it passes. This is deterministic, free, and catches the exact class of
error (wrong arithmetic) that LLM-as-judge reliably misses. It is
intentionally conservative: it does **not** validate complex multi-number
calculations (e.g. three-way weighted averages) — those report
`numeric_check_pass=False` with a `reason` explaining why, rather than
silently passing.

### Why three verification signals are stored, not collapsed to one pass/fail
Per the output schema, each row carries `context_recall_score`,
`faithfulness_pass` (or `numeric_check_pass`), and `verbatim_match_score`
independently. This means you can re-filter the dataset at a stricter or
looser threshold later (e.g. "give me only rows where all three signals
agree") without re-running any LLM calls.

### Why dedup runs on question embeddings, not answer/passage embeddings
The same underlying fact (e.g. a headline revenue growth %) often gets
asked about in both an Overview bullet and the detailed segment narrative,
phrased differently each time. Embedding the *question* text and dropping
near-duplicates above a 0.90 cosine similarity threshold catches this
rephrasing — a real revenue-growth duplicate it will catch even with totally
different wording, where comparing the underlying passages or answers
(which would often be byte-for-byte different anyway) wouldn't help nearly
as much.

A meaningful drop rate here is **expected, not a bug** — see the spec note
this carries forward from.

### Why difficulty labeling never asks an LLM
Difficulty is derived purely from generation metadata: `question_type` and
how many source chunks the generation call used. `fact_extraction` over a
single chunk → `easy`; `numeric_calculation` → `medium`; `comparison` /
`multi_step_reasoning`, or **any** question generated from 2+ chunks
(regardless of type) → `hard`. This keeps difficulty fully reproducible and
free — no model call, no subjectivity, no risk of a model rating its own
question as "easy" to seem more impressive.

---

## Known Limitations

- **HF Inference Providers free-tier capacity is the practical bottleneck.**
  `gemma-3-27b-it` verification calls are retried 2–3 times with backoff to
  absorb cold starts and transient 503s on whichever partner provider's free
  tier the request lands on (`provider="auto"`). If you're scaling past a
  single filing, budget for this — see Scaling Note below.
- **`numeric_check.py` is conservative by design**, not exhaustive. It
  checks single-step relationships (difference, percent change, sum,
  ratio, product) between *pairs* of numbers found in the passage. A
  legitimate multi-number calculation a human could verify (e.g. a
  three-segment weighted average) will report `numeric_check_pass=False`
  rather than a false pass — inspect the `verifier_reason` column for these
  before discarding, since the underlying QA pair may still be valid.
- **EDGAR's raw HTML is not perfectly consistent across filers or years.**
  The heading-candidate heuristic (short text, no terminal punctuation,
  bold/underline/italic) is tuned against Microsoft's filing structure and
  common 10-K conventions; a filer with unusual formatting (e.g. headers as
  plain-text ALL CAPS with no styling at all) may need threshold tweaks in
  `config.yaml`'s `chunking` section.
- **No native page numbers.** Raw EDGAR `.htm` doesn't carry page numbers —
  those are a PDF-print artifact. The output schema's `source_page` column
  is left blank rather than fabricated; `source_section` (the full heading
  path) is the reliable traceability anchor instead.
- **`--skip-verification` exists only for pipeline-wiring debugging.** It
  marks every candidate as "kept" without running any faithfulness or
  numeric checks. Do not treat its output as a verified deliverable.
- **Heavy import cost on first test run.** `sentence-transformers` (used for
  embeddings) pulls in `torch`/`transformers` even before a model is
  loaded, adding a few seconds to the first pytest collection. This is a
  one-time session cost, not a per-test cost.
- **Live network calls (EDGAR, Groq, HF) were not exercised against
  production endpoints during this build** — `tests/` validates all pure
  logic (chunking, table extraction, numeric checks, dedup with a mocked
  embedding function) without requiring network access. Run
  `python src/pipeline.py` end-to-end yourself on first use and watch the
  logs, especially around the fetch and verification stages.

---

## Scaling Note

- **Groq generation is not the bottleneck.** `llama-3.1-8b-instant` has a
  high daily request quota on Groq's free tier.
- **HF Inference Providers verification is the practical bottleneck**,
  depending on which partner provider's free capacity a given request lands
  on. Plan around this if scaling up.
- To scale to multiple documents / 1000+ pairs:
  - Parallelize chunk-level generation calls (async, rate-limited).
  - Cache chunk→question results so re-runs don't regenerate from scratch.
  - Make filing metadata (ticker, CIK, fiscal year — already present in
    `config.yaml`'s `filing` block and carried through `Chunk`/`GeneratedQA`
    objects) a first-class output column so pairs from multiple filings
    merge cleanly into one dataset.
  - Spread verification calls across multiple days/runs if a provider's
    free daily quota becomes the binding constraint.

---

## Project Structure

```
caliper-qa-pipeline/
├── README.md
├── requirements.txt
├── .env.example
├── config.yaml
├── pytest.ini
│
├── src/
│   ├── config.py            # loads config.yaml, shared logging/env helpers
│   ├── fetch_filing.py      # EDGAR .htm fetch + cache
│   ├── chunker.py           # heading-hierarchy chunking
│   ├── table_extractor.py   # pandas.read_html table pull, heading-path tagged
│   ├── embeddings.py        # shared bge-small-en-v1.5 wrapper (verifier + dedup)
│   ├── generator.py         # Groq llama-3.1-8b-instant, per-question-type prompts
│   ├── verifier.py          # HF gemma-3-27b-it faithfulness check + context-recall
│   ├── numeric_check.py     # deterministic recompute for numeric_calculation
│   ├── dedup.py             # cosine-similarity near-duplicate filter
│   ├── difficulty.py        # rule-based difficulty labeling
│   └── pipeline.py          # orchestrates all stages, writes final output
│
├── prompts/
│   ├── generation_fact.txt
│   ├── generation_numeric.txt
│   ├── generation_comparison.txt
│   ├── generation_multistep.txt
│   └── verification_faithfulness.txt
│
├── data/
│   ├── raw/                 # cached downloaded filing(s)
│   └── chunks/               # intermediate chunked JSON + extracted tables JSON
│
├── output/
│   └── msft_10k_2025_qa_dataset.csv
│
└── tests/
    ├── conftest.py
    ├── test_chunker.py
    ├── test_numeric_check.py
    └── test_dedup.py
```

## Output Schema

One row per verified QA pair:

| Column | Description |
|---|---|
| `id` | UUID |
| `question` / `answer` | Generated question and answer |
| `source_passage` | Exact verbatim text the answer is grounded in |
| `source_item` / `source_section` / `source_page` | Traceability fields (`source_page` is blank — see Known Limitations) |
| `question_type` | `fact_extraction` / `numeric_calculation` / `comparison` / `multi_step_reasoning` |
| `difficulty` | `easy` / `medium` / `hard` |
| `context_recall_score` | bge-small cosine similarity, answer vs. source_passage |
| `faithfulness_pass` | Gemma re-derivation match (null for numeric_calculation) |
| `numeric_check_pass` | Deterministic arithmetic check result (null for non-numeric types) |
| `verbatim_match_score` | Generation-time fuzzy match of source_passage against the original chunk |
| `verifier_confidence` / `verifier_reason` | Gemma's self-reported confidence + one-line reasoning, or the numeric_check's reasoning |
| `source_chunk_ids` | Semicolon-joined chunk ID(s) used to generate this pair |
