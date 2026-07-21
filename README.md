# Company Ranking & Qualification

Company search is not only a retrieval problem—it is also a qualification problem. A query such as *“Find logistics companies in Germany”* may retrieve a German freight forwarder, a software vendor building logistics tools, and a foreign warehouse operating near the border. All are related to the query, but they do not match the user’s intent equally well.

Qualifying every candidate with a large language model can improve relevance, but doing so is costly, slow, and difficult to scale. It can also produce inconsistent decisions for borderline cases and wastes expensive reasoning on queries that simple filters can resolve.

This project implements a multi-stage ranking and qualification pipeline designed to balance **accuracy, latency, cost, and scalability**. It classifies each request as structured, semantic, or hybrid; retrieves candidates using typed filters and embeddings; reranks them with field-aware evidence; and optionally uses an LLM to produce a grounded answer from the final results.

The system is evaluated on a Veridion-style dataset of approximately 477 companies and was developed for the Veridion ML Engineer Intern challenge.

For the reasoning behind the design, trade-offs, and implementation decisions, see the [project write-up](write-up.md).

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional LLM (recommended): local Ollama
ollama pull qwen2.5:7b
export LLM_PROVIDER=ollama
# export OLLAMA_MODEL=qwen2.5:7b   # default

python main.py 'Logistics companies in Romania'
```

Use **single quotes** around prompts that contain `$` (e.g. revenue amounts), or the shell will expand them.

Without an LLM, classification falls back to rules and you can skip answer synthesis with `--no-answer` / `--rules-only`.

---

## Example queries

| Prompt | Expected route |
|---|---|
| `Companies in Germany` | `structured` — country filter |
| `Top 5 companies by revenue in US` | `structured` — filter + sort |
| `Firms focused on renewable energy` | `context` — semantic search |
| `Logistics companies in Romania` | `hybrid` — filter, then rank |

```bash
python main.py 'Top 5 companies by revenue in US'
python main.py 'Companies supplying packaging for cosmetics brands'
python main.py 'Logistics companies in Romania' --no-answer
python main.py 'Public software companies in Europe' --json
```

---

## What you get back

| Field | Description |
|---|---|
| `route` | `structured` \| `context` \| `hybrid` |
| `filters` | Parsed structured constraints (country, revenue, employees, year, public/private, sort, …) |
| `matches` | Ranked company summaries (name, country, revenue, employees, website); includes rerank/retrieval scores when semantic |
| `evidence` | Per-match confidence and why it matched (field overlap) |
| `answer` | LLM synthesis grounded only in evidence (unless `--no-answer`) |
| `intent` / `embedding_queries` | Decomposed theme used for retrieval (context/hybrid) |

---

## How it works

```
prompt
  │
  ├─ LLM classifier (+ rules reconcile)  →  route + intent
  │
  ├─ structured  →  typed filters / sort / count on company fields
  ├─ context     →  embedding recall over text fields → field rerank
  └─ hybrid      →  structured subset → embedding recall → field rerank
  │
  └─ evidence objects → LLM answer (optional)
```

- **Structured fields:** country/region/city, revenue, employees, year founded, public/private, sort/limit/count
- **Context and qualification fields:** description, NAICS, business model, target markets, core offerings, and related enrichment text
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (cached under `.embedding_cache/`)
- **Reranker:** field-aware scoring with theme / tool / contrast / exclusion handling

---

## CLI flags

```bash
python main.py '<prompt>' [options]

  -n, --limit N       Max matches to return (default: 10)
  --json              Print full result dict as JSON
  --no-answer         Retrieval + evidence only (no LLM write-up)
  --rules-only        Disable LLM classifier; rules route only
  --keyword           Keyword retrieval instead of embeddings
  --rebuild-index     Rebuild embedding cache
  -f, --file PATH     Alternate companies JSONL path
```

---

## LLM providers

| Provider | Setup |
|---|---|
| **Ollama** (default in `.env.example`) | `export LLM_PROVIDER=ollama` and run a local model |
| **Gemini** | Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) |
| **OpenAI-compatible** | Set `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL` / `OPENAI_MODEL`) |

See `.env.example` for a minimal Ollama config.

---

## Project layout

```
main.py                 CLI entry point
data/companies.jsonl    Company dataset
router/                 Classification, retrieval, rerank, evidence, answer
scripts/
  eval_router.py        Golden-set routing evaluation
  generate_context_hints.py
  fix_nested_fields.py
requirements.txt
.env.example
```

---

## Evaluation

Check routing quality against a fixed golden set:

```bash
export LLM_PROVIDER=ollama
python scripts/eval_router.py
python scripts/eval_router.py --rules-only
python scripts/eval_router.py --verbose
```

---

## Dataset fields

Each line in `data/companies.jsonl` includes roughly:

`operational_name`, `website`, `address`, `year_founded`, `employee_count`, `revenue`, `is_public`, `primary_naics`, `secondary_naics`, `description`, `business_model`, `target_markets`, `core_offerings`
