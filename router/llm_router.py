#!/usr/bin/env python3
"""LLM-based prompt router (OpenAI-compatible). Falls back to None if unavailable."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from router.query_intent import THEME_EXPANSIONS

SCHEMA_CARD = """
Structured fields you MAY filter on (never put these into themes/embeddings):
- country / region (ISO country or europe/eu/asia/north america/scandinavia)
- is_public (public vs private)
- revenue_min / revenue_max
- employee_min / employee_max
- year_founded min/max
- top-N sort by revenue or employees

Semantic / text fields (need embeddings, NOT SQL-like filters):
- description, target_markets, core_offerings, business_model, NAICS label
""".strip()

# Tokens that belong in structured filters — must never appear in themes/embeddings.
_FILTER_NOISE = frozenset(
    {
        "public",
        "private",
        "revenue",
        "employee",
        "employees",
        "headcount",
        "founded",
        "after",
        "before",
        "since",
        "over",
        "under",
        "more",
        "less",
        "fewer",
        "than",
        "million",
        "billion",
        "thousand",
        "startup",
        "startups",
        "fast-growing",
        "growing",
        "leading",
        "top",
        "largest",
        "smallest",
    }
)

SYSTEM = f"""You classify company-search prompts into one route AND decompose semantic intent.

Routes:
- structured: ONLY typed filters/aggregates (geo, public, revenue, employees, top-N). No industry theme.
- context: ONLY industry/product/theme meaning. No country/region/public/revenue/employee/founded/top-N.
- hybrid: BOTH at least one structured filter AND a semantic theme.

HARD RULES for themes (violations cause retrieval failure):
1. primary_theme = SHORT industry/product target only (1-4 words). NEVER the full user sentence.
2. NEVER put filters into primary_theme / semantic_query / embedding_queries:
   no country/region, public/private, revenue/$, employee counts, founded years, top-N, "fast-growing".
3. embedding_queries = 2-5 SHORT phrases (1-3 words each). primary_theme FIRST, then synonyms.
   Never paste the full prompt. Never embed contrast/exclusions.
4. "using Y or similar" (Shopify, Stripe, SAP, Salesforce, Epic, …): primary = industry
   operators; put Y in contrast to embedding — prefer category synonyms, not the brand.
5. "competing with / vs X" → contrast=X (do NOT embed X). Target stays the primary industry.
6. "X for Y" / "X providing Y" → primary=X (or X+Y compound), secondary=Y when Y refines the buyer/use-case.
7. Fix obvious typos toward canonical forms (logistic→logistics).

Few-shot (follow these shapes exactly):
- "Logistic companies in Romania"
  → hybrid; primary="logistics"; emb=["logistics","freight transport","warehousing"]
- "Public software companies with more than 1,000 employees."
  → hybrid; primary="software"; emb=["software","enterprise software","saas"]
- "Food and beverage manufacturers in France"
  → hybrid; primary="food and beverage"; emb=["food and beverage","food manufacturing","beverage production"]
- "Companies that could supply packaging materials for a direct-to-consumer cosmetics brand"
  → context; primary="packaging materials"; secondary="cosmetics"
  → emb=["packaging materials","cosmetic packaging","packaging suppliers"]
- "Construction companies in the United States with revenue over $50 million"
  → hybrid; primary="construction"; emb=["construction","building construction"]
- "Pharmaceutical companies in Switzerland"
  → hybrid; primary="pharmaceutical"; emb=["pharmaceutical","pharma","biopharma"]
- "B2B SaaS companies providing HR solutions in Europe"
  → hybrid; primary="HR SaaS"; secondary=""; emb=["HR SaaS","hr software","payroll software","saas"]
- "Clean energy startups founded after 2018 with fewer than 200 employees"
  → hybrid; primary="clean energy"; emb=["clean energy","renewable energy","solar","wind energy"]
- "Fast-growing fintech companies competing with traditional banks in Europe."
  → hybrid; primary="fintech"; contrast="traditional banks"
  → emb=["fintech","digital banking","neobank","payment solutions"]
- "E-commerce companies using Shopify or similar platforms"
  → context; primary="e-commerce"; secondary=""; contrast=""
  → emb=["e-commerce","online retail","ecommerce","direct to consumer","online store"]
  # operators that may use Shopify-class tools; NOT storefront-platform vendors
- "Fintech companies using Stripe or similar platforms"
  → context; primary="fintech"; secondary=""
  → emb=["fintech","payment processing","online payments","digital payments"]
  # payment operators; NOT payment-gateway platform vendors
- "Logistics companies using SAP or similar systems"
  → context; primary="logistics"; secondary=""
  → emb=["logistics","freight transport","warehousing","supply chain operations"]
- "Healthcare companies using Epic or similar EHR systems"
  → context; primary="healthcare"; secondary=""
  → emb=["healthcare","hospital IT","clinical software","ehr systems"]
- "Renewable energy equipment manufacturers in Scandinavia"
  → hybrid; primary="renewable energy equipment"; emb=["renewable energy","wind turbines","solar equipment"]
- "Companies that manufacture or supply critical components for electric vehicle battery production"
  → context; primary="EV battery components"; secondary="electric vehicles"
  → emb=["EV battery components","battery materials","electric vehicle batteries"]

{SCHEMA_CARD}

Return ONLY compact JSON:
{{"route":"structured"|"context"|"hybrid","primary_theme":"...","secondary_theme":"...","contrast":"...","exclusions":["..."],"semantic_query":"...","embedding_queries":["..."],"reason":"..."}}
"""


@dataclass
class LLMRoute:
    route: str
    semantic_query: str
    reason: str
    embedding_queries: list[str] = field(default_factory=list)
    primary_theme: str = ""
    secondary_theme: str = ""
    contrast: str = ""
    exclusions: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def _strip_filter_noise(text: str) -> str:
    """Remove structured-filter language that must not enter themes/embeddings."""
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"\blogistic\b", "logistics", t, flags=re.I)
    # Money / headcount / year fragments
    t = re.sub(
        r"\$?\s*\d[\d,]*(?:\.\d+)?\s*(?:k|m|million|b|billion|thousand|employees?|people|staff)?",
        " ",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"\b(?:more than|less than|fewer than|greater than|at least|at most|over|under|"
        r"above|below|after|before|since|founded)\b",
        " ",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"\b(?:revenue|employee(?:s)?|headcount|public|private|startup(?:s)?|"
        r"fast[- ]growing|growing|leading|top)\b",
        " ",
        t,
        flags=re.I,
    )
    # Common geos that leak into themes (full geo handling is in reconcile/filters)
    t = re.sub(
        r"\b(?:romania|france|switzerland|europe|european|eu|asia|scandinavia|nordics|"
        r"united states|usa|us|uk|germany|china|india|australia|israel|spain|"
        r"netherlands|north america|middle east)\b",
        " ",
        t,
        flags=re.I,
    )
    t = re.sub(r"\s+", " ", t).strip(" ,.-?")
    # Drop leftover filter-only tokens
    kept = [
        w
        for w in t.split()
        if w.lower().strip(".,") not in _FILTER_NOISE and not w.isdigit()
    ]
    return " ".join(kept).strip(" ,.-?")


def _expand_theme_synonyms(theme: str) -> list[str]:
    key = theme.lower().strip()
    if key in THEME_EXPANSIONS:
        return list(THEME_EXPANSIONS[key])
    matches: list[tuple[int, str, list[str]]] = []
    for anchor, syns in THEME_EXPANSIONS.items():
        if anchor in key:
            matches.append((len(anchor), anchor, syns))
    if not matches:
        return []
    matches.sort(key=lambda x: x[0], reverse=True)
    _, anchor, syns = matches[0]
    # Don't dilute niche themes ("carbon accounting software") with generic saas
    if anchor in {"software", "saas", "hr"} and key != anchor:
        return []
    return list(syns)


def _normalize_embedding_queries(
    raw: Any,
    *,
    semantic_query: str = "",
) -> list[str]:
    """Dedupe, strip filter noise, and expand short themes into synonyms."""
    items: list[str] = []
    if isinstance(raw, str) and raw.strip():
        items = [raw.strip()]
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                items.append(item.strip())

    if semantic_query.strip():
        items = [semantic_query.strip()] + items

    seen: set[str] = set()
    out: list[str] = []

    def _add(q: str) -> None:
        q = _strip_filter_noise(q)
        if not q or len(q) > 60 or len(q.split()) > 5:
            return
        key = q.lower()
        if key in seen:
            return
        # Skip pure filter leftovers
        if all(tok in _FILTER_NOISE or tok.isdigit() for tok in key.split()):
            return
        seen.add(key)
        out.append(q)

    for q in items:
        _add(q)
        if len(out) >= 5:
            break

    # If the model returned a single thin phrase, expand synonyms
    if out and len(out) < 3:
        for syn in _expand_theme_synonyms(out[0]):
            _add(syn)
            if len(out) >= 5:
                break

    # Drop generic SaaS fillers when a longer niche primary is present
    if out and len(out[0].split()) >= 2:
        generic = {"software", "saas", "enterprise software", "software as a service", "cloud software"}
        trimmed = [q for q in out if q.lower() not in generic or q.lower() == out[0].lower()]
        if trimmed:
            out = trimmed

    return out[:5]


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def classify_with_llm(
    prompt: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 60.0,
) -> LLMRoute | None:
    """
    Classify via OpenAI-compatible Chat Completions API.
    Works with OpenAI, Gemini (AI Studio), Groq, Ollama, etc.
    Returns None if no API key / request fails.
    """
    provider = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    env_base = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OLLAMA_BASE_URL")

    using_ollama = provider == "ollama" or bool(
        os.environ.get("OLLAMA_HOST")
        or (env_base and "11434" in env_base)
    )
    using_gemini = bool(
        provider == "gemini"
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or (env_base and "generativelanguage.googleapis.com" in env_base)
    )

    api_key = (
        api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ("ollama" if using_ollama else None)
    )
    if not api_key and not using_ollama:
        return None

    if using_ollama:
        base_url = env_base or "http://127.0.0.1:11434/v1"
        model = model or os.environ.get("OPENAI_MODEL") or os.environ.get("OLLAMA_MODEL") or "qwen2.5:7b"
        api_key = api_key or "ollama"
    elif using_gemini and not env_base:
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        model = model or os.environ.get("OPENAI_MODEL") or "gemini-2.0-flash"
    else:
        model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        base_url = env_base or "https://api.openai.com/v1"

    base_url = base_url.rstrip("/")
    url = f"{base_url}/chat/completions"

    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    # Some local models reject response_format; Gemini/OpenAI benefit from it.
    if not using_ollama:
        body["response_format"] = {"type": "json_object"}

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        if exc.code == 429:
            print(
                "llm_router: quota/rate limit hit (HTTP 429). "
                "Falling back to rules. Wait, switch model, or use --rules-only.",
                file=__import__("sys").stderr,
            )
        else:
            print(f"llm_router HTTP {exc.code}: {detail}", file=__import__("sys").stderr)
        return None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"llm_router error: {exc}", file=__import__("sys").stderr)
        return None

    try:
        content = payload["choices"][0]["message"]["content"]
        data = _extract_json(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        print("llm_router: could not parse model JSON; falling back to rules", file=__import__("sys").stderr)
        return None

    route = str(data.get("route", "")).strip().lower()
    if route not in {"structured", "context", "hybrid"}:
        return None

    semantic = str(data.get("semantic_query") or data.get("primary_theme") or "").strip()
    primary = str(data.get("primary_theme") or semantic).strip()
    secondary = str(data.get("secondary_theme") or "").strip()
    contrast = str(data.get("contrast") or "").strip()
    exclusions_raw = data.get("exclusions") or []
    exclusions: list[str] = []
    if isinstance(exclusions_raw, str) and exclusions_raw.strip():
        exclusions = [exclusions_raw.strip()]
    elif isinstance(exclusions_raw, list):
        for item in exclusions_raw:
            if isinstance(item, str) and item.strip():
                exclusions.append(item.strip())
    exclusions = exclusions[:5]

    if route == "structured":
        semantic = ""
        primary = ""
        secondary = ""
        contrast = ""
        exclusions = []
        embedding_queries: list[str] = []
    else:
        primary = _strip_filter_noise(primary)
        secondary = _strip_filter_noise(secondary)
        semantic = _strip_filter_noise(semantic) or primary
        if primary and not semantic:
            semantic = primary
        elif semantic and not primary:
            primary = semantic

        embedding_queries = _normalize_embedding_queries(
            data.get("embedding_queries"),
            semantic_query=primary or semantic,
        )
        # Strip contrast/exclusion phrases from embedding queries
        blocked = {c.lower() for c in ([contrast] + exclusions) if c.strip()}
        embedding_queries = [
            q
            for q in embedding_queries
            if not any(b in q.lower() or q.lower() in b for b in blocked if len(b) >= 4)
        ]
        if not embedding_queries and (primary or semantic):
            embedding_queries = _normalize_embedding_queries([], semantic_query=primary or semantic)
        if not semantic and embedding_queries:
            semantic = embedding_queries[0]
            primary = primary or semantic

    reason = str(data.get("reason") or "llm classifier").strip()
    return LLMRoute(
        route=route,
        semantic_query=semantic,
        embedding_queries=embedding_queries,
        primary_theme=primary,
        secondary_theme=secondary,
        contrast=contrast,
        exclusions=exclusions,
        reason=reason,
        raw=data,
    )
