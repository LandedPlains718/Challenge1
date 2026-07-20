#!/usr/bin/env python3
"""Rerank retrieved companies using structured business fields."""

from __future__ import annotations

import re
from typing import Any

# Field weights — NAICS and offerings are highest-signal for industry fit.
FIELD_WEIGHTS = {
    "naics": 0.32,
    "core_offerings": 0.30,
    "target_markets": 0.23,
    "business_model": 0.15,
}

_STOP = frozenset(
    {
        "and",
        "the",
        "for",
        "with",
        "from",
        "other",
        "general",
        "services",
        "service",
        "products",
        "product",
        "solutions",
        "management",
        "company",
        "companies",
        "business",
        "manufacturing",
        "manufacturer",
    }
)

# High-frequency tokens that must not dominate rerank scores alone.
_GENERIC_TOKENS = frozenset(
    {
        "software",
        "saas",
        "platform",
        "platforms",
        "solution",
        "solutions",
        "service",
        "services",
        "system",
        "systems",
        "tool",
        "tools",
        "technology",
        "technologies",
        "tech",
        "digital",
        "online",
        "cloud",
        "enterprise",
        "application",
        "applications",
        "app",
        "apps",
        "data",
        "automation",
        "automated",
        "ai",
        "ml",
        "analytics",
        "consulting",
        "development",
        "developer",
        "providers",
        "provider",
        "vendor",
        "vendors",
        "retail",
        "wholesale",
        "commercial",
        "industrial",
        "consumer",
        "business",
        # Keep "ecommerce" specific — it is often the primary search theme.
        "management",
        "support",
        "process",
        "processes",
        "smart",
        "native",
        "powered",
        "based",
        "advanced",
        "modern",
        "direct",  # alone (e.g. "direct to consumer") matches packaging suppliers
    }
)

# Generic business-model terms that should not dominate reranking.
# Note: ecommerce is intentionally NOT weak — "E-commerce" on business_model
# is a strong merchant signal for e-commerce / Shopify-style queries.
_WEAK_MODEL_TERMS = frozenset(
    {
        "retail",
        "wholesale",
        "enterprise",
        "commercial",
        "industrial",
        "consumer",
        "business",
        "service",
        "provider",
        "software",
        "saas",
    }
)

# Query themes that map to NAICS 522xx (finance / payments / lending).
_FINANCE_QUERY_TERMS = frozenset(
    {
        "fintech",
        "bank",
        "banking",
        "payment",
        "payments",
        "lending",
        "finance",
        "financial",
        "neobank",
        "wallet",
        "remittance",
        "lender",
        "credit",
    }
)


def _normalize_theme_text(text: str) -> str:
    """Normalize hyphenated / spaced theme forms before tokenization."""
    t = (text or "").lower()
    t = t.replace("e-commerce", "ecommerce")
    t = t.replace("e commerce", "ecommerce")
    t = t.replace("e–commerce", "ecommerce")
    return t


def _tokens(text: str) -> list[str]:
    text = _normalize_theme_text(text)
    return [
        t
        for t in re.findall(r"[a-z0-9]+", text)
        if len(t) > 2 and t not in _STOP
    ]


# Tokens that are too weak alone (false friends: "Store Development" ≠ "online store").
_WEAK_ALONE_TOKENS = frozenset(
    {
        "store",
        "stores",
        "shop",
        "shops",
        "commerce",
        "sale",
        "sales",
        "channel",
        "channels",
    }
)


def _is_generic_token(token: str) -> bool:
    return token in _GENERIC_TOKENS or token in _STOP or token in _WEAK_ALONE_TOKENS


def _specific_tokens(text: str) -> list[str]:
    return [t for t in _tokens(text) if not _is_generic_token(t)]


def _partition_query_tokens(queries: list[str]) -> tuple[list[str], list[str]]:
    """Split all query tokens into (specific, generic), preserving order."""
    specific: list[str] = []
    generic: list[str] = []
    seen: set[str] = set()
    for q in queries:
        for t in _tokens(q):
            if t in seen:
                continue
            seen.add(t)
            if _is_generic_token(t):
                generic.append(t)
            else:
                specific.append(t)
    return specific, generic


def _token_in_blob(token: str, blob: str) -> bool:
    if token in blob:
        return True
    if len(token) >= 5:
        stem = token.rstrip("s")
        if stem in blob:
            return True
    # "logistics" should match "intralogistics", etc.
    if len(token) >= 5 and token in blob.replace("-", ""):
        return True
    return False


def _phrase_strength(query: str, blob: str) -> float:
    q = _normalize_theme_text(query).strip()
    if not q or not blob:
        return 0.0
    blob_l = _normalize_theme_text(blob)
    if q in blob_l:
        return 1.0
    # Compound forms: logistics ⊂ intralogistics
    if len(q) >= 5 and q in blob_l.replace("-", " "):
        return 0.90
    if len(q) >= 5 and any(
        q in tok for tok in re.findall(r"[a-z0-9]+", blob_l) if len(tok) > len(q)
    ):
        return 0.85

    # Prefer specific-token overlap; ignore generics for the strength signal
    specific = _specific_tokens(q)
    parts = specific or _tokens(q)
    if not parts:
        return 0.0
    if all(_token_in_blob(p, blob_l) for p in parts):
        # Full specific match — strong even if generics missing
        return 0.85 if specific else 0.55
    hits = sum(1 for p in parts if _token_in_blob(p, blob_l))
    if not hits:
        return 0.0
    frac = hits / len(parts)
    # Generic-only partial matches stay weak
    if not specific:
        return min(frac, 1.0) * 0.25
    return min(frac, 1.0) * 0.55


def _is_ecommerce_query(queries: list[str]) -> bool:
    blob = _normalize_theme_text(" ".join(queries))
    return any(
        t in blob
        for t in ("ecommerce", "shopify", "online store", "online retail", "webshop")
    )


def _has_ecommerce_platform_signal(offerings: str) -> bool:
    """True when offerings look like storefront/platform/checkout capability."""
    needles = (
        "ecommerce platform",
        "e-commerce platform",
        "online store",
        "online storefront",
        "online shop platform",
        "webshop",
        "shopping cart",
        "storefront",
        "checkout",
        "point-of-sale",
        "point of sale",
    )
    if any(n in offerings for n in needles):
        return True
    if re.search(r"\bpos\b", offerings):
        return True
    if re.search(r"\bpayment\b", offerings) and re.search(
        r"\b(online|ecommerce|shopping|checkout)\b", offerings
    ):
        return True
    return False


def _ecommerce_channel_only(row: dict[str, Any]) -> bool:
    """True when E-commerce is mainly a business_model tag on a retail/wholesale firm."""
    models = {m.lower() for m in (row.get("business_model") or []) if isinstance(m, str)}
    has_ecom_model = any("ecommerce" in _normalize_theme_text(m) for m in models)
    if not has_ecom_model:
        return False
    omnichannel = bool(models & {"retail", "wholesale"})
    markets = _normalize_theme_text(_field_text(row, "target_markets"))
    offerings = _normalize_theme_text(_field_text(row, "core_offerings"))
    thematic_ecom = "ecommerce" in markets or "ecommerce" in offerings
    platformish = _has_ecommerce_platform_signal(offerings)
    # Omnichannel retailer with only a model tag → still a valid *merchant* signal;
    # kept for diagnostics only (merchant boost uses business_model directly).
    return omnichannel and not thematic_ecom and not platformish


def _is_ecommerce_platform_vendor(row: dict[str, Any]) -> bool:
    """Builds storefront/platform software — not an e-commerce merchant."""
    offerings = _normalize_theme_text(_field_text(row, "core_offerings"))
    models = {
        _normalize_theme_text(m)
        for m in (row.get("business_model") or [])
        if isinstance(m, str)
    }
    has_merchant_model = any("ecommerce" in m for m in models)
    vendor_signals = (
        "ecommerce platform development",
        "e-commerce platform development",
        "online shop platform development",
        "ecommerce platform",
        "online storefront",
        "shopping cart",
        "storefront platform",
        "online shop platform",
    )
    if any(s in offerings for s in vendor_signals):
        return not has_merchant_model
    if _has_ecommerce_platform_signal(offerings) and not has_merchant_model:
        if any(x in " ".join(models) for x in ("software", "saas", "service provider")):
            return True
    return False


def _packaging_supplier_for_ecommerce(row: dict[str, Any]) -> bool:
    """Makes packaging for DTC/retail — not an online merchant itself."""
    offerings = _normalize_theme_text(_field_text(row, "core_offerings"))
    if not offerings:
        return False
    pack_hits = len(re.findall(r"packag", offerings))
    if pack_hits < 2 and "packaging manufacturing" not in offerings:
        return False
    # Real retailers may mention packaging once (sustainability); suppliers dominate.
    retail_sell = any(
        p in offerings
        for p in (
            "apparel",
            "clothing",
            "grocery",
            "sporting goods",
            "fashion",
            "footwear",
            "beverage retail",
            "product retail",
            "online retail",
        )
    )
    return pack_hits >= 2 and not retail_sell


def _ecommerce_adjacent_only(row: dict[str, Any]) -> bool:
    """
    Serves e-commerce as a customer segment (logistics/postal/payments/packaging)
    but is not itself an e-commerce merchant.
    """
    if _packaging_supplier_for_ecommerce(row):
        return True
    models = {
        _normalize_theme_text(m)
        for m in (row.get("business_model") or [])
        if isinstance(m, str)
    }
    # Explicit E-commerce business model ⇒ merchant identity, not "adjacent".
    if any("ecommerce" in m for m in models):
        return False
    markets = _normalize_theme_text(_field_text(row, "target_markets"))
    offerings = _normalize_theme_text(_field_text(row, "core_offerings"))
    if "ecommerce" not in markets and "ecommerce" not in offerings:
        adjacent_model = any(
            x in " ".join(models)
            for x in ("logistics", "transportation", "postal", "shipping")
        )
        return adjacent_model and not _has_ecommerce_platform_signal(offerings)

    if _has_ecommerce_platform_signal(offerings):
        return False
    # E-commerce listed as a market for a logistics/warehouse/postal operator
    adjacent_model = any(
        x in " ".join(models)
        for x in ("logistics", "transportation", "warehouse", "postal", "shipping")
    )
    naics = _normalize_theme_text(_field_text(row, "naics"))
    adjacent_naics = any(
        x in naics
        for x in ("postal", "courier", "warehous", "truck", "freight", "logistics")
    )
    return adjacent_model or adjacent_naics


def _is_ecommerce_merchant(row: dict[str, Any]) -> bool:
    """Company whose identity includes e-commerce / online retail selling."""
    if _is_ecommerce_platform_vendor(row):
        return False
    if _packaging_supplier_for_ecommerce(row):
        return False
    models = {
        _normalize_theme_text(m)
        for m in (row.get("business_model") or [])
        if isinstance(m, str)
    }
    if any("ecommerce" in m for m in models):
        return True
    if _ecommerce_adjacent_only(row):
        return False
    markets = _normalize_theme_text(_field_text(row, "target_markets"))
    offerings = _normalize_theme_text(_field_text(row, "core_offerings"))
    if "ecommerce" in markets:
        return True
    if any(
        p in offerings
        for p in (
            "online retail",
            "ecommerce retail",
            "webshop",
            "online store operation",
            "direct-to-consumer",
            "direct to consumer",
        )
    ):
        return True
    return False


def _field_text(row: dict[str, Any], field: str) -> str:
    if field == "naics":
        code = row.get("naics_code") or ""
        label = row.get("naics_label") or ""
        return f"{code} {label}".strip()
    if field == "core_offerings":
        return " ".join(row.get("core_offerings") or [])
    if field == "target_markets":
        return " ".join(row.get("target_markets") or [])
    if field == "business_model":
        return " ".join(row.get("business_model") or [])
    return ""


def _company_blob(row: dict[str, Any]) -> str:
    return " ".join(
        [
            _field_text(row, "naics"),
            _field_text(row, "core_offerings"),
            _field_text(row, "target_markets"),
            _field_text(row, "business_model"),
            row.get("description") or "",
        ]
    ).lower()


def _finance_naics_bonus(queries: list[str], row: dict[str, Any]) -> float:
    blob = " ".join(queries).lower()
    if not any(term in blob for term in _FINANCE_QUERY_TERMS):
        return 0.0
    code = str(row.get("naics_code") or "")
    label = (row.get("naics_label") or "").lower()
    if code.startswith("522"):
        return 0.60
    if any(w in label for w in ("financial", "credit", "bank", "payment", "lending")):
        return 0.40
    return 0.0


def _specific_coverage(queries: list[str], row: dict[str, Any]) -> float:
    """
    Fraction of distinctive query tokens found on the company.
    0.0 → only generics (or nothing) matched; 1.0 → all specific tokens hit.
    """
    specific, _generic = _partition_query_tokens(queries)
    if not specific:
        # Query itself is generic (e.g. just "software") — fall back to any token hit
        return 0.5
    blob = _company_blob(row)
    hits = sum(1 for t in specific if _token_in_blob(t, blob))
    return hits / len(specific)


def _field_score(
    queries: list[str],
    row: dict[str, Any],
    field: str,
    *,
    specific_tokens: list[str],
) -> float:
    blob = _field_text(row, field)
    if not blob.strip():
        return 0.0

    primary = queries[0] if queries else ""
    primary_score = _phrase_strength(primary, blob)

    # Expansions: only count if they carry specific tokens (or reinforce primary)
    expansion = 0.0
    for q in queries[1:6]:
        q_specific = _specific_tokens(q)
        # Skip pure-generic expansion phrases ("saas", "enterprise software")
        if not q_specific and _tokens(q):
            continue
        strength = _phrase_strength(q, blob)
        # Soft-cap expansions so they cannot outrank a weak primary via generics
        expansion = max(expansion, strength * (1.0 if q_specific else 0.35))

    # Primary (usually the niche theme) dominates; expansions only refine
    if primary_score >= 0.15:
        score = 0.70 * primary_score + 0.30 * expansion
    elif expansion > 0:
        score = 0.45 * primary_score + 0.55 * expansion
    else:
        score = primary_score

    # Token bonus — weight specific hits much higher than generic hits
    blob_l = blob.lower()
    if specific_tokens:
        spec_hits = sum(1 for t in specific_tokens if _token_in_blob(t, blob_l))
        spec_frac = spec_hits / len(specific_tokens)
        score = max(score, spec_frac * 0.90)
        # Tiny generic bonus only when some specific signal already exists
        if spec_hits:
            generics = [t for t in _tokens(" ".join(queries[:6])) if _is_generic_token(t)]
            if generics:
                gen_hits = sum(1 for t in generics if _token_in_blob(t, blob_l))
                score = min(1.0, score + (gen_hits / len(generics)) * 0.08)
    else:
        all_tokens = list(dict.fromkeys(t for q in queries[:6] for t in _tokens(q)))
        if all_tokens:
            tok_hits = sum(1 for t in all_tokens if _token_in_blob(t, blob_l))
            score = max(score, tok_hits / len(all_tokens) * 0.40)

    if field == "business_model":
        primary_tokens = _tokens(primary)
        if primary_tokens and all(t in _WEAK_MODEL_TERMS for t in primary_tokens):
            score *= 0.40
        # Business-model-only generic hits are weak evidence
        if specific_tokens:
            if not any(_token_in_blob(t, blob_l) for t in specific_tokens):
                score *= 0.35
        # Explicit E-commerce model tag is strong merchant evidence for ecom queries.
        if _is_ecommerce_query(queries) and _is_ecommerce_platform_vendor(row):
            score *= 0.25
        elif _is_ecommerce_query(queries) and _is_ecommerce_merchant(row):
            models_n = _normalize_theme_text(blob_l)
            if "ecommerce" in models_n:
                score = max(score, 0.92)
            else:
                score = min(1.0, score * 1.15)

    if field == "naics":
        score = max(score, _finance_naics_bonus(queries, row))

    return min(score, 1.0)


def score_row(
    queries: list[str],
    row: dict[str, Any],
    *,
    exclusions: list[str] | None = None,
    embed_scores: dict[str, float] | None = None,
    uses_tool: str | None = None,
) -> tuple[float, dict[str, float]]:
    """Return (total rerank score, per-field breakdown).

    When ``embed_scores`` is provided (from EmbeddingIndex.field_embed_scores),
    blends lexical/specificity scoring with per-field embedding similarity so
    paraphrase matches (e.g. patient data ↔ health IT) can rank above pure
    keyword overlap.
    """
    specific, _generic = _partition_query_tokens(queries)
    breakdown: dict[str, float] = {}
    lexical_total = 0.0
    for field, weight in FIELD_WEIGHTS.items():
        fs = _field_score(queries, row, field, specific_tokens=specific)
        breakdown[field] = round(fs, 4)
        lexical_total += weight * fs

    # Gate: companies that only match generics cannot outrank specific matches
    coverage = _specific_coverage(queries, row)
    breakdown["specific_coverage"] = round(coverage, 4)
    if specific:
        if coverage <= 0.0:
            # Pure generic overlap — hard cap; embeddings may still rescue paraphrase hits
            lexical_total = min(lexical_total, 0.12) * 0.5
        elif coverage < 0.34:
            lexical_total *= 0.35 + 0.65 * coverage
        else:
            lexical_total *= 0.75 + 0.35 * min(coverage, 1.0)

    embed_total = float((embed_scores or {}).get("embed_total") or 0.0)
    if embed_scores:
        for k, v in embed_scores.items():
            breakdown[k] = v

    if embed_scores and embed_total > 0:
        # Embeddings carry meaning; lexical keeps keyword precision / generic gate
        if specific and coverage <= 0.0:
            # No specific token hit: trust embedding more (paraphrase case),
            # but require a solid embed score to stay in the list
            total = 0.25 * lexical_total + 0.75 * embed_total
            if embed_total < 0.32:
                total *= 0.55
        elif specific and coverage < 0.34:
            total = 0.40 * lexical_total + 0.60 * embed_total
        else:
            total = 0.45 * lexical_total + 0.55 * embed_total
    else:
        total = lexical_total

    # E-commerce / storefront-tool queries: prefer merchants, not platform vendors
    storefront_query = _is_ecommerce_query(queries)
    if uses_tool:
        from router.tool_profiles import is_storefront_tool

        storefront_query = storefront_query or is_storefront_tool(uses_tool)

    if storefront_query:
        if _is_ecommerce_platform_vendor(row):
            total *= 0.35
            breakdown["ecommerce_vendor_penalty"] = 0.35
        elif _is_ecommerce_merchant(row):
            total = min(1.0, total * 1.35)
            breakdown["ecommerce_merchant_boost"] = 1.35
            models = {
                _normalize_theme_text(m)
                for m in (row.get("business_model") or [])
                if isinstance(m, str)
            }
            if any("ecommerce" in m for m in models):
                total = min(1.0, total * 1.1)
                breakdown["ecommerce_model_boost"] = 1.1
        elif _ecommerce_adjacent_only(row):
            total *= 0.45
            breakdown["ecommerce_adjacent_penalty"] = 0.45
    elif uses_tool:
        # Non-storefront tools: demote builders of that tool class
        from router.tool_profiles import is_tool_class_vendor

        if is_tool_class_vendor(row, uses_tool):
            total *= 0.40
            breakdown["tool_vendor_penalty"] = 0.40

    # Soft penalty when exclusion/contrast terms dominate the company text
    if exclusions:
        blob = _company_blob(row)
        penalty = 0.0
        for term in exclusions:
            t = (term or "").strip().lower()
            if len(t) < 4:
                continue
            if t in blob or _phrase_strength(term, blob) >= 0.65:
                penalty = max(penalty, 0.25)
        total = max(0.0, total - penalty)

    return round(total, 4), breakdown


def rerank_candidates(
    queries: list[str],
    candidates: list[tuple[float, dict[str, Any]]],
    *,
    limit: int = 10,
    min_rerank_score: float = 0.15,
    exclusions: list[str] | None = None,
    embedding_index: Any | None = None,
    uses_tool: str | None = None,
) -> list[tuple[float, float, dict[str, Any], dict[str, float]]]:
    """
    Rerank retrieval candidates using business fields.

    Ordering prioritizes specific-theme overlap over generic tokens
    (software / AI / saas / platform, …). When ``embedding_index`` is set,
    also blends per-field embedding similarity for paraphrase matching.
    Retrieval score is metadata only. Candidates below min_rerank_score are dropped.

    Returns list of (rerank_score, retrieval_score, row, field_scores).
    """
    clean_queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    if not clean_queries or not candidates:
        out: list[tuple[float, float, dict[str, Any], dict[str, float]]] = []
        for ret_score, row in candidates[:limit]:
            out.append((0.0, ret_score, row, {}))
        return out

    rows = [row for _score, row in candidates]
    embed_list: list[dict[str, float]] = [{} for _ in rows]
    if embedding_index is not None:
        try:
            embed_list = embedding_index.field_embed_scores(clean_queries, rows)
        except Exception as exc:  # pragma: no cover - degrade to lexical-only
            print(f"reranker: field embeddings unavailable ({exc})", file=__import__("sys").stderr)
            embed_list = [{} for _ in rows]

    scored: list[tuple[float, float, dict[str, Any], dict[str, float]]] = []
    for (ret_score, row), emb in zip(candidates, embed_list):
        rerank_score, breakdown = score_row(
            clean_queries,
            row,
            exclusions=exclusions,
            embed_scores=emb or None,
            uses_tool=uses_tool,
        )
        if rerank_score < min_rerank_score:
            continue
        scored.append((rerank_score, ret_score, row, breakdown))

    # Prefer higher specific_coverage / embed_total when rerank scores are close
    scored.sort(
        key=lambda x: (
            x[0],
            float((x[3] or {}).get("embed_total") or 0.0),
            float((x[3] or {}).get("specific_coverage") or 0.0),
            x[1],
        ),
        reverse=True,
    )
    return scored[:limit]


# Public aliases for helpers shared with the evidence module.
tokens = _tokens
token_in_blob = _token_in_blob
phrase_strength = _phrase_strength
field_text = _field_text
