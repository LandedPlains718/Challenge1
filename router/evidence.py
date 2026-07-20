#!/usr/bin/env python3
"""Build grounded evidence objects for each matched company before LLM answering."""

from __future__ import annotations

from typing import Any

from router.query_intent import QueryIntent
from router.reranker import (
    FIELD_WEIGHTS,
    field_text,
    phrase_strength,
    token_in_blob,
    tokens,
)


def _format_revenue(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"${value:,.0f}"
    return "n/a"


def _confidence(rerank_score: float, supporting: dict[str, float]) -> str:
    strong_fields = sum(1 for v in supporting.values() if v >= 0.45)
    if rerank_score >= 0.40 and strong_fields >= 2:
        return "high"
    if rerank_score >= 0.22 or strong_fields >= 1:
        return "medium"
    return "low"


def _business_field_scores(field_scores: dict[str, float]) -> dict[str, float]:
    """Keep only real business-field scores (drop embed_*/boost metadata)."""
    return {
        k: float(v)
        for k, v in field_scores.items()
        if k in FIELD_WEIGHTS and isinstance(v, (int, float))
    }


def _matched_snippets(queries: list[str], blob: str, *, limit: int = 2) -> list[str]:
    if not blob.strip() or not queries:
        return []
    hits: list[str] = []
    blob_l = blob.lower()
    for q in queries:
        q = q.strip()
        if not q:
            continue
        if q.lower() in blob_l:
            # Pull a short surrounding fragment if the blob is a long description
            idx = blob_l.find(q.lower())
            start = max(0, idx - 40)
            end = min(len(blob), idx + len(q) + 40)
            frag = blob[start:end].strip()
            if start > 0:
                frag = "…" + frag
            if end < len(blob):
                frag = frag + "…"
            hits.append(frag if len(blob) > 120 else blob.strip()[:160])
        else:
            strength = phrase_strength(q, blob)
            if strength >= 0.65:
                hits.append(blob.strip()[:160])
        if len(hits) >= limit:
            break
    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for h in hits:
        key = h.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def _why_it_matches(
    queries: list[str],
    row: dict[str, Any],
    field_scores: dict[str, float],
    *,
    intent: QueryIntent | None = None,
) -> list[str]:
    reasons: list[str] = []
    field_labels = {
        "naics": "NAICS",
        "core_offerings": "core offerings",
        "target_markets": "target markets",
        "business_model": "business model",
    }

    # Merchant / uses_tool queries: surface E-commerce business model for storefront tools.
    from router.tool_profiles import is_storefront_tool

    storefrontish = bool(
        intent
        and (
            is_storefront_tool(intent.uses_tool)
            or "ecommerce"
            in (intent.primary_theme or "").lower().replace("-", "").replace(" ", "")
        )
    )
    if storefrontish:
        models = [m for m in (row.get("business_model") or []) if isinstance(m, str)]
        if any("e-commerce" in m.lower() or "ecommerce" in m.lower() for m in models):
            reasons.append(
                "business model lists E-commerce (online merchant / retailer operator)"
            )

    business_scores = _business_field_scores(field_scores)
    for field, score in sorted(business_scores.items(), key=lambda kv: -kv[1]):
        if score < 0.25:
            continue
        blob = field_text(row, field)
        snippets = _matched_snippets(queries, blob, limit=1)
        label = field_labels.get(field, field)
        if field == "business_model" and reasons and any("business model lists" in r for r in reasons):
            # Already covered the merchant model tag above
            continue
        if snippets:
            reasons.append(f"{label} match (score {score:.2f}): {snippets[0]}")
        else:
            reasons.append(f"{label} overlap (score {score:.2f})")
        if len(reasons) >= 3:
            break

    if intent and intent.secondary_theme:
        sec = intent.secondary_theme.lower()
        for field_name, blob in (
            ("target_markets", field_text(row, "target_markets")),
            ("core_offerings", field_text(row, "core_offerings")),
            ("naics", field_text(row, "naics")),
            ("description", row.get("description") or ""),
        ):
            if sec and sec in blob.lower():
                reasons.append(f"secondary theme '{intent.secondary_theme}' appears in {field_name}")
                break

    if intent and intent.uses_tool.strip() and reasons:
        from router.tool_profiles import operator_label_for

        tool = intent.uses_tool.strip()
        label = operator_label_for(tool, primary_theme=intent.primary_theme or "")
        reasons.append(
            f"dataset does not record {tool} usage; included as {label}"
        )

    if not reasons:
        name = row.get("operational_name") or "company"
        reasons.append(f"{name} was retrieved as a candidate but field overlap is weak")
    return reasons[:4]


def _weaknesses(
    queries: list[str],
    row: dict[str, Any],
    field_scores: dict[str, float],
    *,
    intent: QueryIntent | None = None,
    rerank_score: float = 0.0,
) -> list[str]:
    weaknesses: list[str] = []

    if rerank_score < 0.22:
        weaknesses.append("low overall field-match confidence")

    empty = [f for f, w in FIELD_WEIGHTS.items() if not field_text(row, f).strip()]
    if empty:
        weaknesses.append(f"missing fields: {', '.join(empty)}")

    weak_fields = [f for f, s in field_scores.items() if s < 0.15]
    if len(weak_fields) >= 3:
        weaknesses.append("little overlap across business fields")

    # Exclusion / contrast hits are weaknesses (or soft negatives)
    blocked: list[str] = []
    if intent:
        if intent.contrast.strip():
            blocked.append(intent.contrast.strip())
        blocked.extend(intent.exclusions)

    blob_all = " ".join(
        [
            row.get("description") or "",
            field_text(row, "naics"),
            field_text(row, "core_offerings"),
            field_text(row, "target_markets"),
            field_text(row, "business_model"),
        ]
    ).lower()

    for term in blocked:
        t = term.lower().strip()
        if len(t) < 4:
            continue
        term_tokens = tokens(t)
        if t in blob_all or (
            term_tokens and all(token_in_blob(tok, blob_all) for tok in term_tokens)
        ):
            weaknesses.append(f"may overlap with contrast/exclusion '{term}'")

    # Primary theme absent from thematic fields
    if queries:
        primary = queries[0].lower()
        thematic = " ".join(
            [
                field_text(row, "naics"),
                field_text(row, "core_offerings"),
                field_text(row, "target_markets"),
            ]
        ).lower()
        if primary and primary not in thematic and phrase_strength(queries[0], thematic) < 0.35:
            weaknesses.append(f"primary theme '{queries[0]}' not explicit in NAICS/offerings/markets")

    return weaknesses[:4]


def build_evidence_object(
    record: dict[str, Any],
    *,
    queries: list[str],
    intent: QueryIntent | None = None,
) -> dict[str, Any]:
    """
    Build a grounded evidence object for one match record.

    Expected record shape: {"row": {...}, "rerank_score": float, "field_scores": {...}, ...}
    """
    row = record.get("row", record)
    field_scores = dict(record.get("field_scores") or {})
    if not field_scores and queries:
        # Structured route / no rerank — score fields on the fly
        from router.reranker import score_row

        _, field_scores = score_row(queries, row)

    rerank_score = float(record.get("rerank_score") or record.get("score") or 0.0)
    business_scores = _business_field_scores(field_scores)
    supporting = {
        k: round(float(v), 4) for k, v in business_scores.items() if float(v) >= 0.20
    }

    company = row.get("operational_name") or "(unnamed)"
    evidence = {
        "company": company,
        "website": row.get("website"),
        "country": (row.get("country_code") or "?").upper(),
        "revenue": _format_revenue(row.get("revenue")),
        "employees": row.get("employee_count"),
        "naics": {
            "code": row.get("naics_code"),
            "label": row.get("naics_label"),
        },
        "why_it_matches": _why_it_matches(queries, row, field_scores, intent=intent),
        "supporting_fields": supporting,
        "confidence": _confidence(rerank_score, supporting),
        "weaknesses": _weaknesses(
            queries, row, field_scores, intent=intent, rerank_score=rerank_score
        ),
        "rerank_score": round(rerank_score, 4),
    }
    return evidence


def build_evidence_list(
    records: list[dict[str, Any]],
    *,
    queries: list[str],
    intent: QueryIntent | None = None,
) -> list[dict[str, Any]]:
    return [
        build_evidence_object(rec, queries=queries, intent=intent) for rec in records
    ]
