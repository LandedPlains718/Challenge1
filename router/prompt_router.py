#!/usr/bin/env python3
"""
Route a natural-language prompt over data/companies.jsonl into:
  - structured: filter/sort/aggregate on typed fields
  - context:    embedding similarity over text enrichment fields
  - hybrid:     structured filter, then embedding rank on the subset
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from router.context_hints import CONTEXT_HINTS
from router.embeddings import EmbeddingIndex, context_blob
from router.llm_answerer import answer_with_llm
from router.llm_router import classify_with_llm
from router.paths import DEFAULT_COMPANIES_PATH
from router.query_intent import QueryIntent, decompose_intent_rules, intent_from_llm_payload
from router.reranker import rerank_candidates

RERANK_CANDIDATE_MULTIPLIER = 5
RERANK_MAX_CANDIDATES = 50
RETRIEVAL_CANDIDATE_MIN_SCORE = 0.0
MIN_RERANK_SCORE = 0.15

# --- schema: what each path can use ---

STRUCTURED_HINTS = {
    "country",
    "country_code",
    "region",
    "town",
    "city",
    "revenue",
    "employee",
    "employees",
    "headcount",
    "founded",
    "year",
    "public",
    "private",
    "naics",
    "website",
    "domain",
    "top",
    "largest",
    "smallest",
    "count",
    "how many",
    "greater",
    "less",
    "more than",
    "less than",
    "over",
    "under",
    "between",
}

COUNTRY_ALIASES = {
    "romania": "ro",
    "romanian": "ro",
    "usa": "us",
    "us": "us",
    "u.s.": "us",
    "u.s.a.": "us",
    "united states": "us",
    "america": "us",
    "uk": "gb",
    "united kingdom": "gb",
    "britain": "gb",
    "british": "gb",
    "switzerland": "ch",
    "swiss": "ch",
    "france": "fr",
    "french": "fr",
    "china": "cn",
    "chinese": "cn",
    "sweden": "se",
    "swedish": "se",
    "norway": "no",
    "norwegian": "no",
    "denmark": "dk",
    "danish": "dk",
    "spain": "es",
    "spanish": "es",
    "germany": "de",
    "german": "de",
    "india": "in",
    "indian": "in",
    "netherlands": "nl",
    "dutch": "nl",
    "australia": "au",
    "australian": "au",
    "japan": "jp",
    "japanese": "jp",
    "canada": "ca",
    "canadian": "ca",
    "italy": "it",
    "italian": "it",
    "brazil": "br",
    "brazilian": "br",
    "israel": "il",
    "israeli": "il",
}

# Continent / multi-country regions → ISO country codes present in typical firmographic data
REGION_ALIASES: dict[str, frozenset[str]] = {
    "europe": frozenset(
        {
            "al", "ad", "at", "ba", "be", "bg", "by", "ch", "cy", "cz", "de", "dk",
            "ee", "es", "fi", "fr", "gb", "gr", "hr", "hu", "ie", "is", "it", "li",
            "lt", "lu", "lv", "mc", "md", "me", "mk", "mt", "nl", "no", "pl", "pt",
            "ro", "rs", "ru", "se", "si", "sk", "sm", "ua", "va", "xk",
        }
    ),
    "european": frozenset(),  # filled below
    "eu": frozenset(
        {
            "at", "be", "bg", "hr", "cy", "cz", "dk", "ee", "fi", "fr", "de", "gr",
            "hu", "ie", "it", "lv", "lt", "lu", "mt", "nl", "pl", "pt", "ro", "sk",
            "si", "es", "se",
        }
    ),
    "asia": frozenset(
        {"cn", "in", "jp", "kr", "sg", "tw", "hk", "id", "vn", "th", "my", "ph", "ae", "sa", "il", "tr", "kw"}
    ),
    "asian": frozenset(),
    "north america": frozenset({"us", "ca", "mx"}),
    "north american": frozenset(),
    "scandinavia": frozenset({"dk", "se", "no", "fi", "is"}),
    "scandinavian": frozenset(),
    "nordic": frozenset({"dk", "se", "no", "fi", "is"}),
    "nordics": frozenset(),
    "middle east": frozenset({"ae", "sa", "il", "tr", "kw", "qa", "bh", "om", "jo", "eg"}),
}
REGION_ALIASES["european"] = REGION_ALIASES["europe"]
REGION_ALIASES["asian"] = REGION_ALIASES["asia"]
REGION_ALIASES["north american"] = REGION_ALIASES["north america"]
REGION_ALIASES["scandinavian"] = REGION_ALIASES["scandinavia"]
REGION_ALIASES["nordics"] = REGION_ALIASES["nordic"]
REGION_ALIASES["middle-east"] = REGION_ALIASES["middle east"]

# City / town names → matched against the address.town field
CITY_ALIASES = {
    "london": "London",
    "paris": "Paris",
    "berlin": "Berlin",
    "bucharest": "Bucharest",
    "new york": "New York",
    "san francisco": "San Francisco",
}


def parse_maybe(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text.lower() == "null":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return value
    return parsed if isinstance(parsed, (dict, list)) else value


def load_companies(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["address"] = parse_maybe(row.get("address"))
            row["primary_naics"] = parse_maybe(row.get("primary_naics"))
            row["secondary_naics"] = parse_maybe(row.get("secondary_naics"))
            rows.append(row)
    return rows


def flatten(row: dict) -> dict:
    addr = row.get("address") if isinstance(row.get("address"), dict) else {}
    naics = row.get("primary_naics") if isinstance(row.get("primary_naics"), dict) else {}
    return {
        "website": row.get("website"),
        "operational_name": row.get("operational_name"),
        "year_founded": row.get("year_founded"),
        "employee_count": row.get("employee_count"),
        "revenue": row.get("revenue"),
        "is_public": row.get("is_public"),
        "country_code": (addr.get("country_code") or "").lower() or None,
        "region_name": addr.get("region_name"),
        "town": addr.get("town"),
        "naics_code": naics.get("code"),
        "naics_label": naics.get("label"),
        "description": row.get("description") or "",
        "business_model": row.get("business_model") or [],
        "target_markets": row.get("target_markets") or [],
        "core_offerings": row.get("core_offerings") or [],
    }


@dataclass
class RouteDecision:
    route: str  # structured | context | hybrid
    reason: str
    semantic_query: str | None = None
    embedding_queries: list[str] | None = None
    intent: QueryIntent | None = None
    classifier: str = "rules"

    def embedding_texts(self) -> list[str]:
        """Phrases to embed for context/hybrid search."""
        if self.intent and not self.intent.is_empty():
            from_intent = self.intent.embedding_queries(self.embedding_queries)
            if from_intent:
                return from_intent
        if self.embedding_queries:
            return list(self.embedding_queries)
        if self.semantic_query and self.semantic_query.strip():
            return [self.semantic_query.strip()]
        return []


# Words that appear in enrichment hints but are not industry themes by themselves
_STRUCTURED_THEME_BLOCKLIST = frozenset(
    {
        "public",
        "private",
        "retail",
        "enterprise",
        "commercial",
        "industrial",
        "consumer",
        "wholesale",
        "manufacturing",
        "services",
        "service",
        "sector",
        "industry",
        "industries",
        "vertical",
        "government",
        "global",
        "international",
    }
)


def _has_industry_theme(text: str) -> bool:
    """True when prompt names a topic beyond pure structured filters."""
    for h in CONTEXT_HINTS:
        if h in _STRUCTURED_THEME_BLOCKLIST:
            continue
        if " " in h or len(h) >= 6:
            if h in text:
                return True
        elif re.search(rf"\b{re.escape(h)}\b", text):
            return True
    # "<industry> companies/firms/manufacturers ..." — skip filter boilerplate
    return bool(
        re.search(
            r"\b(?!list|show|find|get|give|name|all|public|private|largest|smallest|how|many|"
            r"number|count|top|based|firms|companies)"
            r"[a-z][a-z\-]{2,}"
            r"(?:\s+(?!public|private|in|from|with|for|by|the|and|or|of|a|an|under|over|more|less)"
            r"[a-z][a-z\-]{2,})?\s+"
            r"(?:companies|company|firms|businesses|manufacturers|manufacturer|"
            r"suppliers|supplier|vendors|vendor|startups|startup|"
            r"banks|bank|platforms|platform|providers|provider)\b",
            text,
        )
    )


def extract_semantic_query(prompt: str) -> str:
    """Remove structured-filter language so hybrid ranking embeds only the theme."""
    text = prompt

    # geo phrases tied to known countries / regions only (avoid eating the theme)
    country_alt = "|".join(
        re.escape(name) for name in sorted(COUNTRY_ALIASES, key=len, reverse=True)
    )
    region_alt = "|".join(
        re.escape(name) for name in sorted(REGION_ALIASES, key=len, reverse=True)
    )
    text = re.sub(
        rf"\b(?:in|from|based in|located in|headquartered in)\s+(?:the\s+)?(?:{country_alt}|{region_alt})\b",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(rf"\b(?:{country_alt})\b", " ", text, flags=re.I)
    text = re.sub(rf"\b(?:{region_alt})\b", " ", text, flags=re.I)
    text = re.sub(r"\bcountry(?:\s*code)?\s*[:=]?\s*[a-z]{2}\b", " ", text, flags=re.I)

    # public/private listing (keep "private equity")
    text = re.sub(r"\b(?:not\s+)?public(?:ly\s+held)?\b", " ", text, flags=re.I)
    text = re.sub(
        r"\b(?:not\s+)?private(?!\s+equity)(?:ly\s+held)?\b",
        " ",
        text,
        flags=re.I,
    )

    # aggregates / size filters (including currency amounts)
    text = re.sub(r"\btop\s+\d+\b", " ", text, flags=re.I)
    text = re.sub(r"\b(?:largest|smallest|how many|count of|number of)\b", " ", text, flags=re.I)
    text = re.sub(
        r"\b(?:with|having|that have|that has)?\s*"
        r"(?:more than|less than|fewer than|greater than|at least|at most|over|under|below|above)\s+"
        r"\$?\s*\d[\d,.]*(?:\s*(?:k|m|million|b|billion))?\s*"
        r"(?:revenue|employees?|headcount|people|staff)?\b",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\b(?:with|having)?\s*revenue\b"
        r"(?:\s*(?:of|over|under|above|below|greater than|less than|more than|at least|at most|>|<|>=|<=))?"
        r"\s*\$?\s*\d[\d,.]*(?:\s*(?:k|m|million|b|billion))?",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\b(?:revenue|employee(?:s)?|headcount|founded|year)\b"
        r"(?:\s*(?:[><=]|of|above|below|over|under|after|before|since|greater than|less than|more than))?"
        r"\s*\$?\s*\d[\d,.]*(?:\s*(?:k|m|million|b|billion))?",
        " ",
        text,
        flags=re.I,
    )
    # orphaned filter fragments (e.g. shell ate "$50" → "revenue over million")
    text = re.sub(
        r"\b(?:with|having)?\s*revenue\b"
        r"(?:\s*(?:of|over|under|above|below|greater than|less than|more than))?"
        r"(?:\s*\$)?(?:\s*(?:million|billion|thousand))?",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\$?\s*\d[\d,.]*(?:\s*(?:k|m|million|b|billion))\b", " ", text, flags=re.I)
    text = re.sub(r"\$\s*", " ", text)

    # generic entity words that dilute the theme once filters are applied
    text = re.sub(
        r"\b(?:companies|company|firms|firm|businesses|business|organizations|organisation|"
        r"manufacturers|manufacturer|suppliers|supplier|vendors|vendor|startups|startup|"
        r"providers|provider|operators|operator|contractors|contractor)\b",
        " ",
        text,
        flags=re.I,
    )

    # leftover glue / question boilerplate
    text = re.sub(
        r"\b(?:in|from|with|and|or|the|a|an|of|for|to|by|who|what|which|that|could|"
        r"provides|providing|offering|offer|focused|specializing|specialising|"
        r"building|developing|using|similar|related|equipment|devices|products|"
        r"materials|solutions|services|backed)\b",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s+", " ", text).strip(" ,.-?")

    # common user spelling: "logistic" → "logistics"
    text = re.sub(r"\blogistic\b", "logistics", text, flags=re.I)

    cleaned = text.strip()
    # if stripping left only noise, fall back to original prompt (context path)
    if not cleaned or len(cleaned.split()) > 12:
        return prompt.strip()
    return cleaned


def _has_structured_cue(text: str) -> bool:
    """Word-boundary structured cues (avoid 'under' matching 'underwater')."""
    for h in STRUCTURED_HINTS:
        if " " in h:
            if h in text:
                return True
        elif re.search(rf"\b{re.escape(h)}\b", text):
            return True
    return False


def classify_prompt_rules(prompt: str) -> RouteDecision:
    """Rule-based router (fallback when LLM is unavailable)."""
    text = prompt.lower()
    structured_hits = _has_structured_cue(text)
    context_hits = _has_industry_theme(text)

    has_number_filter = bool(
        re.search(r"(revenue|employee|employees|founded|year).{0,20}\d", text)
        or re.search(r"\d.{0,20}(revenue|employee|employees|million|billion)", text)
    )
    has_geo = any(re.search(rf"\b{re.escape(k)}\b", text) for k in COUNTRY_ALIASES) or any(
        re.search(rf"\b{re.escape(k)}\b", text) for k in REGION_ALIASES
    )
    has_aggregate = any(k in text for k in ("top ", "largest", "smallest", "how many", "count"))

    if (has_geo or has_number_filter or has_aggregate or structured_hits) and context_hits:
        decision = RouteDecision(
            "hybrid", "structured filters + semantic theme", classifier="rules"
        )
    elif has_number_filter or has_aggregate:
        decision = RouteDecision(
            "structured", "maps to typed fields / aggregates", classifier="rules"
        )
    elif has_geo and not context_hits:
        decision = RouteDecision(
            "structured", "geo/structured cues without semantic theme", classifier="rules"
        )
    elif context_hits:
        decision = RouteDecision("context", "needs text / meaning fields", classifier="rules")
    elif structured_hits:
        decision = RouteDecision(
            "structured", "structured cues without semantic theme", classifier="rules"
        )
    else:
        decision = RouteDecision(
            "context", "ambiguous; default to text search", classifier="rules"
        )

    return _reconcile_route_with_filters(prompt, decision)


def classify_prompt(
    prompt: str,
    *,
    use_llm: bool = True,
    llm_model: str | None = None,
) -> RouteDecision:
    """Prefer LLM classifier; fall back to rules. Validate hybrid/structured vs real filters."""
    if use_llm:
        llm = classify_with_llm(prompt, model=llm_model)
        if llm is not None:
            intent = intent_from_llm_payload(
                {
                    "primary_theme": llm.primary_theme,
                    "secondary_theme": llm.secondary_theme,
                    "contrast": llm.contrast,
                    "exclusions": list(llm.exclusions or []),
                },
                semantic_query=llm.semantic_query or "",
            )
            decision = RouteDecision(
                route=llm.route,
                reason=llm.reason,
                semantic_query=intent.primary_theme or llm.semantic_query or None,
                embedding_queries=list(llm.embedding_queries or []),
                intent=intent,
                classifier="llm",
            )
            return _reconcile_route_with_filters(prompt, decision)
    return classify_prompt_rules(prompt)


def _usable_structured_filters(filters: dict[str, Any]) -> dict[str, Any]:
    """Filters that actually narrow the dataset (ignore sort/limit-only)."""
    return {
        k: v
        for k, v in filters.items()
        if k
        in {
            "country_code",
            "country_codes",
            "region",
            "town",
            "is_public",
            "revenue_min",
            "revenue_max",
            "employee_min",
            "employee_max",
            "year_min",
            "year_max",
        }
    }


def _looks_like_semantic_theme(prompt: str, semantic_query: str | None) -> bool:
    """True when there is a real industry/product theme, not filter boilerplate."""
    sem = (semantic_query or "").strip().lower()
    prompt_l = prompt.lower()

    filter_noise = {
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "by", "as",
        "at", "is", "are", "be", "public", "private", "companies", "company",
        "firms", "firm", "employees", "employee", "headcount", "people", "staff",
        "revenue", "founded", "year", "million", "billion", "thousand", "dataset",
        "list", "show", "find", "give", "largest", "smallest", "top", "how",
        "many", "count", "number", "based", "under", "over", "more", "less",
        "than", "with", "from", "after", "before", "fewer", "only", "just",
        "about", "into", "their", "them", "this", "that", "those", "these",
    }
    for name in list(COUNTRY_ALIASES) + list(REGION_ALIASES):
        filter_noise.update(re.findall(r"[a-z0-9]+", name.lower()))

    if sem:
        # dumped full prompt / long clause is not a clean theme
        if len(sem.split()) > 5:
            return False
        tokens = set(re.findall(r"[a-z0-9]+", sem))
        thematic = {t for t in tokens - filter_noise if len(t) > 2 and not t.isdigit()}
        if thematic:
            return True

    return _has_industry_theme(prompt_l)


def _has_aggregate_intent(prompt: str) -> bool:
    text = prompt.lower()
    return bool(
        re.search(r"\btop\s+\d+\b", text)
        or re.search(r"\b(largest|smallest|how many|count of|number of|ranked by)\b", text)
    )


def _reconcile_route_with_filters(prompt: str, decision: RouteDecision) -> RouteDecision:
    """
    Reconcile LLM route with filters we can actually parse.
    - filters + theme → hybrid
    - filters/aggregates + no theme → structured
    - no filters + theme → context
    """
    filters = _usable_structured_filters(extract_structured_filters(prompt))
    has_struct = bool(filters) or _has_aggregate_intent(prompt)

    # Prefer primary_theme from intent when available
    theme_hint = (
        (decision.intent.primary_theme if decision.intent else None)
        or decision.semantic_query
    )
    has_theme = _looks_like_semantic_theme(prompt, theme_hint)

    if has_theme and (theme_hint or "").strip() and len((theme_hint or "").split()) <= 6:
        sem: str | None = (theme_hint or "").strip()
    elif has_theme:
        sem = extract_semantic_query(prompt)
    else:
        sem = None

    # Build / refresh intent — prefer decomposing the prompt over noisy semantic dumps
    if has_theme:
        rules_intent = decompose_intent_rules(prompt, semantic_query=sem)
        if decision.intent and not decision.intent.is_empty():
            intent = decision.intent
            # Fill missing fields from rules
            if not intent.primary_theme:
                intent.primary_theme = rules_intent.primary_theme
            if not intent.secondary_theme:
                intent.secondary_theme = rules_intent.secondary_theme
            if not intent.uses_tool and rules_intent.uses_tool:
                intent.uses_tool = rules_intent.uses_tool
            # "X using Y" — prefer rules primary (LLM often invents "CRM SaaS" etc.)
            if rules_intent.uses_tool and rules_intent.primary_theme:
                intent.uses_tool = intent.uses_tool or rules_intent.uses_tool
                intent.primary_theme = rules_intent.primary_theme
                if not intent.secondary_theme and rules_intent.secondary_theme:
                    intent.secondary_theme = rules_intent.secondary_theme
            if not intent.contrast:
                intent.contrast = rules_intent.contrast
            if not intent.exclusions:
                intent.exclusions = list(rules_intent.exclusions)
            # If LLM dumped contrast into primary, prefer rules primary
            if (
                intent.contrast
                and intent.primary_theme
                and intent.contrast.lower() in intent.primary_theme.lower()
                and rules_intent.primary_theme
            ):
                intent.primary_theme = rules_intent.primary_theme
            # Prefer short cleaned primary over verbose LLM dumps
            if (
                rules_intent.primary_theme
                and intent.primary_theme
                and (
                    len(intent.primary_theme.split()) > 4
                    or re.search(r"\b(?:using|or similar)\b", intent.primary_theme, flags=re.I)
                )
            ):
                intent.primary_theme = rules_intent.primary_theme
                if not intent.secondary_theme and rules_intent.secondary_theme:
                    intent.secondary_theme = rules_intent.secondary_theme
            # Prefer specific "Y" over generic "software/platform for Y"
            _generic = {
                "software",
                "saas",
                "platform",
                "platforms",
                "solutions",
                "services",
                "tools",
                "systems",
            }
            if (
                rules_intent.primary_theme
                and intent.primary_theme.lower() in _generic
                and rules_intent.primary_theme.lower() not in _generic
            ):
                intent.primary_theme = rules_intent.primary_theme
                if rules_intent.secondary_theme:
                    intent.secondary_theme = rules_intent.secondary_theme
        else:
            intent = rules_intent
    else:
        intent = QueryIntent()

    if has_theme and intent.primary_theme:
        sem = intent.primary_theme

    emb = list(decision.embedding_queries or [])
    if has_theme:
        # Drop embedding phrases that are really contrast/exclusions
        emb = intent.embedding_queries(emb or ([sem] if sem else []))
        # Tool brands are almost never in company text — expand to operator signals
        if intent.uses_tool.strip():
            from router.tool_profiles import (
                operator_expansions_for,
                query_mentions_tool_brand,
            )

            emb = [q for q in emb if not query_mentions_tool_brand(q, intent.uses_tool)]
            seen_e = {q.lower() for q in emb}
            for phrase in operator_expansions_for(intent.uses_tool):
                if phrase.lower() in seen_e:
                    continue
                emb.append(phrase)
                seen_e.add(phrase.lower())
                if len(emb) >= 5:
                    break
    else:
        emb = []
        intent = QueryIntent()

    if has_struct and has_theme:
        final_sem = sem or intent.primary_theme or extract_semantic_query(prompt)
        if not intent.primary_theme:
            intent.primary_theme = final_sem or ""
        return RouteDecision(
            route="hybrid",
            reason=f"{decision.reason} (reconciled: filters + theme)",
            semantic_query=final_sem,
            embedding_queries=emb or ([final_sem] if final_sem else []),
            intent=intent,
            classifier=decision.classifier,
        )

    if has_struct and not has_theme:
        return RouteDecision(
            route="structured",
            reason=f"{decision.reason} (reconciled: filters only)",
            semantic_query=None,
            embedding_queries=[],
            intent=QueryIntent(),
            classifier=decision.classifier,
        )

    final_sem = sem or decision.semantic_query
    if final_sem and not intent.primary_theme:
        intent.primary_theme = final_sem
    return RouteDecision(
        route="context",
        reason=f"{decision.reason} (reconciled: theme only)" if has_theme else decision.reason,
        semantic_query=final_sem,
        embedding_queries=emb or ([final_sem] if final_sem else []),
        intent=intent,
        classifier=decision.classifier,
    )


def parse_amount(text: str) -> float | None:
    text = text.replace("$", "").replace("€", "").replace("£", "")
    m = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(k|m|million|b|billion)?",
        text,
        flags=re.I,
    )
    if not m:
        return None
    n = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    if unit in {"k"}:
        n *= 1_000
    elif unit in {"m", "million"}:
        n *= 1_000_000
    elif unit in {"b", "billion"}:
        n *= 1_000_000_000
    return n


def extract_structured_filters(prompt: str) -> dict[str, Any]:
    text = prompt.lower()
    filters: dict[str, Any] = {}

    # Regions first (multi-country), then single-country aliases, then cities/towns
    for name, codes in sorted(REGION_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(name)}\b", text):
            filters["country_codes"] = set(codes)
            filters["region"] = name
            break
    else:
        for name, code in COUNTRY_ALIASES.items():
            if re.search(rf"\b{re.escape(name)}\b", text):
                filters["country_code"] = code
                break
        m = re.search(r"\bcountry(?:\s*code)?\s*[:=]?\s*([a-z]{2})\b", text)
        if m:
            filters["country_code"] = m.group(1)

    for name, town in CITY_ALIASES.items():
        if re.search(rf"\b{re.escape(name)}\b", text):
            filters["town"] = town
            break

    if re.search(r"\bpublic\b", text) and not re.search(r"\bnot public\b", text):
        filters["is_public"] = True
    if re.search(r"\bprivate\b", text) and not re.search(r"\bprivate\s+equity\b", text):
        filters["is_public"] = False

    m = re.search(r"\btop\s+(\d+)\b", text)
    if m:
        filters["limit"] = int(m.group(1))
        if "employee" in text:
            filters["sort"] = ("employee_count", True)
        else:
            filters["sort"] = ("revenue", True)

    # largest / smallest (without an explicit top-N)
    if "sort" not in filters:
        if re.search(r"\b(largest|biggest|highest)\b", text):
            if "employee" in text:
                filters["sort"] = ("employee_count", True)
            else:
                filters["sort"] = ("revenue", True)
        elif re.search(r"\b(smallest|lowest|fewest)\b", text):
            if "revenue" in text:
                filters["sort"] = ("revenue", False)
            else:
                filters["sort"] = ("employee_count", False)

    # Strip "top N" so its number is not reused as a revenue/employee threshold
    text_for_amounts = re.sub(r"\btop\s+\d+\b", " ", text)

    # Year range: founded between 2010 and 2020
    m = re.search(
        r"(?:founded|year).{0,20}between\s+(\d{4})\s+and\s+(\d{4})",
        text,
    )
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        filters["year_min"] = min(y1, y2)
        filters["year_max"] = max(y1, y2)
    else:
        m = re.search(r"(?:founded|year)\s*(?:after|since|>|>=)\s*(\d{4})", text)
        if m:
            filters["year_min"] = int(m.group(1))
        m = re.search(r"(?:founded|year)\s*(?:before|<|<=)\s*(\d{4})", text)
        if m:
            filters["year_max"] = int(m.group(1))

    # Revenue range: between $10 million and $100 million
    m = re.search(
        r"revenue.{0,40}between\s+(\$?[\d.,]+\s*(?:k|m|million|b|billion)?)\s+and\s+"
        r"(\$?[\d.,]+\s*(?:k|m|million|b|billion)?)",
        text_for_amounts,
    )
    if not m:
        m = re.search(
            r"between\s+(\$?[\d.,]+\s*(?:k|m|million|b|billion)?)\s+and\s+"
            r"(\$?[\d.,]+\s*(?:k|m|million|b|billion)?).{0,20}revenue",
            text_for_amounts,
        )
    if m and "revenue" in text_for_amounts:
        a1, a2 = parse_amount(m.group(1)), parse_amount(m.group(2))
        if a1 is not None and a2 is not None:
            filters["revenue_min"] = min(a1, a2)
            filters["revenue_max"] = max(a1, a2)
    elif "revenue" in text_for_amounts:
        # Prefer number near the word "revenue"; do not fall back to unrelated digits
        after = text_for_amounts.split("revenue", 1)[1][:60]
        before = text_for_amounts.split("revenue", 1)[0][-60:]
        amount = parse_amount(after) or parse_amount(before)
        if amount is not None:
            # "over / more than / above / greater than" → min; "under / below / less than" → max
            if re.search(
                r"\b(less than|fewer than|under|below|at most|no more than|<)\b",
                text_for_amounts,
            ):
                filters["revenue_max"] = amount
            else:
                filters["revenue_min"] = amount

    if any(w in text_for_amounts for w in ("employee", "employees", "headcount")):
        m_emp = re.search(
            r"(employee|employees|headcount).{0,30}(\d+(?:[.,]\d+)?\s*(?:k|m|million|b|billion)?)",
            text_for_amounts,
            flags=re.I,
        )
        if not m_emp:
            m_emp = re.search(
                r"(\d+(?:[.,]\d+)?\s*(?:k|m|million|b|billion)?).{0,30}(employee|employees|headcount)",
                text_for_amounts,
                flags=re.I,
            )
            amount = parse_amount(m_emp.group(1)) if m_emp else None
        else:
            amount = parse_amount(m_emp.group(2))
        if amount is not None and amount < 10_000_000:
            if re.search(
                r"\b(less than|fewer than|under|below|at most|no more than|<)\b",
                text_for_amounts,
            ):
                filters["employee_max"] = amount
            else:
                filters["employee_min"] = amount

    return filters


def serialize_filters(filters: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in filters.items():
        if key == "country_codes":
            out[key] = sorted(value)
        elif key == "sort":
            out[key] = list(value)
        else:
            out[key] = value
    return out


def apply_structured(rows: list[dict], filters: dict[str, Any]) -> list[dict]:
    out = rows

    def keep(pred: Callable[[dict], bool]) -> None:
        nonlocal out
        out = [r for r in out if pred(r)]

    if "country_code" in filters:
        code = filters["country_code"]
        keep(lambda r: (r.get("country_code") or "") == code)
    if "country_codes" in filters:
        codes = filters["country_codes"]
        keep(lambda r: (r.get("country_code") or "") in codes)
    if "town" in filters:
        town = filters["town"].lower()
        keep(lambda r: (r.get("town") or "").lower() == town)
    if "is_public" in filters:
        val = filters["is_public"]
        keep(lambda r: r.get("is_public") is val)
    if "revenue_min" in filters:
        vmin = filters["revenue_min"]
        keep(lambda r: isinstance(r.get("revenue"), (int, float)) and r["revenue"] >= vmin)
    if "revenue_max" in filters:
        vmax = filters["revenue_max"]
        keep(lambda r: isinstance(r.get("revenue"), (int, float)) and r["revenue"] <= vmax)
    if "employee_min" in filters:
        vmin = filters["employee_min"]
        keep(
            lambda r: isinstance(r.get("employee_count"), (int, float))
            and r["employee_count"] >= vmin
        )
    if "employee_max" in filters:
        vmax = filters["employee_max"]
        keep(
            lambda r: isinstance(r.get("employee_count"), (int, float))
            and r["employee_count"] <= vmax
        )
    if "year_min" in filters:
        ymin = filters["year_min"]
        keep(
            lambda r: isinstance(r.get("year_founded"), (int, float)) and r["year_founded"] >= ymin
        )
    if "year_max" in filters:
        ymax = filters["year_max"]
        keep(
            lambda r: isinstance(r.get("year_founded"), (int, float)) and r["year_founded"] <= ymax
        )

    sort = filters.get("sort")
    if sort:
        key, reverse = sort
        out = sorted(
            [r for r in out if isinstance(r.get(key), (int, float))],
            key=lambda r: r[key],
            reverse=reverse,
        )

    limit = filters.get("limit")
    if limit:
        out = out[:limit]

    return out


def tokenize(text: str) -> set[str]:
    stop = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "in",
        "on",
        "for",
        "to",
        "with",
        "who",
        "what",
        "that",
        "companies",
        "company",
        "find",
        "show",
        "list",
        "me",
        "about",
        "like",
        "similar",
    }
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in stop}


def apply_context_keyword(rows: list[dict], prompt: str, limit: int = 10) -> list[tuple[float, dict]]:
    """Fallback when sentence-transformers is unavailable."""
    tokens = tokenize(prompt)
    if not tokens:
        return [(0.0, r) for r in rows[:limit]]

    scored: list[tuple[float, dict]] = []
    for row in rows:
        blob = context_blob(row).lower()
        score = sum(1.0 for t in tokens if t in blob)
        for t in tokens:
            if len(t) >= 5 and t in blob:
                score += 0.25
        if score > 0:
            scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:limit]


def apply_context_embeddings(
    index: EmbeddingIndex,
    rows: list[dict],
    prompt: str | list[str],
    limit: int = 10,
    subset: list[dict] | None = None,
    min_score: float | None = None,
    recall_mode: bool = False,
) -> list[tuple[float, dict]]:
    kwargs: dict[str, Any] = {"limit": limit, "recall_mode": recall_mode}
    if min_score is not None:
        kwargs["min_score"] = min_score
    if subset is None:
        return index.search(prompt, **kwargs)

    # Map subset rows back to positions in the full indexed list (by identity)
    id_to_idx = {id(r): i for i, r in enumerate(rows)}
    indices = [id_to_idx[id(r)] for r in subset if id(r) in id_to_idx]
    return index.search(prompt, indices=indices, **kwargs)


def _candidate_limit(limit: int) -> int:
    return min(max(limit * RERANK_CANDIDATE_MULTIPLIER, limit), RERANK_MAX_CANDIDATES)


def _rerank_scored_rows(
    scored_rows: list[tuple[float, dict]],
    queries: list[str],
    *,
    limit: int,
    intent: QueryIntent | None = None,
    embedding_index: EmbeddingIndex | None = None,
) -> tuple[list[tuple[float, dict]], list[dict[str, Any]]]:
    """Rerank retrieval hits; return updated rows and per-row rerank metadata."""
    exclusions: list[str] = []
    if intent:
        if intent.contrast.strip():
            exclusions.append(intent.contrast.strip())
        exclusions.extend(intent.exclusions)

    # Prefer niche primary; drop pure-generic expansion phrases for rerank scoring
    from router.reranker import _specific_tokens, _tokens

    ordered: list[str] = []
    seen: set[str] = set()

    def _add(q: str) -> None:
        q = (q or "").strip()
        if not q:
            return
        key = q.lower()
        if key in seen:
            return
        seen.add(key)
        ordered.append(q)

    if intent and intent.primary_theme.strip():
        _add(intent.primary_theme.strip())
    for q in queries:
        # Skip expansions that are only generic tokens (saas, software, ai, …)
        spec = _specific_tokens(q)
        if not spec and _tokens(q):
            continue
        _add(q)
    if intent and intent.secondary_theme.strip():
        sec = intent.secondary_theme.strip()
        if _specific_tokens(sec):
            _add(sec)
    # "using <Tool>" → operator expansions for that tool class (not the vendor brand)
    if intent and intent.uses_tool.strip():
        from router.tool_profiles import (
            operator_expansions_for,
            query_mentions_tool_brand,
        )

        # Drop the bare brand from rerank queries — it rarely appears in company text
        ordered = [q for q in ordered if not query_mentions_tool_brand(q, intent.uses_tool)]
        for phrase in operator_expansions_for(intent.uses_tool):
            _add(phrase)
    rerank_queries = ordered or list(queries)

    reranked = rerank_candidates(
        rerank_queries,
        scored_rows,
        limit=limit,
        min_rerank_score=MIN_RERANK_SCORE,
        exclusions=exclusions or None,
        embedding_index=embedding_index,
        uses_tool=(intent.uses_tool.strip() if intent else "") or None,
    )
    out_rows: list[tuple[float, dict]] = []
    meta: list[dict[str, Any]] = []
    for rerank_score, ret_score, row, breakdown in reranked:
        out_rows.append((rerank_score, row))
        meta.append(
            {
                "rerank_score": rerank_score,
                "retrieval_score": round(ret_score, 4),
                "field_scores": breakdown,
            }
        )
    return out_rows, meta


def summarize_row(row: dict) -> str:
    name = row.get("operational_name") or "(unnamed)"
    country = row.get("country_code") or "?"
    rev = row.get("revenue")
    emp = row.get("employee_count")
    rev_s = f"${rev:,.0f}" if isinstance(rev, (int, float)) else "n/a"
    emp_s = f"{emp:,.0f}" if isinstance(emp, (int, float)) else "n/a"
    return f"{name} [{country}] revenue={rev_s} employees={emp_s} web={row.get('website')}"


def run_router(
    prompt: str,
    companies_path: Path,
    limit: int = 10,
    *,
    use_embeddings: bool = True,
    rebuild_index: bool = False,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    use_llm: bool = True,
    llm_model: str | None = None,
    use_answer: bool = True,
) -> dict[str, Any]:
    raw = load_companies(companies_path)
    rows = [flatten(r) for r in raw]
    decision = classify_prompt(prompt, use_llm=use_llm, llm_model=llm_model)

    result: dict[str, Any] = {
        "prompt": prompt,
        "route": decision.route,
        "reason": decision.reason,
        "classifier": decision.classifier,
        "matches": [],
        "match_records": [],
        "retriever": "none",
    }

    index: EmbeddingIndex | None = None
    if use_embeddings and decision.route in {"context", "hybrid"}:
        try:
            index = EmbeddingIndex(rows, model_name=model_name, rebuild=rebuild_index)
            result["retriever"] = "embeddings"
        except ImportError:
            result["retriever"] = "keyword (sentence-transformers not installed)"
            use_embeddings = False

    def score_context(
        subset: list[dict] | None = None,
        query: str | list[str] | None = None,
        *,
        candidate_limit: int | None = None,
    ) -> list[tuple[float, dict]]:
        q: str | list[str] = query if query is not None else prompt
        fetch_limit = candidate_limit if candidate_limit is not None else limit
        use_recall = candidate_limit is not None and candidate_limit > limit
        candidate_min_score = RETRIEVAL_CANDIDATE_MIN_SCORE if use_recall else None
        if use_embeddings and index is not None:
            return apply_context_embeddings(
                index,
                rows,
                q,
                limit=fetch_limit,
                subset=subset,
                min_score=candidate_min_score,
                recall_mode=use_recall,
            )
        target = subset if subset is not None else rows
        kw = " ".join(q) if isinstance(q, list) else q
        return apply_context_keyword(target, kw, limit=fetch_limit)

    scored_rows: list[tuple[float, dict]] = []
    matched_rows: list[dict] = []
    match_count: int | None = None
    answer_filters: dict[str, Any] | None = None
    rerank_meta: list[dict[str, Any]] = []
    rerank_queries: list[str] = []

    if decision.route == "structured":
        filters = extract_structured_filters(prompt)
        result["filters"] = serialize_filters(filters)
        answer_filters = result["filters"]
        matched = apply_structured(rows, filters)
        match_count = len(
            apply_structured(rows, {k: v for k, v in filters.items() if k != "limit"})
        )
        if not filters.get("limit"):
            matched = matched[:limit]
        matched_rows = matched[:limit]
        result["match_count"] = match_count
        result["matches"] = [summarize_row(r) for r in matched_rows]

    elif decision.route == "context":
        emb = decision.embedding_texts() or (
            [decision.semantic_query] if decision.semantic_query else [prompt]
        )
        result["semantic_query"] = decision.semantic_query or emb[0]
        result["embedding_queries"] = emb
        if decision.intent and not decision.intent.is_empty():
            result["intent"] = decision.intent.to_dict()
        rerank_queries = emb
        scored_rows = score_context(query=emb, candidate_limit=_candidate_limit(limit))
        scored_rows, rerank_meta = _rerank_scored_rows(
            scored_rows,
            emb,
            limit=limit,
            intent=decision.intent,
            embedding_index=index,
        )
        result["matches"] = [
            f"rerank={meta['rerank_score']:.3f} | ret={meta['retrieval_score']:.3f} | {summarize_row(r)}"
            for (_score, r), meta in zip(scored_rows, rerank_meta)
        ]

    else:  # hybrid
        filters = extract_structured_filters(prompt)
        pre = {k: v for k, v in filters.items() if k not in {"sort", "limit"}}
        semantic_query = decision.semantic_query or extract_semantic_query(prompt)
        emb = decision.embedding_texts() or ([semantic_query] if semantic_query else [])
        result["filters"] = serialize_filters(pre)
        answer_filters = result["filters"]
        result["semantic_query"] = semantic_query
        result["embedding_queries"] = emb
        if decision.intent and not decision.intent.is_empty():
            result["intent"] = decision.intent.to_dict()
        rerank_queries = emb or ([semantic_query] if semantic_query else [])
        subset = apply_structured(rows, pre) if pre else rows
        scored_rows = score_context(
            subset,
            query=emb or semantic_query,
            candidate_limit=_candidate_limit(limit),
        )
        if rerank_queries:
            scored_rows, rerank_meta = _rerank_scored_rows(
                scored_rows,
                rerank_queries,
                limit=limit,
                intent=decision.intent,
                embedding_index=index,
            )
        result["subset_size"] = len(subset)
        if rerank_meta:
            result["matches"] = [
                f"rerank={meta['rerank_score']:.3f} | ret={meta['retrieval_score']:.3f} | {summarize_row(r)}"
                for (_score, r), meta in zip(scored_rows, rerank_meta)
            ]
        else:
            result["matches"] = [f"{score:.3f} | {summarize_row(r)}" for score, r in scored_rows]

    if rerank_meta:
        result["rerank_applied"] = True
        result["rerank_queries"] = rerank_queries

    if scored_rows:
        if rerank_meta:
            result["match_records"] = [
                {
                    "rerank_score": meta["rerank_score"],
                    "retrieval_score": meta["retrieval_score"],
                    "field_scores": meta["field_scores"],
                    "row": row,
                }
                for (_score, row), meta in zip(scored_rows, rerank_meta)
            ]
        else:
            result["match_records"] = [
                {"score": score, "row": row} for score, row in scored_rows
            ]
    else:
        result["match_records"] = [{"row": row} for row in matched_rows]

    if use_answer and use_llm:
        answer_records = result["match_records"]
        answer_queries = (
            rerank_queries
            or decision.embedding_texts()
            or ([decision.semantic_query] if decision.semantic_query else [])
        )
        answer, evidence = answer_with_llm(
            prompt,
            route=decision.route,
            records=answer_records,
            match_count=match_count,
            filters=answer_filters,
            intent=decision.intent,
            queries=answer_queries,
            model=llm_model,
        )
        result["evidence"] = evidence
        if answer:
            result["answer"] = answer
        else:
            result["answer_error"] = "LLM answer unavailable (check API key / provider)"

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify + query companies.jsonl")
    parser.add_argument("prompt", help="Natural language question")
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        default=DEFAULT_COMPANIES_PATH,
        help="Path to companies.jsonl",
    )
    parser.add_argument("-n", "--limit", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="Print raw JSON result")
    parser.add_argument(
        "--keyword",
        action="store_true",
        help="Force keyword retrieval instead of embeddings",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild embedding cache even if it exists",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="sentence-transformers model name",
    )
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="Disable LLM classifier; use rule-based router only",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="OpenAI-compatible chat model (default: OPENAI_MODEL or gpt-4o-mini)",
    )
    parser.add_argument(
        "--no-answer",
        action="store_true",
        help="Skip LLM synthesis of retrieved matches into a natural-language answer",
    )
    args = parser.parse_args()

    # Double-quoted "$50" is eaten by the shell → "revenue over million"
    if re.search(
        r"\b(?:revenue|employees?|headcount)\b.{0,20}\b(?:over|under|above|below|more than|less than)\b\s+(?:million|billion)\b",
        args.prompt,
        flags=re.I,
    ):
        print(
            "error: dollar amount missing — the shell expanded $50 inside double quotes.\n"
            "  Use single quotes:\n"
            "    python3 prompt_router.py 'Construction companies in the United States "
            "with revenue over $50 million'\n"
            "  Or escape the dollar sign:\n"
            r'    python3 prompt_router.py "... revenue over \$50 million"',
            file=sys.stderr,
        )
        return 2

    result = run_router(
        args.prompt,
        args.file,
        limit=args.limit,
        use_embeddings=not args.keyword,
        rebuild_index=args.rebuild_index,
        model_name=args.model,
        use_llm=not args.rules_only,
        llm_model=args.llm_model,
        use_answer=not args.no_answer and not args.rules_only,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    print(f"route: {result['route']}  ({result['reason']})")
    print(f"classifier: {result.get('classifier')}")
    print(f"retriever: {result.get('retriever')}")
    if result.get("filters"):
        shown = {k: v for k, v in result["filters"].items() if k != "country_codes"}
        if "country_codes" in result["filters"]:
            shown["countries"] = len(result["filters"]["country_codes"])
        print(f"filters: {shown}")
    if result.get("semantic_query"):
        print(f"semantic_query: {result['semantic_query']!r}")
    if result.get("intent"):
        intent = result["intent"]
        bits = []
        if intent.get("primary_theme"):
            bits.append(f"primary={intent['primary_theme']!r}")
        if intent.get("secondary_theme"):
            bits.append(f"secondary={intent['secondary_theme']!r}")
        if intent.get("uses_tool"):
            bits.append(f"uses={intent['uses_tool']!r}")
        if intent.get("contrast"):
            bits.append(f"contrast={intent['contrast']!r}")
        if intent.get("exclusions"):
            bits.append(f"exclusions={intent['exclusions']}")
        if bits:
            print(f"intent: {', '.join(bits)}")
    if result.get("embedding_queries"):
        print(f"embedding_queries: {result['embedding_queries']}")
    if "subset_size" in result:
        print(f"subset_size: {result['subset_size']}")
    print("matches:")
    for m in result["matches"]:
        print(f"  - {m}")
    if not result["matches"]:
        print("  (none)")
    if result.get("evidence") and not args.json:
        print("evidence:")
        for ev in result["evidence"][:5]:
            print(
                f"  - {ev.get('company')} [{ev.get('confidence')}] "
                f"why={ev.get('why_it_matches', [])[:1]}"
            )
    if result.get("answer"):
        print("\nanswer:")
        print(result["answer"])
    elif result.get("answer_error"):
        print(f"\nanswer: ({result['answer_error']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
