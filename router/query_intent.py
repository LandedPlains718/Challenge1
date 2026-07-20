#!/usr/bin/env python3
"""Decompose a company-search prompt into primary/secondary themes and exclusions."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

# Short synonym expansions for common themes (rare brands alone rarely appear in data).
THEME_EXPANSIONS: dict[str, list[str]] = {
    "e-commerce": ["e-commerce", "ecommerce", "online retail", "online store"],
    "ecommerce": ["e-commerce", "ecommerce", "online retail", "online store"],
    "fintech": ["fintech", "digital banking", "online lending", "payment solutions"],
    "logistics": ["logistics", "freight transport", "warehousing", "port operations"],
    "logistic": ["logistics", "freight transport", "warehousing", "port operations"],
    "packaging": ["packaging", "packaging materials", "packaging suppliers"],
    "cybersecurity": ["cybersecurity", "cyber security", "information security"],
    "saas": ["saas", "software as a service", "cloud software"],
    "pharmaceutical": ["pharmaceutical", "pharma", "drug manufacturing", "biopharma"],
    "pharma": ["pharmaceutical", "pharma", "drug manufacturing", "biopharma"],
    "construction": ["construction", "building construction", "civil engineering"],
    "clean energy": ["clean energy", "renewable energy", "solar", "wind energy"],
    "renewable energy": ["renewable energy", "clean energy", "solar", "wind turbines"],
    "food and beverage": ["food and beverage", "food manufacturing", "beverage production"],
    "hr": ["hr software", "human resources", "hr saas", "payroll"],
    "software": ["software", "saas", "enterprise software"],
}


@dataclass
class QueryIntent:
    """Structured semantic intent extracted from a natural-language prompt."""

    primary_theme: str = ""
    secondary_theme: str = ""
    exclusions: list[str] = field(default_factory=list)
    contrast: str = ""  # competitive/context foil, not a target to retrieve
    # Soft tool/platform the company *uses* (e.g. Shopify) — not what they build.
    uses_tool: str = ""

    def embedding_queries(self, fallback: list[str] | None = None) -> list[str]:
        """Phrases to embed: primary first, then secondary/synonyms, never exclusions/contrast."""
        out: list[str] = []
        seen: set[str] = set()

        def _add(q: str) -> None:
            q = (q or "").strip()
            if not q or len(q) > 80:
                return
            key = q.lower()
            if key in seen or self._is_excluded_phrase(q):
                return
            seen.add(key)
            out.append(q)

        # Prefer short primary + expansions over long dumped clauses
        primary = (self.primary_theme or "").strip()
        if primary:
            _add(primary)
            for syn in _theme_synonyms(primary):
                _add(syn)
                if len(out) >= 5:
                    return out

        if self.secondary_theme.strip():
            _add(self.secondary_theme.strip())

        # Tool brands aren't in the corpus — expand to operator signals for that tool class
        if self.uses_tool.strip():
            from router.tool_profiles import operator_expansions_for

            for phrase in operator_expansions_for(self.uses_tool):
                _add(phrase)
                if len(out) >= 5:
                    return out

        for q in fallback or []:
            # Skip long multi-clause dumps that dilute embedding recall
            if len((q or "").split()) > 5:
                for piece in _split_long_theme(q):
                    _add(piece)
                    if len(out) >= 5:
                        return out
                continue
            _add(q)
            if len(out) >= 5:
                break
        return out

    def _is_excluded_phrase(self, phrase: str) -> bool:
        p = phrase.lower().strip()
        blocked = [e.lower() for e in self.exclusions if e.strip()]
        if self.contrast.strip():
            blocked.append(self.contrast.lower().strip())
        return any(p == b or p in b or b in p for b in blocked if len(b) >= 4)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_empty(self) -> bool:
        return not (
            self.primary_theme.strip()
            or self.secondary_theme.strip()
            or self.exclusions
            or self.contrast.strip()
            or self.uses_tool.strip()
        )


# Anchors that only expand when they ARE the whole theme (not a suffix of a niche phrase).
_GENERIC_THEME_ANCHORS = frozenset({"software", "saas", "hr"})


def _theme_synonyms(theme: str) -> list[str]:
    key = theme.lower().strip()
    if key in THEME_EXPANSIONS:
        return [s for s in THEME_EXPANSIONS[key] if s.lower() != key]
    # Prefer the longest matching anchor (e.g. "clean energy" over "energy")
    matches: list[tuple[int, str, list[str]]] = []
    for anchor, syns in THEME_EXPANSIONS.items():
        if anchor in key:
            matches.append((len(anchor), anchor, syns))
    if not matches:
        return []
    matches.sort(key=lambda x: x[0], reverse=True)
    _, anchor, syns = matches[0]
    # "predictive maintenance software" must NOT expand to generic saas/HR noise
    if anchor in _GENERIC_THEME_ANCHORS and key != anchor:
        return []
    return [s for s in syns if s.lower() != key]


def _split_long_theme(text: str) -> list[str]:
    """Break verbose themes into short embeddable anchors."""
    cleaned = _clean_theme(_strip_geo_noise(text))
    if not cleaned:
        return []
    # Prefer known category anchors inside the phrase
    found: list[str] = []
    lower = cleaned.lower()
    for anchor in sorted(THEME_EXPANSIONS, key=len, reverse=True):
        if anchor in lower:
            found.append(anchor)
    if found:
        return found
    # First 1-3 content tokens
    toks = cleaned.split()
    if len(toks) <= 3:
        return [cleaned]
    return [" ".join(toks[:2]), toks[0]]


def _clean_theme(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" ,.-?")
    text = re.sub(
        r"\b(?:companies|company|firms|firm|businesses|manufacturers|manufacturer|"
        r"suppliers|supplier|vendors|vendor|providers|provider|startups|startup|"
        r"operators|operator|brands|brand)\b",
        " ",
        text,
        flags=re.I,
    )
    # Boilerplate verbs / growth adjectives that dilute themes
    text = re.sub(
        r"\b(?:find|show|list|get|give|name|that|could|can|would|should|supply|"
        r"supplies|offering|offer|providing|provide|focused|specializing|"
        r"fast[- ]growing|growing|leading|top|best|direct[- ]to[- ]consumer|"
        r"d2c|a|an|the|of|to|in|from|with|for)\b",
        " ",
        text,
        flags=re.I,
    )
    # Platform hedges: "using Shopify or similar platforms"
    text = re.sub(
        r"\b(?:using|on|via|with)\s+[A-Za-z0-9][\w.-]*(?:\s+or\s+similar(?:\s+\w+)*)?",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\bor\s+similar(?:\s+\w+)*\b", " ", text, flags=re.I)
    text = re.sub(r"\bplatforms?\b", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip(" ,.-?")


def _strip_geo_noise(text: str) -> str:
    """Remove country/region phrases that belong in structured filters, not themes."""
    geo = (
        r"europe|european|eu|asia|asian|north america|scandinavia|nordics|"
        r"middle east|romania|romanian|france|french|germany|german|uk|"
        r"united kingdom|united states|usa|us|spain|spanish|italy|italian|"
        r"netherlands|dutch|switzerland|swiss|china|chinese|india|indian|"
        r"australia|australian|canada|canadian|israel|israeli|brazil|brazilian|"
        r"japan|japanese"
    )
    text = re.sub(rf"\b(?:in|from|based in)\s+(?:the\s+)?(?:{geo})\b", " ", text, flags=re.I)
    text = re.sub(rf"\b(?:{geo})\b", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip(" ,.-?")


def decompose_intent_rules(prompt: str, *, semantic_query: str | None = None) -> QueryIntent:
    """
    Lightweight rule-based intent decomposition (fallback when LLM is unavailable).

    Patterns:
      "X for Y" / "X suppliers for Y" → primary=X, secondary=Y
      "X using Y or similar" → primary=X, secondary=Y
      "competing with/against Y" → contrast=Y
      "not Y" / "excluding Y" / "instead of Y" → exclusion=Y
    """
    text = prompt.strip()
    intent = QueryIntent()

    # Contrast: competing with / vs / versus traditional banks
    m = re.search(
        r"\b(?:competing with|compete with|vs\.?|versus|against)\s+(.+?)(?:\.|$)",
        text,
        flags=re.I,
    )
    if m:
        intent.contrast = _clean_theme(_strip_geo_noise(m.group(1)))
        text = re.sub(
            r"\b(?:competing with|compete with|vs\.?|versus|against)\s+.+$",
            " ",
            text,
            flags=re.I,
        )

    # Exclusions
    for pat in (
        r"\b(?:not|excluding|except|instead of|rather than)\s+([^,.]+)",
    ):
        for m in re.finditer(pat, text, flags=re.I):
            excl = _clean_theme(_strip_geo_noise(m.group(1)))
            if excl and excl.lower() not in {e.lower() for e in intent.exclusions}:
                intent.exclusions.append(excl)

    # "e-commerce companies using Shopify or similar platforms"
    # → primary = e-commerce merchants; uses_tool = shopify (not platform vendors)
    m = re.search(
        r"(.+?)\s+(?:using|on|via)\s+([A-Za-z0-9][\w.-]*)(?:\s+or\s+similar.*)?",
        text,
        flags=re.I,
    )
    if m:
        left = _clean_theme(_strip_geo_noise(m.group(1)))
        platform = m.group(2).strip()
        left = re.sub(
            r"^(?:find|show|list|get|give|name|companies that|who)\s+",
            "",
            left,
            flags=re.I,
        ).strip()
        if left:
            intent.primary_theme = left
            if platform and platform.lower() not in {"similar", "platforms", "platform"}:
                intent.uses_tool = platform
            return intent

    # "packaging materials for cosmetics" / "X for Y"
    m = re.search(
        r"(.+?)\s+(?:for|serving|targeting)\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\.|$)",
        text,
        flags=re.I,
    )
    if m:
        left = _clean_theme(_strip_geo_noise(m.group(1)))
        right = _clean_theme(_strip_geo_noise(m.group(2)))
        left = re.sub(
            r"^(?:find|show|list|get|give|name|companies that|who)\s+",
            "",
            left,
            flags=re.I,
        ).strip()
        if left and right and len(left.split()) <= 6 and len(right.split()) <= 6:
            # "software for fleet optimization" → primary=fleet optimization (not software)
            generic_left = {
                "software",
                "saas",
                "platform",
                "platforms",
                "solutions",
                "services",
                "tools",
                "systems",
                "technology",
                "tech",
            }
            if left.lower() in generic_left and right.lower() not in generic_left:
                intent.primary_theme = right
                intent.secondary_theme = left
            else:
                intent.primary_theme = left
                intent.secondary_theme = right

    if not intent.primary_theme:
        cleaned = _clean_theme(_strip_geo_noise(text))
        if cleaned and len(cleaned.split()) <= 8:
            intent.primary_theme = cleaned

    # Only use caller semantic_query if it is clean and does not contain contrast
    if not intent.primary_theme and semantic_query:
        sq = _clean_theme(_strip_geo_noise(semantic_query))
        contrast_l = intent.contrast.lower()
        if sq and (not contrast_l or contrast_l not in sq.lower()):
            # Drop leftover contrast tokens from semantic_query
            if contrast_l:
                for tok in contrast_l.split():
                    if len(tok) > 3:
                        sq = re.sub(rf"\b{re.escape(tok)}\b", " ", sq, flags=re.I)
                sq = re.sub(r"\s+", " ", sq).strip()
            if sq:
                intent.primary_theme = sq

    return intent


def intent_from_llm_payload(
    data: dict[str, Any],
    *,
    semantic_query: str = "",
) -> QueryIntent:
    """Build QueryIntent from LLM JSON fields."""
    primary = str(data.get("primary_theme") or "").strip()
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

    if not primary and semantic_query:
        primary = semantic_query.strip()

    # Clean verbose LLM dumps ("E-commerce using Shopify or similar platforms")
    uses_tool = ""
    if primary and (
        len(primary.split()) > 4
        or re.search(r"\b(?:using|or similar)\b", primary, flags=re.I)
    ):
        cleaned = decompose_intent_rules(primary)
        if cleaned.primary_theme:
            if not secondary and cleaned.secondary_theme:
                secondary = cleaned.secondary_theme
            if cleaned.uses_tool:
                uses_tool = cleaned.uses_tool
            primary = cleaned.primary_theme

    # LLM sometimes puts the used platform in secondary_theme
    from router.tool_profiles import is_known_tool

    if not uses_tool and secondary and is_known_tool(secondary):
        uses_tool = secondary
        secondary = ""

    return QueryIntent(
        primary_theme=primary,
        secondary_theme=secondary,
        exclusions=exclusions[:5],
        contrast=contrast,
        uses_tool=uses_tool,
    )
