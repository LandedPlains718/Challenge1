#!/usr/bin/env python3
"""Regenerate context_hints.py from companies.jsonl."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

STOP = {
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
    "by",
    "as",
    "at",
    "from",
    "into",
    "other",
    "all",
    "set",
    "units",
    "related",
    "activities",
    "general",
    "miscellaneous",
    "based",
    "provider",
    "providers",
    "company",
    "companies",
    "business",
    "services",
    "service",
    "products",
    "product",
    "solutions",
    "solution",
    "systems",
    "system",
    "management",
    "development",
    "manufacturing",
    "except",
    "non",
    "dba",
    "inc",
    "ltd",
    "llc",
    "sa",
    "srl",
    "group",
    "international",
    "global",
    "new",
    "high",
    "low",
    "end",
    "use",
    "using",
    "including",
    "via",
    "per",
    "its",
    "their",
    "than",
    "more",
    "less",
    "over",
    "under",
    "between",
    "within",
    "across",
    "through",
    "also",
    "such",
    "these",
    "those",
    "this",
    "that",
    "which",
    "who",
    "what",
    "when",
    "where",
    "how",
    "many",
    "most",
    "some",
    "any",
    "both",
    "each",
    "every",
    "same",
    "own",
    "only",
    "just",
    "not",
    "advanced",
    "basic",
    "special",
    "primary",
    "secondary",
    "main",
}

GENERIC = {
    "enterprise",
    "wholesale",
    "retail",
    "technology",
    "technologies",
    "equipment",
    "material",
    "materials",
    "process",
    "processing",
    "production",
    "distribution",
    "sales",
    "support",
    "custom",
    "commercial",
    "industrial",
    "consumer",
    "application",
    "applications",
    "platform",
    "platforms",
    "model",
    "models",
    "data",
    "digital",
    "project",
    "projects",
    "design",
    "research",
    "integration",
    "implementation",
    "deployment",
    "installation",
    "maintenance",
    "training",
    "consulting",
    "advisory",
    "assessment",
    "optimization",
    "automation",
    "innovation",
    "performance",
    "efficiency",
    "reduction",
    "enhancement",
    "delivery",
    "tracking",
    "transfer",
    "facility",
    "planning",
    "building",
    "home",
    "care",
    "goods",
    "component",
    "components",
    "device",
    "devices",
    "preparation",
    "centers",
    "center",
    "agents",
    "brokers",
    "builders",
    "apparatus",
    "allied",
    "analysis",
    "businesses",
    "chain",
    "changer",
    "cities",
    "aged",
}

INTENT = {
    "like",
    "similar",
    "about",
    "focused",
    "specialize",
    "specialized",
    "specialising",
    "offering",
    "offerings",
    "describe",
    "involved in",
    "working on",
    "looking for",
    "searching for",
    "who does",
    "what does",
    "sector",
    "industry",
    "industries",
    "vertical",
}


def parse_maybe(value):
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


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["primary_naics"] = parse_maybe(row.get("primary_naics"))
            row["secondary_naics"] = parse_maybe(row.get("secondary_naics"))
            rows.append(row)
    return rows


def add_phrase_tokens(hints: set[str], phrase: str) -> None:
    hints.add(phrase)
    for word in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", phrase):
        if len(word) >= 3 and word not in STOP and word not in GENERIC and not word.isdigit():
            hints.add(word)


def build_hints(rows: list[dict]) -> tuple[set[str], dict[str, int]]:
    hints = set(INTENT)
    stats = {"markets": 0, "models": 0, "naics": 0}

    market_phrases: set[str] = set()
    model_phrases: set[str] = set()
    naics_phrases: set[str] = set()

    for row in rows:
        for item in row.get("target_markets") or []:
            if isinstance(item, str) and item.strip():
                phrase = item.strip().lower()
                market_phrases.add(phrase)
                add_phrase_tokens(hints, phrase)
        for item in row.get("business_model") or []:
            if isinstance(item, str) and item.strip():
                phrase = item.strip().lower()
                model_phrases.add(phrase)
                add_phrase_tokens(hints, phrase)
        for key in ("primary_naics", "secondary_naics"):
            naics = row.get(key)
            if isinstance(naics, dict) and naics.get("label"):
                label = str(naics["label"]).strip().lower()
                naics_phrases.add(label)
                add_phrase_tokens(hints, label)

    # light singular/plural variants for single-token hints
    extras: set[str] = set()
    for hint in list(hints):
        if " " in hint or len(hint) < 4:
            continue
        if hint.endswith("ies") and len(hint) > 5:
            extras.add(hint[:-3] + "y")
        elif hint.endswith("s") and not hint.endswith("ss"):
            extras.add(hint[:-1])
        else:
            extras.add(hint + "s")
    hints |= {e for e in extras if len(e) >= 3 and e not in STOP and e not in GENERIC}

    stats["markets"] = len(market_phrases)
    stats["models"] = len(model_phrases)
    stats["naics"] = len(naics_phrases)
    return hints, stats


def write_hints(hints: set[str], stats: dict[str, int], out: Path) -> None:
    lines = [
        '"""Auto-generated from companies.jsonl target_markets, business_model, NAICS labels."""',
        "",
        (
            f"# markets={stats['markets']} models={stats['models']} "
            f"naics={stats['naics']} total_hints={len(hints)}"
        ),
        "CONTEXT_HINTS = frozenset({",
    ]
    for hint in sorted(hints, key=lambda s: (0 if " " in s else 1, s)):
        lines.append(f"    {hint!r},")
    lines.append("})")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        default=root / "data" / "companies.jsonl",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=root / "router" / "context_hints.py",
    )
    args = parser.parse_args()

    rows = load_rows(args.file)
    hints, stats = build_hints(rows)
    write_hints(hints, stats, args.output)
    print(
        f"wrote {args.output} "
        f"(markets={stats['markets']} models={stats['models']} "
        f"naics={stats['naics']} hints={len(hints)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
