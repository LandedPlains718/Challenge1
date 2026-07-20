#!/usr/bin/env python3
"""
Verify LLM/rules routing against a golden set.

Usage:
  export LLM_PROVIDER=ollama
  python scripts/eval_router.py              # LLM (+ reconcile)
  python scripts/eval_router.py --rules-only
  python scripts/eval_router.py --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router.prompt_router import classify_prompt, classify_prompt_rules

# expected: route, and optional semantic_query substring that must appear
GOLDEN = [
    {
        'prompt': 'Companies in Romania',
        'route': 'structured',
        'semantic_empty': True,
    },
    {
        'prompt': 'Top 5 companies by revenue in US',
        'route': 'structured',
        'semantic_empty': True,
    },
    {
        'prompt': 'Public companies with more than 10000 employees',
        'route': 'structured',
        'semantic_empty': True,
    },
    {
        'prompt': 'Companies supplying packaging for cosmetics brands',
        'route': 'context',
        'semantic_contains': ['packaging', 'cosmetic'],
        'semantic_any': True,
    },
    {
        'prompt': 'Firms focused on decarbonization and renewable energy',
        'route': 'context',
        'semantic_contains': ['renewable', 'decarbon', 'energy'],
        'semantic_any': True,
    },
    {
        'prompt': 'Logistics companies in Romania',
        'route': 'hybrid',
        'semantic_contains': ['logistic'],
    },
    {
        'prompt': 'Logistic companies in Romania',
        'route': 'hybrid',
        'semantic_contains': ['logistic'],
    },
    {
        'prompt': 'Public software companies with more than 1000 employees from europe',
        'route': 'hybrid',
        'semantic_contains': ['software'],
    },
    {
        'prompt': 'Public software companies with more than 1,000 employees.',
        'route': 'hybrid',
        'semantic_contains': ['software'],
    },
    {
        'prompt': 'Food and beverage manufacturers in France',
        'route': 'hybrid',
        'semantic_contains': ['food', 'beverage'],
        'semantic_any': True,
    },
    {
        'prompt': 'Companies that could supply packaging materials for a direct-to-consumer cosmetics brand',
        'route': 'context',
        'semantic_contains': ['packaging', 'cosmetic'],
        'semantic_any': True,
    },
    {
        'prompt': 'Construction companies in the United States with revenue over $50 million',
        'route': 'hybrid',
        'semantic_contains': ['construction'],
    },
    {
        'prompt': 'Pharmaceutical companies in Switzerland',
        'route': 'hybrid',
        'semantic_contains': ['pharma'],
    },
    {
        'prompt': 'B2B SaaS companies providing HR solutions in Europe',
        'route': 'hybrid',
        'semantic_contains': ['hr', 'saas', 'human'],
        'semantic_any': True,
    },
    {
        'prompt': 'Clean energy startups founded after 2018 with fewer than 200 employees',
        'route': 'hybrid',
        'semantic_contains': ['energy', 'clean', 'renewable'],
        'semantic_any': True,
    },
    {
        'prompt': 'Fast-growing fintech companies competing with traditional banks in Europe.',
        'route': 'hybrid',
        'semantic_contains': ['fintech', 'bank', 'financ'],
        'semantic_any': True,
    },
    {
        'prompt': 'E-commerce companies using Shopify or similar platforms',
        'route': 'context',
        'semantic_contains': ['shopify', 'commerce', 'ecommerce', 'e-commerce'],
        'semantic_any': True,
    },
    {
        'prompt': 'Renewable energy equipment manufacturers in Scandinavia',
        'route': 'hybrid',
        'semantic_contains': ['renewable', 'energy'],
        'semantic_any': True,
    },
    {
        'prompt': 'Companies that manufacture or supply critical components for electric vehicle battery production',
        'route': 'context',
        'semantic_contains': ['battery', 'electric', 'vehicle', 'ev'],
        'semantic_any': True,
    },
    {
        'prompt': 'How many public companies are in the dataset?',
        'route': 'structured',
        'semantic_empty': True,
    },
    {
        'prompt': 'Largest companies by employee count',
        'route': 'structured',
        'semantic_empty': True,
    },
    {
        'prompt': 'Private companies founded before 2000',
        'route': 'structured',
        'semantic_empty': True,
    },
    {
        'prompt': 'Companies based in Germany',
        'route': 'structured',
        'semantic_empty': True,
    },
    {
        'prompt': 'Firms with revenue under $1 million',
        'route': 'structured',
        'semantic_empty': True,
    },
    {
        'prompt': 'List companies in Asia',
        'route': 'structured',
        'semantic_empty': True,
    },
    {
        'prompt': 'Who provides payroll and talent management software?',
        'route': 'context',
        'semantic_contains': ['payroll', 'talent', 'hr', 'software'],
        'semantic_any': True,
    },
    {
        'prompt': 'Companies similar to Rompetrol',
        'route': 'context',
        'semantic_contains': ['rompetrol', 'energy', 'oil', 'petroleum', 'refin'],
        'semantic_any': True,
    },
    {
        'prompt': 'Vendors offering cybersecurity and cloud infrastructure services',
        'route': 'context',
        'semantic_contains': ['cyber', 'cloud', 'security', 'infrastructure'],
        'semantic_any': True,
    },
    {
        'prompt': 'Manufacturers of wind turbines and related equipment',
        'route': 'context',
        'semantic_contains': ['wind', 'turbine'],
        'semantic_any': True,
    },
    {
        'prompt': 'Businesses specializing in machine learning model deployment',
        'route': 'context',
        'semantic_contains': ['machine learning', 'ml', 'ai', 'model'],
        'semantic_any': True,
    },
    {
        'prompt': 'Suppliers of lithium-ion battery materials and cathodes',
        'route': 'context',
        'semantic_contains': ['battery', 'lithium', 'cathode'],
        'semantic_any': True,
    },
    {
        'prompt': 'Companies building autonomous underwater or ocean energy devices',
        'route': 'context',
        'semantic_contains': ['ocean', 'underwater', 'tidal', 'marine', 'wave'],
        'semantic_any': True,
    },
    {
        'prompt': 'Agtech firms developing biostimulants and plant nutrition products',
        'route': 'context',
        'semantic_contains': ['agtech', 'agricultur', 'biostimul', 'plant', 'fertiliz'],
        'semantic_any': True,
    },
    {
        'prompt': 'Automotive suppliers in Germany with more than 500 employees',
        'route': 'hybrid',
        'semantic_contains': ['automotive', 'auto', 'vehicle'],
        'semantic_any': True,
    },
    {
        'prompt': 'Semiconductor companies in Asia',
        'route': 'hybrid',
        'semantic_contains': ['semiconductor', 'chip'],
        'semantic_any': True,
    },
    {
        'prompt': 'Real estate service providers in Romania',
        'route': 'hybrid',
        'semantic_contains': ['real estate', 'property', 'estate'],
        'semantic_any': True,
    },
    {
        'prompt': 'Public renewable energy companies in Europe',
        'route': 'hybrid',
        'semantic_contains': ['renewable', 'energy'],
        'semantic_any': True,
    },
    {
        'prompt': 'Healthcare IT vendors in the United States',
        'route': 'hybrid',
        'semantic_contains': ['health', 'healthcare', 'it', 'software'],
        'semantic_any': True,
    },
    {
        'prompt': 'Mining and metals companies in Australia with revenue over $10 million',
        'route': 'hybrid',
        'semantic_contains': ['mining', 'metal'],
        'semantic_any': True,
    },
    {
        'prompt': 'Insurance technology startups in the UK founded after 2015',
        'route': 'hybrid',
        'semantic_contains': ['insurance', 'insurtech', 'fintech'],
        'semantic_any': True,
    },
    {
        'prompt': 'Defense and aerospace contractors in the US',
        'route': 'hybrid',
        'semantic_contains': ['defense', 'aerospace', 'defence'],
        'semantic_any': True,
    },
    {
        'prompt': 'Water treatment and environmental services companies in France',
        'route': 'hybrid',
        'semantic_contains': ['water', 'environment'],
        'semantic_any': True,
    },
    {
        'prompt': 'Telecom operators in Spain',
        'route': 'hybrid',
        'semantic_contains': ['telecom', 'telecommunication'],
        'semantic_any': True,
    },
    {
        'prompt': 'Private equity-backed industrial manufacturers in North America',
        'route': 'hybrid',
        'semantic_contains': ['industrial', 'manufactur', 'equity'],
        'semantic_any': True,
    },
    {
        'prompt': 'Battery manufacturers in China with fewer than 100 employees',
        'route': 'hybrid',
        'semantic_contains': ['battery'],
    },
    {
        'prompt': 'Count of private companies founded after 2010',
        'route': 'structured',
        'semantic_empty': True,
    },
    {
        'prompt': 'Show companies headquartered in Japan',
        'route': 'structured',
        'semantic_empty': True,
    },
    {
        'prompt': 'Top 10 firms by headcount in Canada',
        'route': 'structured',
        'semantic_empty': True,
    },
    {
        'prompt': 'Companies with between 50 and 500 employees',
        'route': 'structured',
        'semantic_empty': True,
    },
    {
        'prompt': 'Public companies in the Nordics',
        'route': 'structured',
        'semantic_empty': True,
    },
    {
        'prompt': 'Smallest companies by revenue in Italy',
        'route': 'structured',
        'semantic_empty': True,
    },
    {
        'prompt': 'How many companies have revenue over $100 million?',
        'route': 'structured',
        'semantic_empty': True,
    },
    {
        'prompt': 'List all companies in Brazil',
        'route': 'structured',
        'semantic_empty': True,
    },
    {
        'prompt': 'Who makes industrial adhesives for packaging lines?',
        'route': 'context',
        'semantic_contains': ['adhesive', 'packaging'],
        'semantic_any': True,
    },
    {
        'prompt': 'Firms helping enterprises migrate workloads to the cloud',
        'route': 'context',
        'semantic_contains': ['cloud', 'migrate', 'workload'],
        'semantic_any': True,
    },
    {
        'prompt': 'Providers of last-mile delivery and courier software',
        'route': 'context',
        'semantic_contains': ['delivery', 'courier', 'last-mile', 'logistics'],
        'semantic_any': True,
    },
    {
        'prompt': 'Companies developing carbon capture and storage technology',
        'route': 'context',
        'semantic_contains': ['carbon', 'capture', 'ccs', 'storage'],
        'semantic_any': True,
    },
    {
        'prompt': 'Vendors of laboratory equipment for biotech research',
        'route': 'context',
        'semantic_contains': ['laboratory', 'lab', 'biotech', 'equipment'],
        'semantic_any': True,
    },
    {
        'prompt': 'Businesses that sell orthodontic and dental implants',
        'route': 'context',
        'semantic_contains': ['dental', 'orthodont', 'implant'],
        'semantic_any': True,
    },
    {
        'prompt': 'Suppliers focused on sustainable textiles and recycled fabrics',
        'route': 'context',
        'semantic_contains': ['textile', 'fabric', 'sustainable', 'recycl'],
        'semantic_any': True,
    },
    {
        'prompt': 'Companies building robotics for warehouse automation',
        'route': 'context',
        'semantic_contains': ['robot', 'warehouse', 'automat'],
        'semantic_any': True,
    },
    {
        'prompt': 'Biotech companies in Switzerland with fewer than 250 employees',
        'route': 'hybrid',
        'semantic_contains': ['biotech', 'bio'],
        'semantic_any': True,
    },
    {
        'prompt': 'Retail banks in Germany',
        'route': 'hybrid',
        'semantic_contains': ['bank', 'retail'],
        'semantic_any': True,
    },
    {
        'prompt': 'Cybersecurity vendors in Israel founded after 2012',
        'route': 'hybrid',
        'semantic_contains': ['cyber', 'security'],
        'semantic_any': True,
    },
    {
        'prompt': 'Solar panel manufacturers in India with revenue over $5 million',
        'route': 'hybrid',
        'semantic_contains': ['solar', 'panel', 'photovolta'],
        'semantic_any': True,
    },
    {
        'prompt': 'Public pharmaceutical companies in the United Kingdom',
        'route': 'hybrid',
        'semantic_contains': ['pharma'],
    },
    {
        'prompt': 'Logistics and freight forwarders in the Netherlands',
        'route': 'hybrid',
        'semantic_contains': ['logistic', 'freight'],
        'semantic_any': True,
    },
    {
        'prompt': 'AI startups in France with more than 20 employees',
        'route': 'hybrid',
        'semantic_contains': ['ai', 'artificial'],
        'semantic_any': True,
    },
    {
        'prompt': 'Oil and gas companies in the Middle East',
        'route': 'hybrid',
        'semantic_contains': ['oil', 'gas', 'petroleum', 'energy'],
        'semantic_any': True,
    },
    {
        'prompt': 'Waste management companies in Spain with revenue under $20 million',
        'route': 'hybrid',
        'semantic_contains': ['waste', 'management', 'recycl'],
        'semantic_any': True,
    },
    {
        'prompt': 'Edtech platforms in the US founded after 2016',
        'route': 'hybrid',
        'semantic_contains': ['edtech', 'education', 'learning'],
        'semantic_any': True,
    },
]


def check_case(case: dict, decision) -> list[str]:
    errors: list[str] = []
    expected_route = case.get("route")
    if expected_route and decision.route != expected_route:
        errors.append(f"route={decision.route!r} expected={expected_route!r}")

    sem = (decision.semantic_query or "").strip().lower()
    if case.get("semantic_empty"):
        if sem:
            errors.append(
                f"semantic_query should be empty, got {decision.semantic_query!r}"
            )
    if "semantic_contains" in case:
        tokens = case["semantic_contains"]
        if case.get("semantic_any"):
            if not any(t in sem for t in tokens):
                errors.append(f"semantic_query {sem!r} missing any of {tokens}")
        else:
            missing = [t for t in tokens if t not in sem]
            if missing:
                errors.append(f"semantic_query {sem!r} missing {missing}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules-only", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    passed = 0
    failed = 0

    for case in GOLDEN:
        prompt = case["prompt"]
        if args.rules_only:
            decision = classify_prompt_rules(prompt)
        else:
            decision = classify_prompt(prompt, use_llm=True)

        errors = check_case(case, decision)
        ok = not errors
        passed += int(ok)
        failed += int(not ok)

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {prompt}")
        if args.verbose or not ok:
            print(
                f"       route={decision.route} semantic={decision.semantic_query!r} "
                f"classifier={decision.classifier}"
            )
            for err in errors:
                print(f"       ! {err}")

    total = passed + failed
    print(f"\n{passed}/{total} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
