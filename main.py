#!/usr/bin/env python3
"""
Central entry point for company search.

Examples:
  python main.py "Logistics companies in Romania"
  python main.py "Top 5 companies by revenue in US" --no-answer
  python main.py "Firms focused on renewable energy" --json
"""

from __future__ import annotations

from router.prompt_router import main

if __name__ == "__main__":
    raise SystemExit(main())
