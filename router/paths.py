"""Project paths shared across the router package and scripts."""

from __future__ import annotations

from pathlib import Path

# router/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_COMPANIES_PATH = DATA_DIR / "companies.jsonl"
EMBEDDING_CACHE_DIR = PROJECT_ROOT / ".embedding_cache"
