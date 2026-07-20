"""
Company search router: classify natural-language prompts and retrieve matching companies.

Use ``main.py`` at the project root as the CLI entry point.
"""

from __future__ import annotations

from router.prompt_router import classify_prompt, run_router

__all__ = ["classify_prompt", "run_router"]
