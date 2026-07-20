#!/usr/bin/env python3
"""Synthesize a natural-language answer from grounded evidence objects."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from router.evidence import build_evidence_list
from router.query_intent import QueryIntent

SYSTEM = """You answer company-search questions using ONLY the evidence objects below.

Rules:
- Use ONLY companies and facts present in the evidence. Do not invent firms or outside examples.
- The evidence list IS the match set. You MUST cover every company with confidence "high" or
  "medium" that has a non-empty why_it_matches. Do not silently drop them.
- Adjacent NAICS is fine when offerings/markets show the theme (e.g. intralogistics equipment,
  logistics & storage services, warehousing, freight, ports). Mention the nuance briefly.
- Prefer higher-confidence companies first. Only omit a company when confidence is "low" AND
  why_it_matches is weak/empty, or weaknesses clearly show it is off-theme.
- Prefer companies whose why_it_matches cites core offerings / target markets / NAICS /
  business model with concrete snippets. Treat "field overlap is weak" as low value.
- Mention material weaknesses briefly; do not let minor caveats erase a valid match.
- If uses_tool is set (Shopify, Stripe, SAP, Salesforce, Epic, etc.):
  * The user wants OPERATORS in the primary theme that might use that tool — not vendors
    that build/sell the tool.
  * Start by saying the dataset does not record which tool each company uses; "or similar"
    means theme operators that might use that class of tool.
  * NEVER open with "no matching companies" when operator evidence exists.
  * NEVER invent or deny tool usage — say it is unverified in the data.
  * For storefront tools (Shopify/Magento/…): prefer e-commerce merchants / online retailers.
- Explain why each named company matches using why_it_matches / supporting_fields.
- Respect query intent: primary_theme is the target; secondary_theme refines it;
  contrast/exclusions are NOT targets — treat them as caveats, not search goals.
- For list/count questions: if Total matching companies is provided, use that number;
  otherwise count the high+medium companies you included — do NOT invent a smaller total.
- If evidence is empty, say no matching companies were found. Do NOT name external examples.
- Be concise but complete: a short bullet per included company is preferred over dropping matches.
- Do not mention embeddings, routing, reranking, or internal mechanics.
"""


def _llm_client_config(
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[str, str, str] | None:
    """Return (base_url, model, api_key) or None if unavailable."""
    provider = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    env_base = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OLLAMA_BASE_URL")

    using_ollama = provider == "ollama" or bool(
        os.environ.get("OLLAMA_HOST") or (env_base and "11434" in env_base)
    )
    using_gemini = bool(
        provider == "gemini"
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or (env_base and "generativelanguage.googleapis.com" in env_base)
    )

    resolved_key = (
        api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ("ollama" if using_ollama else None)
    )
    if not resolved_key and not using_ollama:
        return None

    if using_ollama:
        resolved_base = (env_base or "http://127.0.0.1:11434/v1").rstrip("/")
        resolved_model = model or os.environ.get("OPENAI_MODEL") or os.environ.get("OLLAMA_MODEL") or "qwen2.5:7b"
        resolved_key = resolved_key or "ollama"
    elif using_gemini and not env_base:
        resolved_base = "https://generativelanguage.googleapis.com/v1beta/openai"
        resolved_model = model or os.environ.get("OPENAI_MODEL") or "gemini-2.0-flash"
    else:
        resolved_base = (env_base or "https://api.openai.com/v1").rstrip("/")
        resolved_model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"

    return resolved_base, resolved_model, resolved_key


def _chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 90.0,
    temperature: float = 0.2,
) -> str | None:
    cfg = _llm_client_config(model=model, api_key=api_key, base_url=base_url)
    if cfg is None:
        return None

    resolved_base, resolved_model, resolved_key = cfg
    url = f"{resolved_base}/chat/completions"

    body: dict[str, Any] = {
        "model": resolved_model,
        "temperature": temperature,
        "messages": messages,
    }
    using_ollama = resolved_base.endswith(":11434/v1") or "11434" in resolved_base
    if not using_ollama:
        body["response_format"] = {"type": "text"}

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {resolved_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        return str(content).strip()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        print(f"llm_answerer HTTP {exc.code}: {detail}", file=__import__("sys").stderr)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        print(f"llm_answerer error: {exc}", file=__import__("sys").stderr)
    return None


def _build_user_message(
    prompt: str,
    *,
    route: str,
    evidence: list[dict[str, Any]],
    intent: QueryIntent | None = None,
    match_count: int | None = None,
    filters: dict[str, Any] | None = None,
) -> str:
    parts = [f"User question:\n{prompt.strip()}", f"Route: {route}"]
    if intent and not intent.is_empty():
        parts.append(f"Query intent: {json.dumps(intent.to_dict(), ensure_ascii=False)}")
        if intent.uses_tool.strip():
            from router.tool_profiles import operator_label_for

            label = operator_label_for(
                intent.uses_tool, primary_theme=intent.primary_theme or ""
            )
            parts.append(
                f"INTERPRETATION (mandatory): uses_tool={intent.uses_tool!r} means list "
                f"{label}. {intent.uses_tool} usage is NOT recorded in the dataset — "
                "do not require proof of that brand. Do not say 'no matches' when evidence "
                "exists. Do not answer with vendors that build that tool class."
            )
    if filters:
        parts.append(f"Structured filters applied: {json.dumps(filters, ensure_ascii=False)}")
    if match_count is not None:
        parts.append(f"Total matching companies in dataset: {match_count}")

    parts.append(f"Evidence objects ({len(evidence)}):")
    if not evidence:
        parts.append("(none)")
    else:
        parts.append(json.dumps(evidence, indent=2, ensure_ascii=False))

    high_med = [
        e.get("company")
        for e in evidence
        if e.get("confidence") in {"high", "medium"} and e.get("why_it_matches")
    ]
    if high_med:
        parts.append(
            "Include ALL of these high/medium matches in your answer: "
            + ", ".join(str(c) for c in high_med)
            + "."
        )
    parts.append("\nWrite the answer for the user using ONLY the evidence above.")
    return "\n".join(parts)


def answer_with_llm(
    prompt: str,
    *,
    route: str,
    records: list[dict[str, Any]],
    match_count: int | None = None,
    filters: dict[str, Any] | None = None,
    intent: QueryIntent | None = None,
    queries: list[str] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 90.0,
) -> tuple[str | None, list[dict[str, Any]]]:
    """
    Build evidence from records (if needed) and return (answer, evidence_list).
    """
    q = list(queries or [])
    if not q and intent and intent.primary_theme:
        q = intent.embedding_queries()
    if evidence is None:
        evidence = build_evidence_list(records, queries=q, intent=intent)

    user_content = _build_user_message(
        prompt,
        route=route,
        evidence=evidence,
        intent=intent,
        match_count=match_count,
        filters=filters,
    )
    answer = _chat_completion(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_content},
        ],
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )
    return answer, evidence
