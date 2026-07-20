#!/usr/bin/env python3
"""Embedding index for company context retrieval."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from router.paths import EMBEDDING_CACHE_DIR

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Rank by strongest thematic field match — full-blob averaging dilutes signal.
FIELDS = ("markets", "naics", "offerings", "description")
THEMATIC_FIELDS = ("markets", "naics", "offerings")

_LEX_STOP = frozenset(
    {
        "and",
        "the",
        "for",
        "with",
        "from",
        "chain",
        "services",
        "service",
        "products",
        "product",
        "solutions",
        "management",
    }
)

# Generic business words that should not drive ranking on their own.
_WEAK_EXPANSION_TOKENS = frozenset(
    {
        "supply",
        "chain",
        "distribution",
        "port",
        "retail",
        "wholesale",
        "trading",
        "consumer",
        "industrial",
        "commercial",
        "enterprise",
        "business",
        "general",
        "other",
    }
)


def context_blob(row: dict) -> str:
    parts = [
        row.get("operational_name") or "",
        row.get("description") or "",
        row.get("naics_label") or "",
        " ".join(row.get("business_model") or []),
        " ".join(row.get("target_markets") or []),
        " ".join(row.get("core_offerings") or []),
    ]
    return " ".join(p for p in parts if p)


def field_texts(row: dict) -> dict[str, str]:
    return {
        "markets": " ".join(row.get("target_markets") or []),
        "naics": row.get("naics_label") or "",
        "offerings": " ".join(row.get("core_offerings") or []),
        "description": row.get("description") or "",
    }


def _tokens(text: str) -> list[str]:
    return [
        t
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) > 2 and t not in _LEX_STOP
    ]


def _token_in_blob(token: str, blob: str) -> bool:
    if token in blob:
        return True
    # logistics ↔ logistic, pharmaceuticals ↔ pharmaceutical
    if len(token) >= 5:
        stem = token.rstrip("s")
        if stem in blob:
            return True
    return False


def _phrase_strength(query: str, blob: str) -> float:
    q = query.lower().strip()
    if not q or not blob:
        return 0.0
    blob_l = blob.lower()
    if q in blob_l:
        return 1.0
    # Compound forms: logistics ⊂ intralogistics
    if len(q) >= 5 and any(q in tok for tok in re.findall(r"[a-z0-9]+", blob_l) if len(tok) > len(q)):
        return 0.85
    parts = _tokens(q)
    if parts and all(_token_in_blob(p, blob_l) for p in parts):
        return 0.65
    if parts:
        return min(sum(1 for p in parts if _token_in_blob(p, blob_l)) / len(parts), 1.0) * 0.45
    return 0.0


def _rows_fingerprint(rows: list[dict]) -> str:
    payload = [field_texts(r) for r in rows]
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class EmbeddingIndex:
    def __init__(
        self,
        rows: list[dict],
        cache_dir: Path | None = None,
        model_name: str = DEFAULT_MODEL,
        rebuild: bool = False,
        min_score: float = 0.28,
    ):
        self.rows = rows
        self.model_name = model_name
        self.min_score = min_score
        self.cache_dir = cache_dir or EMBEDDING_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self.matrices = self._load_or_build(rebuild=rebuild)

    def _cache_paths(self) -> tuple[Path, Path]:
        fp = _rows_fingerprint(self.rows)
        safe_model = self.model_name.replace("/", "_")
        base = self.cache_dir / f"{safe_model}_{fp}_fields_v2"
        return base.with_suffix(".npz"), base.with_suffix(".meta.json")

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    local_files_only=True,
                )
            except Exception:
                self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        # Empty strings get a zero vector so they never win via max()
        clean = [t if t.strip() else "" for t in texts]
        vectors = model.encode(
            [t if t else " " for t in clean],
            normalize_embeddings=True,
            show_progress_bar=len(clean) > 50,
            convert_to_numpy=True,
        ).astype(np.float32)
        for i, t in enumerate(clean):
            if not t:
                vectors[i] = 0.0
        return vectors

    def _load_or_build(self, rebuild: bool = False) -> dict[str, np.ndarray]:
        npz_path, meta_path = self._cache_paths()
        if not rebuild and npz_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("n_rows") == len(self.rows)
                and meta.get("model") == self.model_name
                and meta.get("fields") == list(FIELDS)
                and meta.get("version") == 2
            ):
                data = np.load(npz_path)
                return {field: data[field] for field in FIELDS}

        per_field: dict[str, list[str]] = {field: [] for field in FIELDS}
        for row in self.rows:
            texts = field_texts(row)
            for field in FIELDS:
                per_field[field].append(texts[field])

        matrices = {field: self._encode(per_field[field]) for field in FIELDS}
        np.savez_compressed(npz_path, **matrices)
        meta_path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "model": self.model_name,
                    "n_rows": len(self.rows),
                    "fields": list(FIELDS),
                    "dim": int(next(iter(matrices.values())).shape[1]),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return matrices

    def _field_similarity(
        self,
        field: str,
        indices: list[int],
        qvecs: np.ndarray,
        *,
        primary_weight: float = 0.72,
    ) -> np.ndarray:
        """Primary theme vector dominates; expansions only refine."""
        mat = self.matrices[field][indices]
        primary = mat @ qvecs[0]
        if len(qvecs) == 1:
            return primary
        expansion = (mat @ qvecs[1:].T).max(axis=1)
        return primary_weight * primary + (1.0 - primary_weight) * expansion

    def _lexical_scores(
        self,
        queries: list[str],
        indices: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (lex_score, anchor_strength).
        anchor_strength measures primary-theme match in markets/naics/offerings only.
        """
        primary = queries[0]
        primary_tokens = _tokens(primary)
        expansion_tokens = [
            t for t in _tokens(" ".join(queries[1:]))
            if t not in _WEAK_EXPANSION_TOKENS
        ]

        lex = np.zeros(len(indices), dtype=np.float32)
        anchor = np.zeros(len(indices), dtype=np.float32)

        for i, row_idx in enumerate(indices):
            texts = field_texts(self.rows[row_idx])
            thematic = " ".join(texts[f] for f in THEMATIC_FIELDS).lower()
            desc = (texts["description"] or "").lower()

            anchor_score = _phrase_strength(primary, thematic)
            if primary_tokens and anchor_score < 0.35:
                tok_hits = sum(1 for t in primary_tokens if _token_in_blob(t, thematic))
                anchor_score = max(anchor_score, tok_hits / len(primary_tokens) * 0.7)

            # Strong expansion phrase in thematic fields also counts as anchor
            for extra in queries[1:3]:
                anchor_score = max(anchor_score, _phrase_strength(extra, thematic) * 0.85)

            anchor[i] = float(anchor_score)

            exp_score = 0.0
            if expansion_tokens:
                exp_hits = sum(1 for t in expansion_tokens if _token_in_blob(t, thematic))
                exp_score = min(exp_hits / len(expansion_tokens), 1.0) * 0.45

            # NAICS label is high-precision for industry fit
            naics_bonus = 0.0
            naics_l = (texts["naics"] or "").lower()
            if primary_tokens and any(_token_in_blob(t, naics_l) for t in primary_tokens):
                naics_bonus = 0.25
            elif _phrase_strength(primary, naics_l) >= 0.65:
                naics_bonus = 0.35

            score = min(anchor_score * 0.78 + exp_score * 0.22 + naics_bonus, 1.0)

            # Description-only match: weak thematic signal but keyword in long text
            thematic_hit = anchor_score >= 0.35 or exp_score >= 0.25
            desc_hit = bool(primary_tokens) and any(_token_in_blob(t, desc) for t in primary_tokens)
            if desc_hit and not thematic_hit:
                score *= 0.45

            lex[i] = score

        return lex, anchor

    def search(
        self,
        prompt: str | list[str],
        indices: list[int] | None = None,
        limit: int = 10,
        min_score: float | None = None,
        recall_mode: bool = False,
    ) -> list[tuple[float, dict[str, Any]]]:
        """
        Rank by a composite score, not raw max alone.

        `prompt` may be one string or several embedding phrases from the LLM.
        The first phrase is the primary theme and is weighted highest.

        When `recall_mode=True`, use broad cosine similarity only (no lexical
        gating or employee tie-breaks) to maximize candidate recall before reranking.
        """
        threshold = self.min_score if min_score is None else min_score
        if isinstance(prompt, str):
            queries = [prompt.strip()] if prompt.strip() else []
        else:
            queries = [q.strip() for q in prompt if isinstance(q, str) and q.strip()]
        if not queries:
            return []

        model = self._get_model()
        qvecs = model.encode(
            queries,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        if indices is None:
            indices = list(range(len(self.rows)))
        if not indices:
            return []

        markets = self._field_similarity("markets", indices, qvecs)
        naics = self._field_similarity("naics", indices, qvecs)
        offerings = self._field_similarity("offerings", indices, qvecs)
        description = self._field_similarity("description", indices, qvecs)

        if recall_mode:
            stacked = np.stack([markets, naics, offerings, description], axis=0)
            scores = stacked.max(axis=0)
            order = np.argsort(-scores)
        else:
            stacked = np.stack([markets, naics, offerings], axis=0)
            thematic_max = stacked.max(axis=0)
            thematic_mean = stacked.mean(axis=0)

            # Only trust description when thematic fields already show some fit
            desc_gate = np.minimum(thematic_max / 0.32, 1.0)
            description_boost = description * desc_gate

            lex, anchor = self._lexical_scores(queries, indices)

            # Final blend: thematic fields + lexical anchor; expansions cannot override primary
            scores = (
                0.44 * thematic_max
                + 0.20 * thematic_mean
                + 0.06 * description_boost
                + 0.30 * lex
            )
            # Down-rank when primary theme is absent from thematic fields
            scores *= 0.55 + 0.45 * anchor

            emp = np.array(
                [float(self.rows[idx].get("employee_count") or 0) for idx in indices],
                dtype=np.float64,
            )
            order = np.lexsort((-emp, -thematic_max, -anchor, -scores))

        results: list[tuple[float, dict[str, Any]]] = []
        for pos in order:
            score = float(scores[pos])
            if score < threshold:
                break
            results.append((score, self.rows[indices[pos]]))
            if len(results) >= limit:
                break
        return results

    def field_embed_scores(
        self,
        queries: list[str],
        rows: list[dict[str, Any]],
        *,
        primary_weight: float = 0.75,
    ) -> list[dict[str, float]]:
        """
        Per-candidate cosine similarity of the query against each cached field vector.

        Used by the reranker so meaning match (not only token overlap) can rank
        offerings / NAICS / markets / description.
        """
        clean = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        if not clean or not rows:
            return [{"embed_total": 0.0} for _ in rows]

        qvecs = self._encode(clean)
        id_to_idx = {id(r): i for i, r in enumerate(self.rows)}
        # Fallback key when object identity differs
        key_to_idx = {
            (
                (r.get("website") or "").lower(),
                (r.get("operational_name") or "").lower(),
            ): i
            for i, r in enumerate(self.rows)
        }

        # Prefer thematic fields; description helps paraphrase matches.
        weights = {
            "offerings": 0.34,
            "naics": 0.28,
            "markets": 0.24,
            "description": 0.14,
        }
        out: list[dict[str, float]] = []
        for row in rows:
            idx = id_to_idx.get(id(row))
            if idx is None:
                idx = key_to_idx.get(
                    (
                        (row.get("website") or "").lower(),
                        (row.get("operational_name") or "").lower(),
                    )
                )
            if idx is None:
                out.append({"embed_total": 0.0})
                continue

            field_scores: dict[str, float] = {}
            total = 0.0
            for field, weight in weights.items():
                mat = self.matrices[field]
                primary = float(mat[idx] @ qvecs[0])
                if len(qvecs) > 1:
                    expansion = float(np.max(mat[idx] @ qvecs[1:].T))
                    sim = primary_weight * primary + (1.0 - primary_weight) * expansion
                else:
                    sim = primary
                # Cosine on normalized vectors is in [-1, 1]; clamp to [0, 1]
                sim = max(0.0, min(1.0, sim))
                field_scores[f"embed_{field}"] = round(sim, 4)
                total += weight * sim
            field_scores["embed_total"] = round(total, 4)
            out.append(field_scores)
        return out
