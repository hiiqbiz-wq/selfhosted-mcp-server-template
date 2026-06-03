"""Cross-encoder reranker client for the HIIQ memory gateway (Phase 4).

Pipeline position: hybrid_recall produces a top-20 candidate set; this reranks
those (query, doc) pairs with a qwen3 cross-encoder reranker served by Ollama on
the 5060 (HIIQ-RTX-5060) over Tailscale, and returns the top-3 — positioned at
the start of the turn with a Primary/Additional structure (anti lost-in-middle).

Reranker call shape (web-verified 2026-05-28):
  Ollama has **no native `/api/rerank` endpoint** (confirmed across the Ollama
  docs + community). The validated pattern for the Qwen3-Reranker GGUF is to call
  the chat endpoint `POST {host}/api/chat` with a binary relevance-grading prompt
  ("Is this document relevant to the query? Reply Yes or No.") at temperature 0,
  then map the textual answer to a score (Yes -> 1.0, else 0.0).
  Source: https://apidog.com/blog/qwen-3-embedding-reranker-ollama/
  (cross-checked against https://ollama.com/AuditAid/Reranker_v2 for the 4B Q5_K_M
  tag). The exact scoring call is isolated in `_score_pair` and clearly marked so
  it can be swapped if a future Ollama build adds a real /api/rerank route.

Operational prerequisite (gated follow-up):
  The model named by RERANK_MODEL (default `qwen3-reranker:4b`) MUST be pulled on
  the 5060 first:  `ollama pull qwen3-reranker:4b`  (or the AuditAid GGUF tag).
  Until then the host returns 404/model-not-found and this client degrades
  gracefully (see below).

Graceful degradation (hard requirement):
  If the reranker is unreachable, errors, or returns nothing usable, `.rerank`
  returns `docs[:top_k]` UNCHANGED and logs a warning. It never raises — recall
  must keep working even when the 5060 is asleep or the model isn't pulled.

Dependencies: httpx only (already in requirements.txt). Hosts/models from env;
no hardcoded values beyond the documented Tailscale defaults.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Default host = HIIQ-RTX-5060 over Tailscale (100.90.91.72), Ollama's :11434.
# The inter-rig ethernet bridge (10.10.10.1:11434) is lower-latency but not
# assumed here — set RERANK_HOST to it when the gateway runs on the 5070.
_DEFAULT_HOST = "http://100.90.91.72:11434"
_DEFAULT_MODEL = "qwen3-reranker:4b"

# Per-pair grading prompt. Kept binary/deterministic — the verified Ollama
# pattern grades relevance Yes/No rather than emitting a continuous logit.
_GRADER_SYSTEM = (
    "You are an expert relevance grader. Decide whether the document is "
    "relevant to the user's query. Answer with exactly one word: Yes or No."
)
_GRADER_USER_TEMPLATE = "Query: {query}\n\nDocument: {document}\n\nRelevant?"

# How much document text to send per pair. Reranker context is finite and most
# signal is up front; truncating keeps latency bounded across 20 candidates.
_DOC_CHAR_BUDGET = 2000


class Reranker:
    """Score (query, doc) pairs with an Ollama-served qwen3 cross-encoder.

    Construct once and reuse — it holds a pooled httpx.Client. Safe to leave
    unconfigured: if the host is down or the model isn't pulled, every call
    degrades to a pass-through slice.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        *,
        timeout: float = 20.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        """Args:
        host: Ollama base URL. Defaults to env RERANK_HOST, then the 5060
              Tailscale default.
        model: Reranker model tag. Defaults to env RERANK_MODEL, then
               'qwen3-reranker:4b'.
        timeout: Per-request timeout (seconds). Reranking is per-pair, so this
                 bounds a single grade, not the whole batch.
        client: Optional pre-built httpx.Client (for tests / shared pools). If
                omitted, one is created and owned by this instance.
        """
        self.host = (host or os.environ.get("RERANK_HOST") or _DEFAULT_HOST).rstrip("/")
        self.model = model or os.environ.get("RERANK_MODEL") or _DEFAULT_MODEL
        self.timeout = float(timeout)
        self._client = client
        self._owns_client = client is None

    # -- lifecycle ---------------------------------------------------------

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self) -> None:
        """Close the owned httpx.Client. No-op if a client was injected."""
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def __enter__(self) -> "Reranker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- scoring (the version-dependent Ollama call lives here) ------------

    def _score_pair(self, query: str, document: str) -> Optional[float]:
        """Return a relevance score for one (query, doc) pair, or None on failure.

        >>> THE OLLAMA RERANKER CALL <<<  (swap here if a native /api/rerank
        endpoint becomes available). Uses POST {host}/api/chat with a Yes/No
        grading prompt at temperature 0, per the web-verified pattern. Returns
        1.0 for a 'yes' answer, 0.0 for a clear 'no', None if the call/parse
        fails (so the caller can fall back rather than mis-rank on a zero).
        """
        doc = (document or "")[:_DOC_CHAR_BUDGET]
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _GRADER_SYSTEM},
                {
                    "role": "user",
                    "content": _GRADER_USER_TEMPLATE.format(query=query, document=doc),
                },
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        try:
            resp = self._get_client().post(f"{self.host}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            answer = (data.get("message", {}) or {}).get("content", "")
            answer = answer.strip().lower()
            if not answer:
                return None
            if "yes" in answer:
                return 1.0
            if "no" in answer:
                return 0.0
            # Unrecognized output — treat as a soft miss, not a hard zero.
            return None
        except (httpx.HTTPError, ValueError, KeyError) as e:
            # Network failure, non-2xx, model-not-pulled (404), bad JSON.
            logger.warning(
                "reranker _score_pair failed (host=%s model=%s): %s",
                self.host, self.model, str(e)[:200],
            )
            return None

    # -- public API --------------------------------------------------------

    def rerank(
        self,
        query: str,
        docs: list[dict],
        *,
        text_key: str = "content",
        top_k: int = 3,
    ) -> list[dict]:
        """Rerank `docs` by cross-encoder relevance to `query`; return top_k.

        Args:
            query: The user query.
            docs: Candidate rows (e.g. hybrid_recall output). Each is a dict; the
                  text scored is docs[i][text_key] (falls back to 'summary' then
                  '' if absent). Dicts are returned as-is plus a 'rerank_score'.
            text_key: Which field holds the scored text. Default 'content'.
            top_k: How many to return. Default 3 (the gateway's Primary/Additional
                   budget). Clamped to >= 1.

        Returns:
            The top_k docs by descending rerank_score, stable on the original
            hybrid order for ties. On ANY reranker failure (unreachable host,
            unpulled model, all-None scores) returns docs[:top_k] UNCHANGED and
            logs a warning — never raises.
        """
        top_k = max(1, int(top_k))
        if not docs:
            return []
        if not (query and query.strip()):
            # Nothing to grade against; preserve incoming order.
            return docs[:top_k]

        scored: list[tuple[float, int, dict]] = []
        any_scored = False
        for idx, doc in enumerate(docs):
            text = doc.get(text_key) or doc.get("summary") or ""
            score = self._score_pair(query, text) if text else None
            if score is not None:
                any_scored = True
            # idx is the tie-breaker so equal scores keep hybrid (RRF) order.
            scored.append((score if score is not None else float("-inf"), idx, doc))

        if not any_scored:
            # Reranker contributed nothing (host down / model missing). Degrade.
            logger.warning(
                "reranker returned no usable scores; falling back to hybrid "
                "top-%d unchanged (host=%s model=%s)",
                top_k, self.host, self.model,
            )
            return docs[:top_k]

        # Higher score first; lower original index wins ties (negated idx).
        scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)

        out: list[dict] = []
        for score, _idx, doc in scored[:top_k]:
            enriched = dict(doc)
            enriched["rerank_score"] = None if score == float("-inf") else float(score)
            out.append(enriched)
        return out
