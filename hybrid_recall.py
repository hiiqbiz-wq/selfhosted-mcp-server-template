"""Hybrid recall for the HIIQ memory gateway (Phase 4 — retrieval precision).

Fuses full-text and vector search over a PACOM table whose rows carry both a
`content_tsv tsvector` (FTS) and an `embedding vector` (pgvector cosine). On the
live `memories` table the embedding is `vector(1024)` (qwen3 MRL-truncated, the
dim the 5060 embedder + recall.py emit); the cb_* tables additionally keep a
legacy `embedding vector(3072)` column. The SQL is dimension-agnostic — it casts
the bound query vector with `::vector` and lets pgvector match the column — so
the only contract is that `query_embedding`'s length equals the target column's.
The two result sets are merged with Reciprocal Rank Fusion (RRF) so neither
modality has to be score-normalized against the other — only ranks matter.

Pipeline position: this produces the hybrid top-N (default 20). The orchestrator
hands those rows to `reranker_client.Reranker.rerank(...)` to get the final top-3.

Design constraints (simple-first):
  * Standalone module. No gateway imports, no global connection. The caller owns
    the psycopg2 connection AND computes `query_embedding` (this module never
    talks to Ollama or PACOM itself).
  * Metadata pre-filters (scope / status / memory_type / sensitivity) are applied
    inside BOTH CTE WHERE clauses *before* ranking, so RRF ranks reflect only the
    eligible candidate set — not a post-hoc trim.
  * Parametrized SQL only. No f-string interpolation of user values; the table
    name is the single identifier substitution and is validated against an
    allow-list.

Expected table columns (mirrors the `memories` table in remote-seo-checker.py;
the migrated `knowledge_base` / `semantic_memories` tables are aligned to match
in Phase 3):
    id              uuid / text   primary key
    content         text          full body
    summary         text          short form (nullable)
    memory_type     text          e.g. 'lesson', 'fact', 'incident'
    scope           text          e.g. 'global', 'project:khs'
    author          text
    ts              timestamptz    creation / event time
    status          text          'pending' | 'approved' | 'archived'
    sensitivity     text          'public' | 'private' | 'sensitive' | 'secret'
    confidence      numeric        (nullable)
    tags            text[]         (nullable)
    archived_at     timestamptz    NULL unless archived
    content_tsv     tsvector       generated FTS column
    embedding       vector(N)      pgvector cosine target (N=1024 on `memories`)

Returned rows are RealDict-shaped (one dict per row) carrying every selected
column plus:
    rrf_score   float    fused score (higher = better; sum of 1/(rrf_k+rank))
    fts_rank    int|None rank in the full-text CTE (1-based; None if absent)
    vec_rank    int|None rank in the vector CTE (1-based; None if absent)
"""

from __future__ import annotations

from typing import Any, Optional

from psycopg2.extras import RealDictCursor

# Tables this module is permitted to query. The table name reaches SQL as an
# identifier (not a bound parameter), so it must come from a closed set —
# never accept an arbitrary caller-supplied name into the FROM clause.
ALLOWED_TABLES = frozenset({"memories", "knowledge_base", "semantic_memories"})

# Filter keys we know how to translate into equality predicates. Anything else
# in `filters` is ignored (logged by the caller if it cares) rather than spliced
# blindly into SQL. All are plain columns on the memories table; values are
# bound (scalar = equality, list/tuple/set = ANY), never interpolated.
_FILTERABLE_COLUMNS = ("scope", "status", "memory_type", "sensitivity", "author")

# How wide each modality reaches before fusion. Pulling a generous per-arm
# candidate pool (relative to the final `limit`) is what lets RRF recover
# documents that rank mid-pack in one modality but high in the other.
_PER_ARM_MULTIPLIER = 4
_PER_ARM_FLOOR = 50


def _build_filter_clause(filters: Optional[dict]) -> tuple[str, dict]:
    """Translate the metadata filter dict into a SQL fragment + bound params.

    Returns ('' , {}) when there is nothing to filter on. The fragment, when
    present, is a leading-AND string ready to append inside a WHERE that already
    has at least one predicate (the base `archived_at IS NULL`).
    """
    if not filters:
        return "", {}

    clauses: list[str] = []
    params: dict[str, Any] = {}
    for col in _FILTERABLE_COLUMNS:
        if col in filters and filters[col] is not None:
            value = filters[col]
            pname = f"f_{col}"
            if isinstance(value, (list, tuple, set)):
                # Allow multi-value filters (e.g. sensitivity in a tier list).
                clauses.append(f"{col} = ANY(%({pname})s)")
                params[pname] = list(value)
            else:
                clauses.append(f"{col} = %({pname})s")
                params[pname] = value

    if not clauses:
        return "", {}
    return " AND " + " AND ".join(clauses), params


def build_hybrid_sql(table: str, filter_clause: str) -> str:
    """Return the parametrized hybrid-recall SQL for `table`.

    Exposed separately so the orchestrator (or a test) can inspect / EXPLAIN the
    exact statement without executing it. `table` MUST already be validated
    against ALLOWED_TABLES; `filter_clause` is the fragment from
    `_build_filter_clause` (trusted — built from a column allow-list, values are
    still bound, not interpolated).

    Named bind params expected by the statement:
        %(q)s        query text          -> plainto_tsquery('english', q)
        %(emb)s      query embedding     -> cast ::vector for cosine distance
        %(per_arm)s  per-modality LIMIT  (candidate pool depth)
        %(rrf_k)s    RRF constant        (typically 60)
        %(limit)s    final row count
        + any %(f_<col>)s metadata params from the filter clause
    """
    # NOTE: every value is a bound parameter. The only interpolations are the
    # validated table identifier and the trusted, allow-list-built filter clause.
    return f"""
    WITH fts AS (
        SELECT
            id,
            ts_rank_cd(content_tsv, plainto_tsquery('english', %(q)s)) AS score,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank_cd(content_tsv, plainto_tsquery('english', %(q)s)) DESC
            ) AS rank
        FROM {table}
        WHERE archived_at IS NULL
          AND content_tsv @@ plainto_tsquery('english', %(q)s){filter_clause}
        ORDER BY score DESC
        LIMIT %(per_arm)s
    ),
    vec AS (
        SELECT
            id,
            (embedding <=> %(emb)s::vector) AS distance,
            ROW_NUMBER() OVER (
                ORDER BY embedding <=> %(emb)s::vector ASC
            ) AS rank
        FROM {table}
        WHERE archived_at IS NULL
          AND embedding IS NOT NULL{filter_clause}
        ORDER BY distance ASC
        LIMIT %(per_arm)s
    ),
    fused AS (
        SELECT
            COALESCE(fts.id, vec.id) AS id,
            fts.rank AS fts_rank,
            vec.rank AS vec_rank,
            COALESCE(1.0 / (%(rrf_k)s + fts.rank), 0.0)
              + COALESCE(1.0 / (%(rrf_k)s + vec.rank), 0.0) AS rrf_score
        FROM fts
        FULL OUTER JOIN vec ON fts.id = vec.id
    )
    SELECT
        m.id, m.content, m.summary, m.memory_type, m.scope, m.author,
        m.ts, m.status, m.sensitivity, m.confidence, m.tags,
        fused.fts_rank,
        fused.vec_rank,
        fused.rrf_score
    FROM fused
    JOIN {table} m ON m.id = fused.id
    ORDER BY fused.rrf_score DESC, m.ts DESC
    LIMIT %(limit)s
    """


def hybrid_recall(
    conn,
    query: str,
    query_embedding,
    *,
    table: str = "memories",
    limit: int = 20,
    filters: Optional[dict] = None,
    rrf_k: int = 60,
) -> list[dict]:
    """Fuse full-text and vector recall over `table` via Reciprocal Rank Fusion.

    Args:
        conn: An open psycopg2 connection (caller-owned; not closed here).
        query: Natural-language query. Drives `plainto_tsquery('english', ...)`.
                Required and non-empty — hybrid recall needs an FTS arm. For the
                no-query "most recent" path, call the gateway's plain list path.
        query_embedding: The query vector, precomputed by the caller. A
                list[float] whose length MATCHES the target column's dimension
                (1024 for `memories`); psycopg2 adapts a Python list to a
                pgvector literal via the ::vector cast. A pgvector-ready type is
                also accepted.
        table: One of ALLOWED_TABLES. Defaults to 'memories'.
        limit: Final row count after fusion. Default 20 (the hybrid top-N feeding
                the reranker). Clamped to 1..100.
        filters: Optional metadata pre-filters applied in BOTH CTEs before
                ranking. Recognized keys: scope, status, memory_type,
                sensitivity. Values may be scalars or lists (ANY match).
        rrf_k: RRF dampening constant. Default 60 (the value from the original
                Cormack et al. RRF paper; larger flattens rank influence).

    Returns:
        Up to `limit` RealDict rows ordered by descending rrf_score, each
        carrying the selected columns plus rrf_score / fts_rank / vec_rank.

    Raises:
        ValueError: if `table` is not allow-listed, `query` is empty, or
                    `query_embedding` is missing. (Connection / SQL errors from
                    psycopg2 propagate unchanged — the caller decides retry vs.
                    fail.)
    """
    if table not in ALLOWED_TABLES:
        raise ValueError(
            f"table {table!r} not in allow-list {sorted(ALLOWED_TABLES)}"
        )
    if not (query and query.strip()):
        raise ValueError("hybrid_recall requires a non-empty query (FTS arm)")
    if query_embedding is None:
        raise ValueError("hybrid_recall requires a precomputed query_embedding")

    limit = max(1, min(100, int(limit)))
    rrf_k = max(1, int(rrf_k))
    per_arm = max(_PER_ARM_FLOOR, limit * _PER_ARM_MULTIPLIER)

    filter_clause, filter_params = _build_filter_clause(filters)
    sql = build_hybrid_sql(table, filter_clause)

    params: dict[str, Any] = {
        "q": query,
        "emb": query_embedding,
        "per_arm": per_arm,
        "rrf_k": rrf_k,
        "limit": limit,
    }
    params.update(filter_params)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]

    # Normalize id -> str for transport parity with the gateway's recall_memory.
    for r in rows:
        if r.get("id") is not None:
            r["id"] = str(r["id"])
        if r.get("rrf_score") is not None:
            r["rrf_score"] = float(r["rrf_score"])
    return rows
