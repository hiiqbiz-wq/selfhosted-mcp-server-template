"""
HIIQ Edge MCP gateway v5.4 — adds Chatterbox TTS proxy tools (text-to-speech
via hiiq-andre's RTX 5060) on top of v5.3 docling document conversion.

Runs on the Hostinger VPS (mcp.hiiqbiz.com). Reaches Pacific PACOM at
100.87.218.106:5430 and the hiiq-andre TTS service at 100.90.91.72:8019
via the host's Tailscale interface. Reaches the sibling docling-serve
container via the internal Coolify docker network.

Auth: GitHub OAuth via FastMCP's GitHubProvider (DCR-compliant, Claude.ai-friendly).
Per-tool gate: require_authorized() checks login against ALLOWED_GH_USERS.

Tools (21 total):
  Existing (Phase 4.1.x):
    ping_hiiq, pacom_tables, pacom_recent_cli, query_pacom,
    search_vault, pacom_skills, pacom_plugins, session_resume

  Phase 4.7 — bilateral memory:
    add_memory, recall_memory, list_memories,
    verify_memory (Andre-only), archive_memory (Andre-only),
    recall_sensitive_memory (stub pending 4.7.1)

  v5.3 — Docling proxy:
    docling_health      — probe sibling docling-serve container
    convert_document    — synchronous PDF/DOCX/PPTX/XLSX/HTML/image → Markdown
                          via internal docling-serve. B3 architecture: docling-serve
                          has zero auth and is reachable only on the private docker
                          network; this gateway is the public OAuth-gated surface.

  New (v5.4 — Chatterbox TTS proxy):
    tts_health          — probe hiiq-andre TTS service /health
    tts_list_voices     — list 10-sec reference clips on hiiq-andre
    tts_save_voice      — upload a new reference clip (base64 WAV)
    tts_delete_voice    — remove a reference clip
    tts_generate        — text → base64-encoded WAV via Chatterbox Turbo /
                          Multilingual / Regular on RTX 5060. Reached over
                          Tailscale; gateway forwards bearer token for auth.
"""

import os
import re
import datetime
import json
import httpx
import psycopg2
from typing import Optional
from psycopg2.extras import RealDictCursor, Json
from fastmcp import FastMCP
from fastmcp.server.auth.providers.github import GitHubProvider
from fastmcp.server.dependencies import get_access_token, get_http_request

# --- PACOM connection ---
PACOM_PG_HOST = os.environ.get("PACOM_PG_HOST", "100.87.218.106")
PACOM_PG_PORT = int(os.environ.get("PACOM_PG_PORT", "5430"))
PACOM_PG_DBNAME = os.environ.get("PACOM_PG_DBNAME", "pacom")
PACOM_PG_USER = os.environ.get("PACOM_PG_USER", "postgres")
PACOM_PG_PASSWORD = os.environ.get("PACOM_PG_PASSWORD", "")

# --- Docling sibling container (B3 architecture, v5.3) ---
# docling-serve runs as a private Coolify container on the same docker network.
# It has zero auth — reachable only inside the network. This gateway is the
# public OAuth-gated surface; convert_document() proxies to it via httpx.
DOCLING_SERVICE_URL = os.environ.get("DOCLING_SERVICE_URL", "http://docling-serve:5001").rstrip("/")
DOCLING_HTTP_TIMEOUT = float(os.environ.get("DOCLING_HTTP_TIMEOUT", "180"))

# --- TTS service on hiiq-andre via Tailscale (v5.4) ---
# Chatterbox TTS (Resemble AI, MIT) runs on hiiq-andre's RTX 5060. Reached
# via Tailscale (hiiqbiz-vps has Tailscale; same path used for PACOM at
# 100.87.218.106). Unlike docling-serve, the TTS service requires bearer
# auth — the gateway forwards CB_MCP_AUTH_TOKEN (HIIQ MCP discipline) so
# the local service can verify the request came from an authorized client.
TTS_SERVICE_URL = os.environ.get("TTS_SERVICE_URL", "http://100.90.91.72:8019").rstrip("/")
TTS_HTTP_TIMEOUT = float(os.environ.get("TTS_HTTP_TIMEOUT", "120"))
TTS_SERVICE_TOKEN = (
    os.environ.get("TTS_SERVICE_TOKEN")
    or os.environ.get("CB_MCP_AUTH_TOKEN", "")
).strip()
TTS_VOICE_LABEL_REGEX = re.compile(r"^[a-z0-9][a-z0-9_-]{0,30}$")

# --- OAuth ---
GITHUB_CLIENT_ID = os.environ.get("GITHUB_OAUTH_CLIENT_ID", "").strip()
GITHUB_CLIENT_SECRET = os.environ.get("GITHUB_OAUTH_CLIENT_SECRET", "").strip()
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "https://mcp.hiiqbiz.com").rstrip("/")
ALLOWED_GH_USERS = {
    u.strip()
    for u in os.environ.get("ALLOWED_GH_USERS", "hiiqbiz-wq").split(",")
    if u.strip()
}
ANDRE_GH_LOGIN = os.environ.get("ANDRE_GH_LOGIN", "hiiqbiz-wq").strip()

# --- Per-rig attribution (Phase 4.7.2 / v5.2) ---
# Clients (Pacific Claude Code, Central Claude Code, etc.) send this header to
# stamp memory writes with the originating rig. Validated against RIG_REGEX +
# optional HIIQ_RIG_ALLOWLIST. Empty / missing / invalid → rig is None and
# attribution falls back to surface-only (e.g. 'claude-code').
RIG_HEADER_NAME = "x-hiiq-rig"  # HTTP headers are case-insensitive; Starlette lowercases
RIG_REGEX = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
HIIQ_RIG_ALLOWLIST = {
    r.strip().lower()
    for r in os.environ.get("HIIQ_RIG_ALLOWLIST", "").split(",")
    if r.strip()
}  # empty set = no allowlist enforcement; any regex-valid rig passes


if GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET:
    auth = GitHubProvider(
        client_id=GITHUB_CLIENT_ID,
        client_secret=GITHUB_CLIENT_SECRET,
        base_url=PUBLIC_BASE_URL,
        redirect_path="/auth/callback",
        required_scopes=["read:user"],
        allowed_client_redirect_uris=[
            "https://claude.ai/api/mcp/auth_callback",
            "https://claude.com/api/mcp/auth_callback",
            "http://localhost:*",
        ],
    )
    mcp = FastMCP(name="HIIQ Edge", auth=auth)
    print(
        f"[boot] GitHub OAuth enabled. Allowlist: {sorted(ALLOWED_GH_USERS)}",
        flush=True,
    )
else:
    mcp = FastMCP(name="HIIQ Edge")
    print(
        "[boot] WARNING: GITHUB_OAUTH_CLIENT_ID/SECRET not set — gateway is OPEN.",
        flush=True,
    )


def require_authorized() -> str:
    """Return GitHub login if authed + allowlisted. Raise otherwise. Returns 'anonymous' if OAuth disabled."""
    if not (GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET):
        return "anonymous"
    token = get_access_token()
    if not token:
        raise PermissionError("No authenticated token")
    login = (token.claims.get("login") or "").strip()
    if login not in ALLOWED_GH_USERS:
        raise PermissionError(f"GitHub user '{login}' is not in the HIIQ Edge allowlist")
    return login


def require_andre() -> str:
    """Same as require_authorized but additionally requires the caller to BE Andre. Andre-only ops."""
    user = require_authorized()
    if user != ANDRE_GH_LOGIN:
        raise PermissionError(
            f"This operation requires '{ANDRE_GH_LOGIN}' (only Andre can modify/verify/archive)"
        )
    return user


def _classify_surface_from_ua(ua: str) -> str:
    """Return the surface tag matching this User-Agent string."""
    ua_lower = ua.lower()
    if "claude-code" in ua_lower or "claude_code" in ua_lower or "claudecode" in ua_lower:
        return "claude-code"
    if "claude-desktop" in ua_lower or "claude_desktop" in ua_lower or "claudedesktop" in ua_lower:
        return "claude-desktop"
    if "anthropicapp" in ua_lower or "claude-mobile" in ua_lower or "claudemobile" in ua_lower:
        return "claude-mobile"
    if "claude.ai" in ua_lower or "claudeai" in ua_lower:
        return "claudeai-web"
    if "claude" in ua_lower:
        return "claude"
    return "claude-unknown"


def _validate_rig(raw: str) -> str | None:
    """Return a normalized rig name if `raw` is well-formed + allowlisted, else None.

    Validation:
      - lowercased + stripped
      - matches RIG_REGEX (`^[a-z0-9][a-z0-9-]{0,31}$`)
      - if HIIQ_RIG_ALLOWLIST is non-empty, must be in it

    Prevents author-field injection from a malicious header value.
    """
    if not raw:
        return None
    rig = raw.strip().lower()
    if not RIG_REGEX.match(rig):
        return None
    if HIIQ_RIG_ALLOWLIST and rig not in HIIQ_RIG_ALLOWLIST:
        return None
    return rig


def detect_caller_surface() -> tuple[str, str, str | None]:
    """Classify the calling Claude surface + originating rig from HTTP headers.

    Stamps `author` on memory writes so cross-session/cross-rig coordination
    can disambiguate which Claude proposed which memory. The surface tag comes
    from User-Agent (claude-code / claude-desktop / claudeai-web / claude-mobile
    / claude / claude-unknown). The rig tag comes from the X-HIIQ-Rig header
    that each rig's Claude Code config injects — this is what tells Pacific
    Claude Code apart from Central Claude Code (their User-Agents are identical).

    Returns:
        (surface_tag, raw_user_agent, rig_name_or_none).
        - surface_tag: one of 'claude-code', 'claude-desktop', 'claudeai-web',
          'claude-mobile', 'claude', 'claude-unknown', 'no-http-ctx'.
        - raw_user_agent: the full UA string (for metadata stashing).
        - rig_name_or_none: the validated X-HIIQ-Rig value (e.g. 'pacific',
          'central', 'edge') if present + well-formed + allowlisted, else None.

    Composing `author`:
        f"{surface_tag}-{rig}" if rig else surface_tag
        e.g. 'claude-code-pacific', 'claude-code-central', or 'claude-code'
        for a session that didn't set the header (e.g. claude.ai web connector).
    """
    try:
        req = get_http_request()
    except RuntimeError:
        return ("no-http-ctx", "", None)
    ua = (req.headers.get("user-agent") or "")
    surface = _classify_surface_from_ua(ua)
    rig = _validate_rig(req.headers.get(RIG_HEADER_NAME) or "")
    return (surface, ua, rig)


def get_pg_conn(readonly: bool = True):
    """Open a Postgres connection to PACOM. Defaults to read-only session."""
    conn = psycopg2.connect(
        host=PACOM_PG_HOST,
        port=PACOM_PG_PORT,
        dbname=PACOM_PG_DBNAME,
        user=PACOM_PG_USER,
        password=PACOM_PG_PASSWORD,
        connect_timeout=10,
    )
    if readonly:
        conn.set_session(readonly=True)
    return conn


# ============================================================================
# Existing read tools (unchanged from v4)
# ============================================================================

@mcp.tool()
def ping_hiiq() -> dict:
    """Health check for the HIIQ Edge MCP gateway. Returns server identity,
    UTC timestamp, authenticated GitHub user, PACOM reachability, and tool list."""
    user = require_authorized()
    surface, _ua, rig = detect_caller_surface()
    out = {
        "status": "ok",
        "server": "HIIQ Edge MCP gateway",
        "version": "v5.4 (docling + TTS proxy)",
        "node": "hiiqbiz-vps (Hostinger KVM 4, US-Boston)",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "authenticated_as": user,
        "calling_surface": surface,
        "calling_rig": rig,  # None unless X-HIIQ-Rig header set + validated
        "pacom_reachable": False,
        "auth_enabled": bool(GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET),
    }
    try:
        with get_pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                if cur.fetchone()[0] == 1:
                    out["pacom_reachable"] = True
    except Exception as e:
        out["pacom_error"] = str(e)[:200]
    return out


@mcp.tool()
def pacom_tables() -> dict:
    """List all user tables in PACOM with row counts (from pg_stat_user_tables; may lag)."""
    require_authorized()
    sql = """
    SELECT schemaname || '.' || relname AS table_name, n_live_tup AS row_count
    FROM pg_stat_user_tables ORDER BY schemaname, relname
    """
    with get_pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            return {"tables": [dict(r) for r in cur.fetchall()]}


@mcp.tool()
def pacom_recent_cli(n: int = 20) -> dict:
    """Return the N most recent cli_audit entries (Pacific CLI activity)."""
    require_authorized()
    n = max(1, min(100, int(n)))
    sql = """
    SELECT ts, rig_hostname, tool_name, args, backend, endpoint,
           status, error_message, duration_ms
    FROM cli_audit ORDER BY ts DESC LIMIT %s
    """
    with get_pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (n,))
            return {
                "entries": [dict(r) for r in cur.fetchall()],
                "source": "PACOM cli_audit table",
            }


@mcp.tool()
def query_pacom(sql: str, max_rows: int = 50) -> dict:
    """Execute a read-only SQL SELECT against PACOM. READ ONLY session enforced at DB level."""
    require_authorized()
    max_rows = max(1, min(500, int(max_rows)))
    try:
        with get_pg_conn(readonly=True) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql)
                rows = cur.fetchmany(max_rows)
                cols = [d[0] for d in cur.description] if cur.description else []
                return {"columns": cols, "rows": [dict(r) for r in rows], "row_count": len(rows)}
    except Exception as e:
        return {"error": str(e)[:500]}


@mcp.tool()
def search_vault(query: str, limit: int = 10, scope: str = None) -> dict:
    """Full-text search the indexed Obsidian vault via PACOM vault_index (~2780 files)."""
    require_authorized()
    limit = max(1, min(50, int(limit)))
    if scope:
        sql = """
        SELECT path, scope, mtime,
               ts_rank(content_tsv, plainto_tsquery('english', %(q)s)) AS rank,
               ts_headline('english', content_text, plainto_tsquery('english', %(q)s),
                           'MaxFragments=2,MaxWords=40,MinWords=10') AS snippet
        FROM vault_index
        WHERE content_tsv @@ plainto_tsquery('english', %(q)s) AND scope = %(scope)s
        ORDER BY rank DESC LIMIT %(limit)s
        """
        params = {"q": query, "scope": scope, "limit": limit}
    else:
        sql = """
        SELECT path, scope, mtime,
               ts_rank(content_tsv, plainto_tsquery('english', %(q)s)) AS rank,
               ts_headline('english', content_text, plainto_tsquery('english', %(q)s),
                           'MaxFragments=2,MaxWords=40,MinWords=10') AS snippet
        FROM vault_index
        WHERE content_tsv @@ plainto_tsquery('english', %(q)s)
        ORDER BY rank DESC LIMIT %(limit)s
        """
        params = {"q": query, "limit": limit}
    with get_pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return {"hits": [dict(r) for r in cur.fetchall()], "query": query, "scope_filter": scope}


@mcp.tool()
def pacom_skills(query: str = None) -> dict:
    """List skills registered in PACOM (~27). Optional substring filter on name/description."""
    require_authorized()
    if query:
        sql = """SELECT name, path, scope, plugin_name, description, when_to_use, model
                 FROM skills_registry WHERE search_text ILIKE %s ORDER BY name"""
        params = (f"%{query}%",)
    else:
        sql = """SELECT name, path, scope, plugin_name, description, when_to_use, model
                 FROM skills_registry ORDER BY name"""
        params = ()
    with get_pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return {"skills": [dict(r) for r in cur.fetchall()]}


@mcp.tool()
def pacom_plugins() -> dict:
    """List Claude plugins registered in PACOM with capabilities + component counts."""
    require_authorized()
    sql = """SELECT name, version, description, author_name, install_path,
                    capabilities, skill_count, agent_count, command_count, hook_count, mcp_count
             FROM plugins_registry ORDER BY name"""
    with get_pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            return {"plugins": [dict(r) for r in cur.fetchall()]}


@mcp.tool()
def session_resume(only_unconsumed: bool = True, limit: int = 5) -> dict:
    """Get recent Claude session handoffs (saved via save-pacom CLI)."""
    require_authorized()
    limit = max(1, min(20, int(limit)))
    where = "WHERE consumed_at IS NULL" if only_unconsumed else ""
    sql = f"""SELECT session_id, rig_hostname, ended_at, last_command,
                     current_chapter, next_action, raw_handoff, consumed_at
              FROM session_handoff {where} ORDER BY ended_at DESC NULLS LAST LIMIT %s"""
    with get_pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (limit,))
            return {"handoffs": [dict(r) for r in cur.fetchall()]}


# ============================================================================
# Phase 4.7 — Bilateral memory tools
# ============================================================================

@mcp.tool()
def add_memory(
    content: str,
    memory_type: str = "observation",
    scope: str = "personal",
    tags: list = None,
    summary: str = None,
    sensitivity: str = "private",
    confidence: int = 5,
    decision_matrix_anchors: list = None,
    confidence_reasoning: str = None,
    related_ids: list = None,
    metadata: dict = None,
) -> dict:
    """
    Save a memory to PACOM. ALWAYS lands as `status='pending'` per Andre's
    governance gate — Andre verifies via verify_memory(id) to promote to
    'approved' and make permanent.

    Args:
        content: The memory itself (required, non-empty).
        memory_type: Free-form type — fact / decision / preference / observation /
                     event / todo / question / reference / etc. Default 'observation'.
        scope: Free-form scope — 'personal' / 'hiiq' / 'khs' / 'project:<name>' / etc.
                Default 'personal'.
        tags: List of tag strings. Optional.
        summary: One-line headline. Optional but recommended.
        sensitivity: 'public' / 'private' / 'sensitive'. 'secret' rejected in v5
                     (requires pgcrypto + passphrase wiring — coming in 4.7.1).
        confidence: 1..10. If >= 7 and author is Claude, decision_matrix_anchors
                    MUST be non-empty (forces citation of which Decision Matrix
                    rows justify this).
        decision_matrix_anchors: List of Decision Matrix 'shape:' strings that
                                  informed this memory. See Decision-Matrix.md.
        confidence_reasoning: Optional one-line "why this score".
        related_ids: List of UUIDs of related memories. Optional.
        metadata: Optional JSON dict for type-specific extras.

    Returns:
        dict with `id`, `status='pending'`, `created_at`, and a short reminder
        on how to verify.
    """
    user = require_authorized()
    if not content or not content.strip():
        return {"error": "content cannot be empty"}
    if sensitivity == "secret":
        return {
            "error": "sensitivity='secret' requires pgcrypto + passphrase wiring "
            "(Phase 4.7.1). Use 'sensitive' for now or wait."
        }
    if sensitivity not in ("public", "private", "sensitive"):
        return {"error": f"sensitivity must be one of public/private/sensitive (got {sensitivity!r})"}

    tags = list(tags or [])
    decision_matrix_anchors = list(decision_matrix_anchors or [])
    related_ids = list(related_ids or [])
    metadata = dict(metadata or {})

    # Author = which Claude surface (+ which rig, if known) proposed this
    # memory. All add_memory calls through this HTTP gateway are by definition
    # Claude doing the writing — Andre interacts via Claude surfaces, not via
    # raw curl. Andre's authority is expressed via verify_memory + archive_memory,
    # not via the writer field. Rig identity comes from the X-HIIQ-Rig header
    # each rig's Claude Code config injects; surface comes from User-Agent.
    surface, raw_ua, rig = detect_caller_surface()
    author = f"{surface}-{rig}" if rig else surface
    if raw_ua:
        metadata.setdefault("client_user_agent", raw_ua[:200])
    metadata.setdefault("authenticated_gh_login", user)
    if rig:
        metadata.setdefault("rig", rig)

    # Governance check: high-confidence Claude writes must cite Decision Matrix
    if confidence >= 7 and not decision_matrix_anchors:
        return {
            "error": "confidence >= 7 requires citing Decision Matrix anchors. "
            "Either lower confidence (Claude's default for unsupported claims) "
            "or cite at least one Decision-Matrix.md 'shape:' string."
        }
    sql = """
    INSERT INTO memories (
        content, summary, memory_type, scope, author,
        source_type, tags, related_ids,
        confidence, decision_matrix_anchors, confidence_reasoning,
        status, sensitivity, metadata
    ) VALUES (
        %s, %s, %s, %s, %s,
        'mcp-gateway', %s, %s,
        %s, %s, %s,
        'pending', %s, %s
    ) RETURNING id, ts, status
    """
    try:
        with get_pg_conn(readonly=False) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    content, summary, memory_type, scope, author,
                    tags, related_ids,
                    int(confidence), decision_matrix_anchors, confidence_reasoning,
                    sensitivity, Json(metadata),
                ))
                row = cur.fetchone()
                conn.commit()
                return {
                    "id": str(row[0]),
                    "status": row[2],
                    "created_at": row[1].isoformat() if row[1] else None,
                    "author": author,
                    "note": (
                        f"Memory saved as 'pending'. Andre can verify via "
                        f"verify_memory(id='{row[0]}') to mark permanent."
                    ),
                }
    except Exception as e:
        return {"error": str(e)[:500]}


@mcp.tool()
def recall_memory(
    query: str = "",
    limit: int = 10,
    memory_type: str = None,
    scope: str = None,
    author: str = None,
    sensitivity_max: str = "sensitive",
    include_pending: bool = True,
) -> dict:
    """
    Full-text search the memories table (vector/semantic search coming when
    embeddings are populated in Phase 4.8+).

    Args:
        query: Search terms (English plainto_tsquery). Empty string returns most recent.
        limit: 1..50, default 10.
        memory_type: Optional filter on type.
        scope: Optional filter on scope.
        author: Optional filter on author (e.g., 'andre', 'claude-via-hiiqbiz-wq').
        sensitivity_max: Highest sensitivity tier to return. 'public' | 'private' | 'sensitive'.
                         'secret' is NEVER returned here — use recall_sensitive_memory.
        include_pending: If True (default), include pending memories. False = approved only.

    Returns ranked memories with snippets where query matched.
    """
    require_authorized()
    limit = max(1, min(50, int(limit)))

    SENS_ORDER = {"public": 1, "private": 2, "sensitive": 3}
    max_sens = SENS_ORDER.get(sensitivity_max, 2)
    allowed_sens = [s for s, n in SENS_ORDER.items() if n <= max_sens]

    where = ["archived_at IS NULL", "sensitivity = ANY(%(allowed_sens)s)"]
    params = {"allowed_sens": allowed_sens, "limit": limit}

    if not include_pending:
        where.append("status = 'approved'")
    if memory_type:
        where.append("memory_type = %(memory_type)s")
        params["memory_type"] = memory_type
    if scope:
        where.append("scope = %(scope)s")
        params["scope"] = scope
    if author:
        where.append("author = %(author)s")
        params["author"] = author

    has_query = bool(query and query.strip())
    if has_query:
        where.append("content_tsv @@ plainto_tsquery('english', %(q)s)")
        params["q"] = query
        order = "ts_rank(content_tsv, plainto_tsquery('english', %(q)s)) DESC"
        snippet_col = ("ts_headline('english', content, plainto_tsquery('english', %(q)s), "
                       "'MaxFragments=2,MaxWords=40,MinWords=10') AS snippet,")
    else:
        order = "ts DESC"
        snippet_col = ""

    sql = f"""
    SELECT id, content, summary, memory_type, scope, author, ts, status,
           sensitivity, confidence, decision_matrix_anchors, tags,
           {snippet_col}
           verified_by, verified_at
    FROM memories
    WHERE {' AND '.join(where)}
    ORDER BY {order}
    LIMIT %(limit)s
    """
    with get_pg_conn(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                r["id"] = str(r["id"])
            return {
                "query": query or None,
                "count": len(rows),
                "memories": rows,
            }


@mcp.tool()
def list_memories(
    scope: str = None,
    memory_type: str = None,
    author: str = None,
    status: str = None,
    limit: int = 20,
) -> dict:
    """
    Browse the memories table without a search query. Use for "what have I saved
    recently?" or "show me everything tagged X" patterns.

    Args:
        scope: Optional filter on scope.
        memory_type: Optional filter on type.
        author: Optional filter on author.
        status: Optional filter on status ('pending' / 'approved' / 'archived').
        limit: 1..100, default 20.
    """
    require_authorized()
    limit = max(1, min(100, int(limit)))

    where = ["archived_at IS NULL"]
    params = {"limit": limit}
    if scope:
        where.append("scope = %(scope)s"); params["scope"] = scope
    if memory_type:
        where.append("memory_type = %(memory_type)s"); params["memory_type"] = memory_type
    if author:
        where.append("author = %(author)s"); params["author"] = author
    if status:
        where.append("status = %(status)s"); params["status"] = status

    sql = f"""
    SELECT id, summary, content, memory_type, scope, author, ts, status,
           sensitivity, confidence, tags
    FROM memories
    WHERE {' AND '.join(where)}
    ORDER BY ts DESC
    LIMIT %(limit)s
    """
    with get_pg_conn(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                r["id"] = str(r["id"])
            return {"count": len(rows), "memories": rows}


@mcp.tool()
def verify_memory(id: str) -> dict:
    """
    Andre-only: promote a pending memory to status='approved' (permanent).

    Args:
        id: UUID of the memory to verify.
    """
    user = require_andre()
    sql = """
    UPDATE memories
    SET status = 'approved', verified_by = %s, verified_at = NOW()
    WHERE id = %s AND status = 'pending'
    RETURNING id, status, verified_at
    """
    try:
        with get_pg_conn(readonly=False) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user, id))
                row = cur.fetchone()
                conn.commit()
                if not row:
                    return {"error": f"No pending memory found with id={id}"}
                return {
                    "id": str(row[0]),
                    "status": row[1],
                    "verified_by": user,
                    "verified_at": row[2].isoformat() if row[2] else None,
                }
    except Exception as e:
        return {"error": str(e)[:500]}


@mcp.tool()
def archive_memory(id: str, reason: str = None) -> dict:
    """
    Andre-only: soft-delete a memory (sets archived_at + archived_reason).
    Data is preserved; just hidden from default queries.

    Args:
        id: UUID of the memory to archive.
        reason: Optional reason string.
    """
    require_andre()
    sql = """
    UPDATE memories
    SET status = 'archived', archived_at = NOW(), archived_reason = %s
    WHERE id = %s AND archived_at IS NULL
    RETURNING id, archived_at
    """
    try:
        with get_pg_conn(readonly=False) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (reason, id))
                row = cur.fetchone()
                conn.commit()
                if not row:
                    return {"error": f"No active memory found with id={id}"}
                return {
                    "id": str(row[0]),
                    "archived_at": row[1].isoformat() if row[1] else None,
                    "reason": reason,
                }
    except Exception as e:
        return {"error": str(e)[:500]}


@mcp.tool()
def recall_sensitive_memory(query: str, passphrase: str) -> dict:
    """
    Recall 'secret'-tier memories (pgcrypto-encrypted). Requires the passphrase
    Andre holds — never stored on VPS.

    STATUS: stub in v5. The pgcrypto encrypt/decrypt path lands in Phase 4.7.1
    when HIIQ_MEMORY_PASSPHRASE is wired through. For now, this tool reports
    that and returns nothing.
    """
    require_authorized()
    return {
        "status": "not-implemented",
        "phase": "4.7.1",
        "note": (
            "secret-tier memories require pgcrypto + master passphrase wiring. "
            "Coming next iteration. For now, use sensitivity='sensitive' which "
            "stores plaintext but is segregated from default queries."
        ),
    }


# ============================================================================
# v5.3 — Docling document-conversion proxy tools
# ============================================================================
# B3 architecture: docling-serve runs as a sibling container with no public
# ingress; it has zero authentication. This gateway is the public OAuth-gated
# surface that proxies to it over the private docker network.
#
# Sync conversion only for now — small/medium docs (under ~10 pages PDF on CPU).
# Larger docs need /v1/convert/source/async + polling; add when needed.


@mcp.tool()
def docling_health() -> dict:
    """Health probe for the internal docling-serve sibling container.

    Confirms the conversion backend is reachable on the private docker
    network. No PDF round-trip — just a /health ping. Use this first if
    convert_document() is failing to localize the problem.

    Returns:
        dict with `ok`, `status_code`, `docling_service_url`, `body_excerpt`.
        On failure: `ok=False` plus `error` and a `hint` about likely cause.
    """
    require_authorized()
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{DOCLING_SERVICE_URL}/health")
        return {
            "ok": r.status_code == 200,
            "status_code": r.status_code,
            "docling_service_url": DOCLING_SERVICE_URL,
            "body_excerpt": r.text[:200],
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e)[:300],
            "docling_service_url": DOCLING_SERVICE_URL,
            "hint": (
                "docling-serve container may not be running, or DOCLING_SERVICE_URL "
                "env var on the gateway points at the wrong hostname/port. Expected "
                "docker DNS hostname is 'docling-serve' inside the coolify network."
            ),
        }


@mcp.tool()
def convert_document(
    source_url: str,
    output_formats: list = None,
    timeout_seconds: float = None,
) -> dict:
    """Convert a public-URL document (PDF/DOCX/PPTX/XLSX/HTML/image) to Markdown
    via the internal docling-serve container. Synchronous.

    Suitable for small/medium docs (under ~10 pages PDF on CPU). First call
    after container restart takes 30-60s for model load. Larger docs may
    exceed the MCP transport timeout — for those, use the async endpoint
    (not yet exposed; add when needed).

    Args:
        source_url: Public URL of the document. docling-serve fetches it
                    directly, so it must be internet-reachable from the VPS.
                    Local files are not supported via this endpoint — base64
                    upload is possible but not exposed here yet.
        output_formats: List of output formats to populate. Default ['md', 'json'].
                        Options: 'md' (markdown), 'json' (structured
                        DoclingDocument), 'html', 'text', 'doctags'. More
                        formats = larger response payload.
        timeout_seconds: Per-call HTTP timeout override. Default 180s from env.

    Returns:
        On success: dict with `status` ('success'|'partial_success'|'failure'),
        `processing_time`, `source_url`, `md_content`, and the other format
        fields you requested. Unrequested format fields are None.
        On failure: dict with `error` and optionally `hint` / `body_excerpt`.
    """
    require_authorized()
    if not source_url or not source_url.strip():
        return {"error": "source_url cannot be empty"}
    output_formats = list(output_formats or ["md", "json"])
    timeout = float(timeout_seconds) if timeout_seconds else DOCLING_HTTP_TIMEOUT
    payload = {
        "sources": [{"kind": "http", "url": source_url.strip()}],
        "options": {
            "output_formats": output_formats,
        },
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(
                f"{DOCLING_SERVICE_URL}/v1/convert/source",
                json=payload,
            )
        if r.status_code != 200:
            return {
                "error": f"docling-serve returned HTTP {r.status_code}",
                "body_excerpt": r.text[:500],
                "hint": (
                    "422 usually means the request payload shape is wrong; "
                    "5xx usually means docling-serve is unhealthy — try docling_health."
                ),
            }
        data = r.json()
        doc = (data.get("document") or {}) if isinstance(data, dict) else {}
        return {
            "status": data.get("status"),
            "processing_time": data.get("processing_time"),
            "source_url": source_url,
            "output_formats": output_formats,
            "md_content": doc.get("md_content"),
            "json_content": doc.get("json_content") if "json" in output_formats else None,
            "html_content": doc.get("html_content") if "html" in output_formats else None,
            "text_content": doc.get("text_content") if "text" in output_formats else None,
            "doctags_content": doc.get("doctags_content") if "doctags" in output_formats else None,
            "errors": data.get("errors") or [],
        }
    except httpx.TimeoutException:
        return {
            "error": "conversion timed out",
            "timeout_seconds": timeout,
            "hint": (
                "first call after container restart needs ~30-60s for model load; "
                "large PDFs may need longer. Retry, or pass a larger timeout_seconds."
            ),
        }
    except Exception as e:
        return {"error": str(e)[:500]}


# ============================================================================
# v5.4 — Chatterbox TTS proxy tools
# ============================================================================
# Text-to-speech via hiiq-andre's RTX 5060. The TTS service is bearer-auth'd
# (HIIQ MCP discipline — single master CB_MCP_AUTH_TOKEN) and reached over
# Tailscale at TTS_SERVICE_URL. This gateway is the public OAuth-gated surface;
# it forwards the bearer token automatically so callers don't need it.
#
# Audio is base64-encoded WAV (PCM_16 24kHz mono) inline in the MCP response.
# Suitable for clips up to ~30 seconds — longer scripts bloat the response.
# For audiobook-length content, a URL-pointer return path would be needed
# (deferred until first real use surfaces the constraint).


def _tts_auth_headers() -> dict:
    """Build the bearer header for hiiq-andre's TTS service."""
    if not TTS_SERVICE_TOKEN:
        raise RuntimeError(
            "TTS_SERVICE_TOKEN / CB_MCP_AUTH_TOKEN not set on gateway. "
            "The TTS service rejects unauthenticated requests."
        )
    return {"Authorization": f"Bearer {TTS_SERVICE_TOKEN}"}


def _validate_tts_voice_label(label: str) -> Optional[str]:
    if not label:
        return "label cannot be empty"
    if len(label) > 31:
        return "label must be 31 chars or fewer (a-z 0-9 _ -)"
    if not TTS_VOICE_LABEL_REGEX.fullmatch(label):
        return "label must match ^[a-z0-9][a-z0-9_-]{0,30}$ (lowercase a-z 0-9 _ -)"
    return None


@mcp.tool()
def tts_health() -> dict:
    """Health probe for the hiiq-andre Chatterbox TTS service.

    Confirms the TTS backend is reachable over Tailscale + CUDA is alive.
    No generation round-trip — just a /health ping. Use this first if
    tts_generate() is failing to localize the problem (gateway-side env
    misconfig vs Tailscale routing vs TTS service down vs CUDA issue).

    Returns:
        dict with `ok`, `status_code`, `tts_service_url`, and the upstream
        health body (active_engine, cuda_device, vram_total_gb, voices_count).
        On failure: `ok=False` plus `error` and a `hint`.
    """
    require_authorized()
    try:
        # /health is no-auth on the TTS service, but send the header anyway
        # so the gateway can fail-fast if the token isn't configured.
        headers = _tts_auth_headers()
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{TTS_SERVICE_URL}/health", headers=headers)
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
        return {
            "ok": r.status_code == 200,
            "status_code": r.status_code,
            "tts_service_url": TTS_SERVICE_URL,
            "body": body,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e)[:300],
            "tts_service_url": TTS_SERVICE_URL,
            "hint": (
                "TTS service may not be running on hiiq-andre (start with "
                "'execution/Running Scripts/_start_tts_service.ps1'), or "
                "Tailscale on hiiqbiz-vps cannot reach 100.90.91.72:8019, "
                "or TTS_SERVICE_URL env on the gateway is wrong."
            ),
        }


@mcp.tool()
def tts_list_voices() -> dict:
    """List the voice reference clips registered on hiiq-andre.

    Each voice is a 10-second WAV at C:/HIIQ-Models/chatterbox/voices/<label>.wav.
    Pass the `label` (filename minus .wav) as `voice=` to tts_generate() to
    clone that speaker. Omitting `voice=` uses Chatterbox's baked-in default.

    Returns:
        dict with `voices`: list of {label, file, size_bytes, added_at}.
        On failure: dict with `error` (e.g. service unreachable, auth failure).
    """
    require_authorized()
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{TTS_SERVICE_URL}/voices", headers=_tts_auth_headers())
        if r.status_code != 200:
            return {"error": f"TTS service returned HTTP {r.status_code}", "body_excerpt": r.text[:300]}
        return {"voices": r.json()}
    except Exception as e:
        return {"error": str(e)[:300]}


@mcp.tool()
def tts_save_voice(label: str, audio_b64: str, overwrite: bool = False) -> dict:
    """Register a new voice reference clip on hiiq-andre.

    Args:
        label: Lowercase identifier, a-z 0-9 _ -, max 31 chars (e.g. "andre",
               "narrator-warm"). This becomes what `voice=` accepts.
        audio_b64: Base64-encoded WAV. Should be ~10 seconds of clean continuous
                   speech, single speaker, normal pace. Stereo → auto-mixed.
        overwrite: Pass True to replace an existing label. Default False (409 on
                   conflict).

    Returns:
        dict with `label`, `file`, `sample_rate`, `duration_s` on success.
        On failure: dict with `error`.
    """
    require_authorized()
    label = (label or "").strip()
    label_error = _validate_tts_voice_label(label)
    if label_error:
        return {"error": label_error}
    payload = {"label": label, "audio_b64": audio_b64, "overwrite": overwrite}
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(
                f"{TTS_SERVICE_URL}/voices",
                headers=_tts_auth_headers(),
                json=payload,
            )
        if r.status_code != 200:
            return {"error": f"TTS service returned HTTP {r.status_code}", "body_excerpt": r.text[:500]}
        return r.json()
    except Exception as e:
        return {"error": str(e)[:300]}


@mcp.tool()
def tts_delete_voice(label: str) -> dict:
    """Remove a voice reference clip from hiiq-andre.

    Args:
        label: The label whose .wav should be deleted.

    Returns:
        dict with `label`, `deleted` on success.
        On failure: dict with `error` (e.g. label not found).
    """
    require_authorized()
    label = (label or "").strip()
    label_error = _validate_tts_voice_label(label)
    if label_error:
        return {"error": label_error}
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.delete(
                f"{TTS_SERVICE_URL}/voices/{label}",
                headers=_tts_auth_headers(),
            )
        if r.status_code != 200:
            return {"error": f"TTS service returned HTTP {r.status_code}", "body_excerpt": r.text[:300]}
        return r.json()
    except Exception as e:
        return {"error": str(e)[:300]}


@mcp.tool()
def tts_generate(
    text: str,
    voice: str = None,
    engine: str = "turbo",
    exaggeration: float = 0.0,
    cfg_weight: float = 0.0,
    temperature: float = 0.8,
    language_id: str = None,
) -> dict:
    """Generate speech audio from text via hiiq-andre's Chatterbox TTS.

    The audio comes back inline as base64-encoded WAV (PCM_16 24kHz mono).
    Suitable for clips up to ~30 seconds — longer scripts bloat the MCP
    response and may hit transport limits. For audiobook-length content,
    chunk by paragraph and concatenate client-side.

    Args:
        text: What to read aloud. Max 8000 chars. Inline paralinguistic tags
              supported: `[laugh]`, `[chuckle]`, `[cough]`.
        voice: Optional label from tts_list_voices(). If omitted, the model's
               baked-in default voice is used. Pass any saved label (e.g.
               "andre", "narrator-warm") to clone that speaker.
        engine: "turbo" (default, fastest, 1-step gen, paralinguistic tags),
                "regular" (emotion exaggeration with stronger defaults), or
                "multilingual" (23 languages, requires language_id).
        exaggeration: 0.0–1.0. Emotion intensity. Turbo defaults 0.0 (flat),
                      Regular defaults 0.5. Higher = more expressive.
        cfg_weight: 0.0–1.0. Classifier-free guidance strength.
        temperature: 0.1–1.5. Sampling randomness. Default 0.8.
        language_id: Required for engine="multilingual". One of: en, es, fr,
                     de, it, pt, ru, ja, ko, zh, ar, hi, tr, pl, nl, sv, id,
                     vi, th, he, fa, uk, el.

    Returns:
        On success: dict with `audio_b64`, `sample_rate` (24000), `duration_s`,
        `generation_time_s`, `engine`, `voice`, `text_len`.
        On failure: dict with `error` and optionally `hint`.

    Note: First call after the TTS service restarts pays a ~50s cold-start
    cost (model load + CUDA kernel JIT). Subsequent calls are ~1.5–2× realtime
    on RTX 5060. Switching engines (turbo → multilingual) evicts the prior
    engine to free VRAM, so back-to-back engine switches are slow.
    """
    require_authorized()
    if not text or not text.strip():
        return {"error": "text cannot be empty"}
    if len(text) > 8000:
        return {"error": "text must be 8000 chars or fewer", "text_len": len(text)}
    if engine == "multilingual" and not language_id:
        return {"error": "language_id is required when engine='multilingual'"}
    if engine not in ("turbo", "regular", "multilingual"):
        return {"error": f"engine must be one of turbo / regular / multilingual (got {engine!r})"}
    payload = {
        "text": text,
        "engine": engine,
        "exaggeration": float(exaggeration),
        "cfg_weight": float(cfg_weight),
        "temperature": float(temperature),
    }
    if voice:
        payload["voice"] = voice
    if language_id:
        payload["language_id"] = language_id
    try:
        with httpx.Client(timeout=TTS_HTTP_TIMEOUT) as client:
            r = client.post(
                f"{TTS_SERVICE_URL}/generate",
                headers=_tts_auth_headers(),
                json=payload,
            )
        if r.status_code != 200:
            return {
                "error": f"TTS service returned HTTP {r.status_code}",
                "body_excerpt": r.text[:500],
                "hint": (
                    "404 usually means an unknown voice label; "
                    "400 means a parameter problem (e.g. multilingual without language_id); "
                    "401 means TTS_SERVICE_TOKEN on the gateway doesn't match the TTS service's CB_MCP_AUTH_TOKEN; "
                    "5xx means the TTS service is unhealthy — try tts_health()."
                ),
            }
        return r.json()
    except httpx.TimeoutException:
        return {
            "error": "TTS generation timed out",
            "timeout_seconds": TTS_HTTP_TIMEOUT,
            "hint": (
                "First call after TTS service restart needs ~50s for model load + CUDA JIT. "
                "Long texts (>30s of audio) may exceed the timeout — chunk shorter or raise "
                "TTS_HTTP_TIMEOUT env on the gateway."
            ),
        }
    except Exception as e:
        return {"error": str(e)[:500]}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port, path="/mcp")
