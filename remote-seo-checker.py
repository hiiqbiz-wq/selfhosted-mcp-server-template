"""
HIIQ Edge MCP gateway — remote MCP server exposing HIIQ tools to any Claude surface.

Runs on the Hostinger VPS (mcp.hiiqbiz.com). Tools that need data reach back to
Pacific (PACOM Postgres at 100.87.218.106:5430) via the host's Tailscale interface.

Auth: GitHub OAuth via FastMCP's GitHubProvider. DCR-compliant — Claude.ai's
custom-connector flow discovers, registers, and walks the user through GitHub
login. After login, the GitHub username is checked against the env-configured
allowlist (`ALLOWED_GH_USERS`, comma-separated).

This file replaces the upstream `remote-seo-checker.py` (filename kept so the
upstream Dockerfile CMD doesn't need to change).

Tools (8):
  ping_hiiq           — health + PACOM reachability + authenticated user
  pacom_tables        — list user tables with row counts
  pacom_recent_cli    — last N cli_audit entries
  query_pacom         — arbitrary read-only SELECT
  search_vault        — full-text search the indexed Obsidian vault (2780+ docs)
  pacom_skills        — list registered skills, optional substring filter
  pacom_plugins       — list installed plugins with capabilities
  session_resume      — most recent Claude session handoffs (unconsumed by default)
"""

import os
import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from fastmcp import FastMCP
from fastmcp.server.auth.providers.github import GitHubProvider
from fastmcp.server.dependencies import get_access_token

PACOM_PG_HOST = os.environ.get("PACOM_PG_HOST", "100.87.218.106")
PACOM_PG_PORT = int(os.environ.get("PACOM_PG_PORT", "5430"))
PACOM_PG_DBNAME = os.environ.get("PACOM_PG_DBNAME", "pacom")
PACOM_PG_USER = os.environ.get("PACOM_PG_USER", "postgres")
PACOM_PG_PASSWORD = os.environ.get("PACOM_PG_PASSWORD", "")

GITHUB_CLIENT_ID = os.environ.get("GITHUB_OAUTH_CLIENT_ID", "").strip()
GITHUB_CLIENT_SECRET = os.environ.get("GITHUB_OAUTH_CLIENT_SECRET", "").strip()
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "https://mcp.hiiqbiz.com").rstrip("/")
ALLOWED_GH_USERS = {
    u.strip()
    for u in os.environ.get("ALLOWED_GH_USERS", "hiiqbiz-wq").split(",")
    if u.strip()
}

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
        f"[boot] GitHub OAuth enabled. Allowlist: {sorted(ALLOWED_GH_USERS) or '(empty — denies everyone)'}",
        flush=True,
    )
else:
    mcp = FastMCP(name="HIIQ Edge")
    print(
        "[boot] WARNING: GITHUB_OAUTH_CLIENT_ID/SECRET not set — gateway is OPEN. "
        "Set both env vars to enable OAuth.",
        flush=True,
    )


def require_authorized() -> str:
    """Raise if the authenticated GitHub user is not in the allowlist. Returns the login on success."""
    if not (GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET):
        return "anonymous"
    token = get_access_token()
    if not token:
        raise PermissionError("No authenticated token")
    login = (token.claims.get("login") or "").strip()
    if login not in ALLOWED_GH_USERS:
        raise PermissionError(
            f"GitHub user '{login}' is not in the HIIQ Edge allowlist"
        )
    return login


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


# ---------------------------------------------------------------------------
# Health + meta
# ---------------------------------------------------------------------------

@mcp.tool()
def ping_hiiq() -> dict:
    """
    Health check for the HIIQ Edge MCP gateway. Returns server identity,
    UTC timestamp, the authenticated GitHub user, and whether PACOM
    Postgres is reachable from this VPS via Tailscale.
    """
    user = require_authorized()
    out = {
        "status": "ok",
        "server": "HIIQ Edge MCP gateway",
        "node": "hiiqbiz-vps (Hostinger KVM 4, US-Boston)",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "authenticated_as": user,
        "pacom_reachable": False,
        "auth_enabled": bool(GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET),
        "tools": [
            "ping_hiiq", "pacom_tables", "pacom_recent_cli", "query_pacom",
            "search_vault", "pacom_skills", "pacom_plugins", "session_resume",
        ],
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


# ---------------------------------------------------------------------------
# PACOM raw SQL access (read-only)
# ---------------------------------------------------------------------------

@mcp.tool()
def pacom_tables() -> dict:
    """
    List all user tables in PACOM with their live row counts. Note: counts come
    from pg_stat_user_tables and may lag real counts until autovacuum runs;
    for an exact count, use `query_pacom` with COUNT(*).
    """
    require_authorized()
    sql = """
    SELECT
      schemaname || '.' || relname AS table_name,
      n_live_tup AS row_count
    FROM pg_stat_user_tables
    ORDER BY schemaname, relname
    """
    with get_pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            return {"tables": [dict(r) for r in cur.fetchall()]}


@mcp.tool()
def pacom_recent_cli(n: int = 20) -> dict:
    """
    Return the N most recent entries from PACOM's cli_audit table. Shows
    what tools ran on Pacific recently, with args, status, and timing.

    Args:
        n: Number of entries to return. Clamped to 1..100. Default 20.
    """
    require_authorized()
    n = max(1, min(100, int(n)))
    sql = """
    SELECT ts, rig_hostname, tool_name, args, backend, endpoint,
           status, error_message, duration_ms
    FROM cli_audit
    ORDER BY ts DESC
    LIMIT %s
    """
    with get_pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (n,))
            return {
                "entries": [dict(r) for r in cur.fetchall()],
                "source": "PACOM cli_audit table (Pacific PG18 :5430)",
            }


@mcp.tool()
def query_pacom(sql: str, max_rows: int = 50) -> dict:
    """
    Execute a read-only SQL SELECT against PACOM. The session is set to
    READ ONLY at the Postgres level, so any INSERT/UPDATE/DELETE/DDL is
    rejected by the database itself.

    Args:
        sql: SELECT statement.
        max_rows: Cap on returned rows (1..500, default 50).

    Returns:
        dict with `columns`, `rows`, and `row_count`. On error, returns
        `{"error": "<message>"}`.
    """
    require_authorized()
    max_rows = max(1, min(500, int(max_rows)))
    try:
        with get_pg_conn(readonly=True) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql)
                rows = cur.fetchmany(max_rows)
                cols = [d[0] for d in cur.description] if cur.description else []
                return {
                    "columns": cols,
                    "rows": [dict(r) for r in rows],
                    "row_count": len(rows),
                }
    except Exception as e:
        return {"error": str(e)[:500]}


# ---------------------------------------------------------------------------
# Vault search (PACOM vault_index, ~2780 docs, ts_rank full-text)
# ---------------------------------------------------------------------------

@mcp.tool()
def search_vault(query: str, limit: int = 10, scope: str = None) -> dict:
    """
    Full-text search the indexed HIIQ vault (~2780 files across `.claude/memory/`,
    `notes/`, `playbooks/`, `knowledge/`, etc.). Uses PostgreSQL ts_rank for
    relevance ordering and ts_headline for snippet generation.

    Args:
        query: Search terms (English). Can be multi-word.
        limit: Max results (1..50, default 10).
        scope: Optional scope filter. Valid scopes include 'notes', 'memory',
               'knowledge', 'claude', 'plans', 'reports', 'playbooks',
               'rules', 'projects'. None = search all scopes.

    Returns:
        dict with `hits` (each: path, scope, mtime, rank, snippet) and the
        echoed query/scope.
    """
    require_authorized()
    limit = max(1, min(50, int(limit)))
    if scope:
        sql = """
        SELECT path, scope, mtime,
               ts_rank(content_tsv, plainto_tsquery('english', %(q)s)) AS rank,
               ts_headline('english', content_text,
                           plainto_tsquery('english', %(q)s),
                           'MaxFragments=2,MaxWords=40,MinWords=10') AS snippet
        FROM vault_index
        WHERE content_tsv @@ plainto_tsquery('english', %(q)s)
          AND scope = %(scope)s
        ORDER BY rank DESC
        LIMIT %(limit)s
        """
        params = {"q": query, "scope": scope, "limit": limit}
    else:
        sql = """
        SELECT path, scope, mtime,
               ts_rank(content_tsv, plainto_tsquery('english', %(q)s)) AS rank,
               ts_headline('english', content_text,
                           plainto_tsquery('english', %(q)s),
                           'MaxFragments=2,MaxWords=40,MinWords=10') AS snippet
        FROM vault_index
        WHERE content_tsv @@ plainto_tsquery('english', %(q)s)
        ORDER BY rank DESC
        LIMIT %(limit)s
        """
        params = {"q": query, "limit": limit}
    with get_pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return {
                "hits": [dict(r) for r in cur.fetchall()],
                "query": query,
                "scope_filter": scope,
            }


# ---------------------------------------------------------------------------
# Skills + plugins registries (snapshot of Pacific's Claude config layer)
# ---------------------------------------------------------------------------

@mcp.tool()
def pacom_skills(query: str = None) -> dict:
    """
    List Claude skills registered in PACOM (~27 skills as of 2026-05-21).
    Each skill includes path, scope, plugin association, description, and
    when-to-use hint.

    Args:
        query: Optional case-insensitive substring filter on name/description/
               when-to-use text. None = return all skills.
    """
    require_authorized()
    if query:
        sql = """
        SELECT name, path, scope, plugin_name, description, when_to_use, model
        FROM skills_registry
        WHERE search_text ILIKE %s
        ORDER BY name
        """
        params = (f"%{query}%",)
    else:
        sql = """
        SELECT name, path, scope, plugin_name, description, when_to_use, model
        FROM skills_registry
        ORDER BY name
        """
        params = ()
    with get_pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return {"skills": [dict(r) for r in cur.fetchall()]}


@mcp.tool()
def pacom_plugins() -> dict:
    """
    List Claude plugins registered in PACOM with their declared capabilities
    and per-component counts (skills/agents/commands/hooks/mcp servers).
    """
    require_authorized()
    sql = """
    SELECT name, version, description, author_name, install_path,
           capabilities, skill_count, agent_count, command_count,
           hook_count, mcp_count
    FROM plugins_registry
    ORDER BY name
    """
    with get_pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            return {"plugins": [dict(r) for r in cur.fetchall()]}


# ---------------------------------------------------------------------------
# Session handoff (resume a previous Claude session)
# ---------------------------------------------------------------------------

@mcp.tool()
def session_resume(only_unconsumed: bool = True, limit: int = 5) -> dict:
    """
    Get recent Claude session handoffs saved to PACOM via `save-pacom`.
    Use this to pick up where a previous session (on any rig) left off —
    last command, current chapter, next planned action, plus the raw
    handoff narrative.

    Args:
        only_unconsumed: If True (default), only show handoffs not yet
                         marked as consumed.
        limit: Max handoffs to return (1..20, default 5).
    """
    require_authorized()
    limit = max(1, min(20, int(limit)))
    where = "WHERE consumed_at IS NULL" if only_unconsumed else ""
    sql = f"""
    SELECT session_id, rig_hostname, ended_at, last_command,
           current_chapter, next_action, raw_handoff, consumed_at
    FROM session_handoff
    {where}
    ORDER BY ended_at DESC NULLS LAST
    LIMIT %s
    """
    with get_pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (limit,))
            return {"handoffs": [dict(r) for r in cur.fetchall()]}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port, path="/mcp")
