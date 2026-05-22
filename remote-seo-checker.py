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
    """
    List all user tables in PACOM with their live row counts. Note that
    row counts come from pg_stat_user_tables and may lag real counts until
    autovacuum runs; for an exact count use query_pacom with COUNT(*).
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
    what tools ran on Pacific recently, with their args, status, and timing.

    Args:
        n: Number of entries to return. Clamped to 1..100. Default 20.
    """
    require_authorized()
    n = max(1, min(100, int(n)))
    sql = """
    SELECT
      ts,
      rig_hostname,
      tool_name,
      args,
      backend,
      endpoint,
      status,
      error_message,
      duration_ms
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
        dict with `columns` (list of column names), `rows` (list of dicts),
        and `row_count`. On error, returns `{"error": "<message>"}`.
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port, path="/mcp")
