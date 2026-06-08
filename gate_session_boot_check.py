"""
In-image behavior check for the session lifecycle grounding contract.

This runs after the capability check. It imports the real gateway entrypoint,
patches the PACOM connection with a small fake cursor, calls session_end(), then
calls session_boot() and asserts that the returned boot block satisfies the
durable grounding slice: closeout write-through lands in session_handoff, the
next boot reads that handoff, approved memories carry citation-ready source_id
+quote pairs, and the response schema exposes grounded_in + speculation.
"""
import contextlib
import datetime as _dt
import importlib.util
import io
import json
import os
import pathlib
import sys
import uuid

APP = pathlib.Path(__file__).resolve().parent
ENTRY = APP / "remote-seo-checker.py"
BASE_NOW = _dt.datetime(2026, 6, 7, 12, 0, 0)
HANDOFF_ROWS = [{
    "session_id": "gate-handoff-1",
    "rig_hostname": "HIIQ-RTX-5070",
    "ended_at": BASE_NOW,
    "last_command": "deploy gate",
    "current_chapter": "session_boot grounding slice",
    "next_action": "Verify session_boot returns citation-ready approved memories.",
    "raw_handoff": "Continue with durable grounding.",
    "consumed_at": None,
}]


def _prime_env():
    os.environ["PACOM_PG_HOST"] = "127.0.0.1"
    os.environ["PACOM_PG_PORT"] = "1"
    os.environ["PACOM_PG_PASSWORD"] = ""
    os.environ["PUBLIC_BASE_URL"] = "http://localhost"
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ.pop("GITHUB_OAUTH_CLIENT_ID", None)
    os.environ.pop("GITHUB_OAUTH_CLIENT_SECRET", None)


class FakeCursor:
    def __init__(self, *_args, **_kwargs):
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, sql, params=None):
        params = params or {}
        if "FROM public.memories" in sql and "count(*) AS total" in sql:
            self._rows = [{
                "total": 3,
                "approved": 2,
                "pending": 1,
                "archived": 0,
                "embedded": 2,
            }]
            return
        if "INSERT INTO public.session_handoff" in sql or "INSERT INTO session_handoff" in sql:
            handoff_id = uuid.UUID("22222222-2222-2222-2222-222222222222")
            ended_at = BASE_NOW + _dt.timedelta(minutes=1)
            HANDOFF_ROWS.append({
                "session_id": params[0],
                "rig_hostname": params[1],
                "ended_at": ended_at,
                "last_command": params[3],
                "current_chapter": params[4],
                "next_action": params[5],
                "raw_handoff": params[6],
                "consumed_at": None,
            })
            self._rows = [(handoff_id, ended_at)]
            return
        if "FROM session_handoff" in sql:
            limit = params[0] if isinstance(params, (list, tuple)) and params else 1
            self._rows = sorted(HANDOFF_ROWS, key=lambda r: r["ended_at"], reverse=True)[:limit]
            return
        if "FROM memories" in sql:
            self._rows = [{
                "id": uuid.UUID("11111111-1111-1111-1111-111111111111"),
                "content": "session_boot must be the durable retrieve-first path for HIIQ project answers.",
                "summary": "session_boot enforces retrieve-first with citation-ready memory evidence.",
                "memory_type": "decision",
                "scope": "project:mcp-gateway",
                "author": "gate-check",
                "ts": BASE_NOW,
                "status": "approved",
                "sensitivity": "private",
                "confidence": 8,
                "decision_matrix_anchors": ["scope of a fix or refactor"],
                "tags": ["grounding", "session_boot"],
                "snippet": "session_boot durable retrieve-first path",
                "verified_by": "hiiqbiz-wq",
                "verified_at": BASE_NOW,
            }]
            return
        raise AssertionError(f"unexpected SQL in fake cursor: {sql[:120]}")

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def cursor(self, *args, **kwargs):
        return FakeCursor(*args, **kwargs)

    def commit(self):
        return None


def _load_gateway():
    spec = importlib.util.spec_from_file_location("hiiq_gateway_entry", ENTRY)
    if not spec or not spec.loader:
        raise RuntimeError("could not create import spec")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    _prime_env()
    module = _load_gateway()
    module.get_pg_conn = lambda readonly=True: FakeConn()
    module.require_authorized = lambda: "gate-check"
    module.detect_caller_surface = lambda: ("gate-check", "", None)

    closeout = module.session_end(
        next_action="Continue the verified session_end to session_boot handoff.",
        chapter="session_end grounding slice",
        last_command="gate_session_boot_check.py",
        last_files=["remote-seo-checker.py", "gate_session_boot_check.py"],
        raw_handoff="session_end wrote this closeout; session_boot must return it next.",
        session_id="gate-closeout-session",
    )

    _assert(closeout["ok"] is True, "session_end should write a handoff with fake PACOM")
    _assert(closeout["handoff"]["session_id"] == "gate-closeout-session", "session_end session_id mismatch")
    _assert(closeout["handoff"]["last_files_count"] == 2, "session_end should count last_files")

    boot = module.session_boot(
        project="MCP Gateway",
        query="grounding compliance",
        memory_limit=3,
        handoff_limit=1,
    )

    _assert(boot["ok"] is True, "session_boot should be ok with fake PACOM")
    grounding = boot["grounding"]
    _assert(grounding["retrieve_first_satisfied_by"] == "session_boot", "retrieve-first marker missing")
    _assert(grounding["citation_required"] is True, "citation_required must be true")
    _assert("grounded_in" in grounding["response_schema"], "response schema must include grounded_in")
    _assert("speculation" in grounding["response_schema"], "response schema must include speculation")

    primary = boot["retrieved"]["approved_memories"]["primary"]
    _assert(primary, "primary approved memory missing")
    citation = primary["citation"]
    _assert(citation["source_type"] == "memory", "memory citation source_type wrong")
    _assert(citation["source_id"] == "11111111-1111-1111-1111-111111111111", "memory citation source_id wrong")
    _assert("session_boot" in citation["quote"], "memory citation quote should support the boot claim")

    handoff = boot["retrieved"]["handoffs"][0]
    _assert(handoff["session_id"] == "gate-closeout-session", "session_boot should read the session_end handoff first")
    _assert(handoff["citation"]["source_type"] == "session_handoff", "handoff citation missing")
    _assert("Continue the verified session_end" in handoff["citation"]["quote"], "handoff citation should cite session_end next_action")
    _assert("pending memories are excluded" in boot["retrieved"]["pending_policy"], "pending policy missing")

    print(json.dumps({
        "ok": True,
        "tools": ["session_end", "session_boot"],
        "session_end_handoff": closeout["handoff"],
        "memory_citation": citation,
        "handoff_citation": handoff["citation"],
        "rules": grounding["rules"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
