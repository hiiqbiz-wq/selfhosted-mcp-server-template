"""
In-image capability-regression gate for the HIIQ Edge MCP gateway.

Runs INSIDE the built Docker image, after gate_import_check.py. It imports the
actual remote gateway entrypoint from /app, enumerates the FastMCP registrations,
and fails if the shipped image no longer exposes the control-plane tools/resources
that define the intended deploy surface.

This catches the bad-merge class where git ancestry says the work is merged, but
the tree content silently reverted to an older valid server with fewer tools or
zero resources.
"""
import asyncio
import contextlib
import importlib
import importlib.util
import inspect
import io
import json
import os
import pathlib
import sys

APP = pathlib.Path(__file__).resolve().parent
ENTRY = APP / "remote-seo-checker.py"

MIN_TOOL_COUNT = 25
MIN_RESOURCE_COUNT = 5

REQUIRED_MODULES = {
    "key_value": "py-key-value-aio[postgresql]",
}

REQUIRED_TOOLS = {
    "session_boot",
    "session_end",
    "dangerous_action_request",
    "approval_resolve",
    "mcp_tool_audit",
    "memory_eval_run",
}

REQUIRED_RESOURCES = {
    "hiiq://stack/health",
    "hiiq://memory/status",
    "hiiq://schemas/pacom",
    "hiiq://tasks/open",
    "hiiq://governance/approvals",
}


def _fail(failures, stage, target, error):
    failures.append({"stage": stage, "target": target, "error": str(error)})


def _prime_fast_fail_env():
    """Avoid slow external boot probes while preserving registration behavior."""
    os.environ["PACOM_PG_HOST"] = "127.0.0.1"
    os.environ["PACOM_PG_PORT"] = "1"
    os.environ["PACOM_PG_PASSWORD"] = ""
    os.environ["PUBLIC_BASE_URL"] = "http://localhost"
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ.pop("GITHUB_OAUTH_CLIENT_ID", None)
    os.environ.pop("GITHUB_OAUTH_CLIENT_SECRET", None)


def _load_gateway(failures):
    if not ENTRY.exists():
        _fail(failures, "entry", ENTRY.name, "entrypoint not present in image")
        return None, "", ""

    spec = importlib.util.spec_from_file_location("hiiq_gateway_entry", ENTRY)
    if not spec or not spec.loader:
        _fail(failures, "entry", ENTRY.name, "could not create import spec")
        return None, "", ""

    module = importlib.util.module_from_spec(spec)
    boot_stdout = io.StringIO()
    boot_stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(boot_stdout), contextlib.redirect_stderr(boot_stderr):
            spec.loader.exec_module(module)
    except Exception as e:
        _fail(failures, "entry", ENTRY.name, repr(e))
        return None, boot_stdout.getvalue(), boot_stderr.getvalue()

    mcp = getattr(module, "mcp", None)
    if mcp is None:
        _fail(failures, "entry", ENTRY.name, "module did not expose global mcp")
    return mcp, boot_stdout.getvalue(), boot_stderr.getvalue()


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _names_from_collection(value, attr):
    if isinstance(value, dict):
        return {str(k) for k in value.keys()}
    return {str(getattr(item, attr, item)) for item in value}


async def _collect_capabilities(mcp, failures):
    tools, resources = set(), set()

    get_tools = getattr(mcp, "get_tools", None) or getattr(mcp, "list_tools", None)
    if not get_tools:
        _fail(failures, "capability", "tools", "FastMCP exposes no get/list tools API")
    else:
        try:
            tools = _names_from_collection(await _maybe_await(get_tools()), "name")
        except Exception as e:
            _fail(failures, "capability", "tools", repr(e))

    get_resources = getattr(mcp, "get_resources", None) or getattr(mcp, "list_resources", None)
    if not get_resources:
        _fail(failures, "capability", "resources", "FastMCP exposes no get/list resources API")
    else:
        try:
            resources = _names_from_collection(await _maybe_await(get_resources()), "uri")
        except Exception as e:
            _fail(failures, "capability", "resources", repr(e))

    return tools, resources


def main():
    failures = []
    checked = {"modules": [], "tools": [], "resources": []}

    for module, package in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module)
            checked["modules"].append(module)
        except Exception as e:
            _fail(failures, "dependency", package, repr(e))

    _prime_fast_fail_env()
    mcp, boot_stdout, boot_stderr = _load_gateway(failures)
    tools, resources = set(), set()
    if mcp is not None:
        try:
            tools, resources = asyncio.run(_collect_capabilities(mcp, failures))
        except Exception as e:
            _fail(failures, "capability", "collect", repr(e))

    missing_tools = sorted(REQUIRED_TOOLS - tools)
    missing_resources = sorted(REQUIRED_RESOURCES - resources)
    if len(tools) < MIN_TOOL_COUNT:
        _fail(failures, "capability", "tool_count", f"{len(tools)} < {MIN_TOOL_COUNT}")
    if len(resources) < MIN_RESOURCE_COUNT:
        _fail(failures, "capability", "resource_count", f"{len(resources)} < {MIN_RESOURCE_COUNT}")
    for name in missing_tools:
        _fail(failures, "capability", f"tool:{name}", "required tool missing")
    for uri in missing_resources:
        _fail(failures, "capability", f"resource:{uri}", "required resource missing")

    checked["tools"] = sorted(tools)
    checked["resources"] = sorted(resources)
    report = {
        "ok": not failures,
        "entry": ENTRY.name,
        "minimums": {"tools": MIN_TOOL_COUNT, "resources": MIN_RESOURCE_COUNT},
        "required_tools": sorted(REQUIRED_TOOLS),
        "required_resources": sorted(REQUIRED_RESOURCES),
        "checked": checked,
        "counts": {"tools": len(tools), "resources": len(resources)},
        "boot_stdout": boot_stdout.splitlines()[-5:],
        "boot_stderr": boot_stderr.splitlines()[-5:],
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
