"""
In-image import-resolution gate for the HIIQ Edge MCP gateway.

Runs INSIDE the built Docker image (pure stdlib — python:3.11-slim has no uv).
Invoked by deploy_gate.py as:  docker run --entrypoint python <image> /app/gate_import_check.py

What it catches (the v5.5 crash-loop class, board task O):
  - a module the entrypoint imports is MISSING from the image
    (e.g. a `COPY <file>.py` Dockerfile that omits a new sibling) -> ImportError
  - a third-party dep the entrypoint imports is absent from requirements.txt -> ImportError
  - a syntax error in any *.py copied into the image -> compile error

What it deliberately does NOT do (would be a false positive locally):
  - construct the FastMCP server / connect to PACOM / require GitHub OAuth env.
    We only resolve the entrypoint's module-level imports, not run main().

Exit 0 = gate pass; exit 1 = gate fail (JSON report on stdout either way).
"""
import ast
import importlib
import json
import pathlib
import py_compile
import sys

APP = pathlib.Path(__file__).resolve().parent
ENTRY = APP / "remote-seo-checker.py"   # hyphenated -> run as a script, never importable

failures = []
checked = {"compiled": [], "imported": [], "skipped_stdlib": []}


def fail(stage, target, error):
    failures.append({"stage": stage, "target": target, "error": str(error)})


# 1) Syntax-compile every .py copied into the image (entry + all siblings).
for py in sorted(APP.glob("*.py")):
    try:
        py_compile.compile(str(py), doraise=True)
        checked["compiled"].append(py.name)
    except py_compile.PyCompileError as e:
        fail("compile", py.name, e)


# 2) Collect the entrypoint's MODULE-LEVEL import roots (direct children of the
#    module body only — imports the author put bare at top level are asserted to
#    exist; optional imports tucked inside try/except are intentionally ignored).
def module_level_import_roots(path):
    roots = set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as e:
        fail("parse", path.name, e)
        return roots
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:   # skip relative imports
                roots.add(node.module.split(".")[0])
    return roots


roots = module_level_import_roots(ENTRY)

# 3) Resolve each non-stdlib root by actually importing it inside the image.
stdlib = set(getattr(sys, "stdlib_module_names", set()))
for root in sorted(roots):
    if root in stdlib:
        checked["skipped_stdlib"].append(root)
        continue
    try:
        importlib.import_module(root)
        checked["imported"].append(root)
    except Exception as e:   # ImportError + anything raised at import time
        fail("import", root, repr(e))

report = {
    "ok": not failures,
    "app": str(APP),
    "entry": ENTRY.name,
    "checked": checked,
    "failures": failures,
}
print(json.dumps(report, indent=2))
sys.exit(0 if not failures else 1)
