# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Pre-push deploy gate for the HIIQ Edge MCP gateway (Phase 0 of the control-plane plan).

Prevents the Coolify crash-loop class where a local `py_compile` / import passes but the
Docker image is broken (e.g. a Dockerfile that COPYs modules by name and omits a new one
-> ImportError on boot -> restart:unknown, while Coolify still reports the build "finished").
See: .claude/memory/lessons/dockerfile-copy-by-name-omits-new-modules.md  (board task O).

Stages:
  0. source sentinels      -> assert deploy-keystone artifact content is present
  1. docker build          -> catches Dockerfile / requirements install errors
  2. in-image import       -> runs gate_import_check.py INSIDE the built image; a missing or
                              broken module the entrypoint imports fails here (the real catch)
  3. in-image capabilities -> imports the actual gateway entrypoint and enumerates FastMCP
                              tools/resources so a valid-but-regressed tree cannot pass
  4. session lifecycle     -> calls session_end + session_boot with fake PACOM
                              and asserts write-through + citation-ready output
  5. smoke (optional)      -> GET a URL, assert status (default 401 = the live gateway's "up")

Pure stdlib + subprocess; needs `docker` on PATH. Run via `uv run python deploy_gate.py`.
Exit 0 = PASS (safe to push); non-zero = FAIL (push blocked when wired as a pre-push hook).
"""
import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
IMAGE = "hiiq-gateway-gate:local"
CHECK = "gate_import_check.py"
CAPABILITY_CHECK = "gate_capability_check.py"
SESSION_LIFECYCLE_CHECK = "gate_session_boot_check.py"

SOURCE_SENTINELS = (
    (
        "Dockerfile",
        re.compile(r"(?m)^\s*COPY\s+\*\.py\s+\./\s*$"),
        "Dockerfile must copy every Python sibling into the image (COPY *.py ./).",
    ),
    (
        "requirements.txt",
        re.compile(r"(?m)^\s*py-key-value-aio\[postgresql\]\s*$"),
        "requirements.txt must keep the PACOM-backed OAuth client-store dependency.",
    ),
)


def run(cmd, **kw):
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, **kw)


def smoke(url, expect):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=10) as r:
            code = r.status
    except urllib.error.HTTPError as e:
        code = e.code
    except Exception as e:
        print(f"smoke: {url} unreachable: {e!r}")
        return False
    print(f"smoke: {url} -> {code} (expect {expect})")
    return code == expect


def source_sentinels_ok(context):
    ok = True
    root = pathlib.Path(context).resolve()
    for rel, pattern, message in SOURCE_SENTINELS:
        path = root / rel
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"source sentinel: {rel} unreadable: {e!r}", file=sys.stderr)
            ok = False
            continue
        if not pattern.search(text):
            print(f"source sentinel: {rel} failed - {message}", file=sys.stderr)
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser(description="HIIQ Edge gateway pre-push deploy gate")
    ap.add_argument("--context", default=str(HERE), help="docker build context (default: repo root)")
    ap.add_argument("--dockerfile", default=None, help="explicit Dockerfile (default: <context>/Dockerfile)")
    ap.add_argument("--image", default=IMAGE, help="tag for the throwaway gate image")
    ap.add_argument("--smoke-url", default=None, help="optional URL to GET after the import check")
    ap.add_argument("--smoke-expect", type=int, default=401, help="expected smoke status code")
    ap.add_argument("--keep", action="store_true", help="keep the built image (default: remove)")
    args = ap.parse_args()

    if not shutil.which("docker"):
        print("GATE FAIL: docker not on PATH (is Docker Desktop running?)", file=sys.stderr)
        return 2

    if not source_sentinels_ok(args.context):
        print("\nGATE FAIL: deploy-keystone source content missing/regressed", file=sys.stderr)
        return 1

    build = ["docker", "build", "-t", args.image]
    if args.dockerfile:
        build += ["-f", args.dockerfile]
    build += [args.context]
    if run(build).returncode != 0:
        print("\nGATE FAIL: docker build failed", file=sys.stderr)
        return 1

    chk = run(["docker", "run", "--rm", "--entrypoint", "python", args.image, f"/app/{CHECK}"],
              capture_output=True, text=True)
    sys.stdout.write(chk.stdout)
    if chk.stderr:
        sys.stderr.write(chk.stderr)
    if chk.returncode != 0:
        print("\nGATE FAIL: in-image import check failed - a module the entrypoint imports is "
              "missing or broken in the image (the v5.5 crash-loop class).", file=sys.stderr)
        _cleanup(args)
        return 1

    cap = run(["docker", "run", "--rm", "--entrypoint", "python", args.image, f"/app/{CAPABILITY_CHECK}"],
              capture_output=True, text=True)
    sys.stdout.write(cap.stdout)
    if cap.stderr:
        sys.stderr.write(cap.stderr)
    if cap.returncode != 0:
        print("\nGATE FAIL: in-image capability check failed - the built gateway no longer "
              "exposes the expected control-plane tools/resources/dependencies.", file=sys.stderr)
        _cleanup(args)
        return 1

    lifecycle = run(["docker", "run", "--rm", "--entrypoint", "python", args.image, f"/app/{SESSION_LIFECYCLE_CHECK}"],
                    capture_output=True, text=True)
    sys.stdout.write(lifecycle.stdout)
    if lifecycle.stderr:
        sys.stderr.write(lifecycle.stderr)
    if lifecycle.returncode != 0:
        print("\nGATE FAIL: session lifecycle behavior check failed - the grounding path no longer "
              "writes session_end handoffs and returns them through session_boot.", file=sys.stderr)
        _cleanup(args)
        return 1

    if args.smoke_url and not smoke(args.smoke_url, args.smoke_expect):
        print("\nGATE FAIL: smoke test", file=sys.stderr)
        _cleanup(args)
        return 1

    _cleanup(args)
    print("\nGATE PASS  (image builds + imports resolve + control-plane surface + session lifecycle contract match baseline)")
    return 0


def _cleanup(args):
    if not args.keep:
        run(["docker", "image", "rm", "-f", args.image], capture_output=True, text=True)


if __name__ == "__main__":
    sys.exit(main())
