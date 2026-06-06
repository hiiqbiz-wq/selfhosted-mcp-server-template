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
  1. docker build      -> catches Dockerfile / requirements install errors
  2. in-image import   -> runs gate_import_check.py INSIDE the built image; a missing or
                          broken module the entrypoint imports fails here (the real catch)
  3. smoke (optional)  -> GET a URL, assert status (default 401 = the live gateway's "up")

Pure stdlib + subprocess; needs `docker` on PATH. Run via `uv run python deploy_gate.py`.
Exit 0 = PASS (safe to push); non-zero = FAIL (push blocked when wired as a pre-push hook).
"""
import argparse
import pathlib
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
IMAGE = "hiiq-gateway-gate:local"
CHECK = "gate_import_check.py"


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

    if args.smoke_url and not smoke(args.smoke_url, args.smoke_expect):
        print("\nGATE FAIL: smoke test", file=sys.stderr)
        _cleanup(args)
        return 1

    _cleanup(args)
    print("\nGATE PASS  (image builds + all entrypoint imports resolve)")
    return 0


def _cleanup(args):
    if not args.keep:
        run(["docker", "image", "rm", "-f", args.image], capture_output=True, text=True)


if __name__ == "__main__":
    sys.exit(main())
