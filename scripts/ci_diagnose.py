#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""CI diagnostic tool for GitHub Actions — failure-centric output.

Usage: uv run scripts/ci_diagnose.py [status|diagnose] [run_id]
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any, cast

YELLOW = "\033[1;33m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
SEP = "━" * 64

_GH = shutil.which("gh")

_NOISE_RE = re.compile(
    r"DeprecationWarning:|warnings summary|warnings$|pytest-of-runner|"
    r"site-packages/|^\s*$|-- Docs: https://docs\.pytest\.org|"
    r"datetime\.datetime\.utcnow|"
    r"asyncio\.(get_event_loop_policy|set_event_loop_policy|iscoroutinefunction)|"
    r"slated for removal|^tests/.*warnings$|^\s+/.*\.py:\d+:|^\[gw\d+\]"
)
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.Z-]+ ")
_TEST_SEP_RE = re.compile(r"^_+ (.+?) _+$")
_FAILED_RE = re.compile(r"^FAILED (\S+?)(?:\s+-\s+(.+))?$")
_MONITOR_RE = re.compile(r"\[MONITOR\] \d{2}:\d{2}:\d{2} ")
_JOB_RE = re.compile(r"Test / Python (\S+) / (\S+) \((.+)\)")


@dataclass
class Failure:
    title: str
    context: str = ""
    jobs: list[str] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    error_line: str = ""
    monitors: list[str] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]


async def gh(*args: str) -> str:
    """Run a gh CLI command and return stdout."""
    gh_path = _GH
    if not gh_path:
        print("Error: gh CLI is required. Install with: brew install gh", file=sys.stderr)
        sys.exit(1)
    env = os.environ.copy()
    env["GH_PAGER"] = ""
    proc = await asyncio.create_subprocess_exec(
        gh_path,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)}: {stderr.decode().strip()}")
    return stdout.decode()


async def gh_json(*args: str) -> Any:
    return json.loads(await gh(*args))


def _shorten_job(name: str) -> str:
    m = _JOB_RE.match(name)
    return f"Py{m.group(1)}/{m.group(2)}({m.group(3)})" if m else name


def _clean_tb(lines: list[str]) -> str:
    """Filter noise and join traceback lines."""
    out = [ln for ln in lines if not _NOISE_RE.search(ln)]
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def _parse_failures_section(log: str) -> dict[str, str]:
    """Extract {test_name: traceback} from the FAILURES section."""
    start = re.search(r"^=+ FAILURES =+$", log, re.MULTILINE)
    if not start:
        return {}
    end = re.search(r"^=+ short test summary", log, re.MULTILINE)
    section = log[start.end() : end.start() if end else len(log)]

    blocks: dict[str, str] = {}
    name: str | None = None
    lines: list[str] = []

    for line in section.splitlines():
        m = _TEST_SEP_RE.match(line)
        if not m:
            if name is not None:
                lines.append(line)
            continue
        if name is not None:
            blocks[name] = _clean_tb(lines)
        name = m.group(1)
        lines = []

    if name is not None:
        blocks[name] = _clean_tb(lines)
    return blocks


def _parse_short_summary(log: str) -> list[tuple[str, str]]:
    """Extract [(node_id, error_msg)] from short test summary."""
    start = re.search(r"^=+ short test summary", log, re.MULTILINE)
    if not start:
        return []
    results: list[tuple[str, str]] = []
    for line in log[start.end() :].splitlines():
        if line.startswith("="):
            break
        m = _FAILED_RE.match(line)
        if m:
            results.append((m.group(1), m.group(2) or ""))
    return results


def _tb_parts(tb_name: str) -> list[str]:
    """Split traceback name on . (class.method), preserving dots inside [params]."""
    bracket = tb_name.find("[")
    base = tb_name[:bracket] if bracket != -1 else tb_name
    parts = base.split(".")
    if bracket != -1:
        parts[-1] += tb_name[bracket:]
    return parts


def _parse_log(raw_log: str) -> tuple[dict[str, Failure], str]:
    """Parse pytest log into failures and last MONITOR snapshot."""
    log = "\n".join(_TS_RE.sub("", line) for line in raw_log.splitlines())

    # Last MONITOR line before FAILURES section
    monitor = ""
    cut = log.find("= FAILURES =")
    for line in (log[:cut] if cut != -1 else log).splitlines():
        if _MONITOR_RE.match(line):
            monitor = line

    tracebacks = _parse_failures_section(log)
    summary = _parse_short_summary(log)

    # Precompute segment splits for matching
    tb_seg: dict[str, list[str]] = {name: _tb_parts(name) for name in tracebacks}

    failures: dict[str, Failure] = {}
    seen: set[str] = set()
    for node_id, error_msg in summary:
        title = node_id.rsplit("::", maxsplit=1)[-1]
        ctx = ""
        node_parts = node_id.split("::")
        for tb_name, tb_text in tracebacks.items():
            parts = tb_seg[tb_name]
            if len(parts) > len(node_parts) or node_parts[-len(parts) :] != parts:
                continue
            ctx = tb_text
            seen.add(tb_name)
            break
        failures[node_id] = Failure(title=title, context=ctx, error_line=error_msg)

    for tb_name, tb_text in tracebacks.items():
        if tb_name not in seen:
            failures[tb_name] = Failure(title=tb_name, context=tb_text)

    return failures, monitor


async def _parse_annotations(repo: str, job_id: int) -> dict[str, Failure]:
    """Fetch check-run annotations (runner crashes, OOM)."""
    try:
        raw: Any = json.loads(await gh("api", f"repos/{repo}/check-runs/{job_id}/annotations"))
    except (RuntimeError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, list):
        return {}
    failures: dict[str, Failure] = {}
    for a in cast("list[dict[str, str]]", raw):
        msg = a.get("message", "")
        if not msg:
            continue
        failures[msg] = Failure(title=msg, context=msg, error_line=msg)
    return failures


async def _process_job(repo: str, job_id: int) -> tuple[dict[str, Failure], str]:
    """Fetch logs for one failed job. Returns ({fingerprint: Failure}, monitor_line)."""
    try:
        log = await gh("api", f"repos/{repo}/actions/jobs/{job_id}/logs")
    except RuntimeError:
        log = ""

    if log:
        failures, monitor = _parse_log(log)
    else:
        failures, monitor = {}, ""

    # Fallback: check-run annotations (runner crashes, OOM)
    if not failures:
        failures = await _parse_annotations(repo, job_id)

    if not failures:
        failures[f"unknown-{job_id}"] = Failure(
            title="Unknown failure (no logs or annotations)", error_line="Unknown failure"
        )

    return failures, monitor


def _print_results(failures: list[Failure]) -> None:
    for fail in failures:
        print(SEP)
        print(f"{YELLOW}❌ {fail.title}{RESET}  ({len(fail.jobs)} job{'s' if len(fail.jobs) != 1 else ''})")
        print(", ".join(fail.jobs))
        if fail.context:
            print()
            print(fail.context)
        if fail.monitors:
            print()
            for m in fail.monitors:
                print(f"{DIM}{m}{RESET}")
        print()

    total = sum(len(f.jobs) for f in failures)
    print(SEP)
    print(f"{BOLD}📊 Summary — {len(failures)} distinct failure(s) across {total} job(s){RESET}")
    print()
    for fail in failures:
        label = fail.error_line or fail.title
        print(f"{YELLOW}{len(fail.jobs)}x{RESET} {label}")
    print()


async def _load_run(run_id: str | None) -> tuple[str, str, list[dict[str, Any]]]:
    """Resolve run, print header + counts. Returns (repo, status, jobs)."""
    if not run_id:
        data = cast("list[dict[str, Any]]", await gh_json("run", "list", "--limit", "1", "--json", "databaseId"))
        run_id = str(data[0]["databaseId"])

    repo = cast("dict[str, str]", await gh_json("repo", "view", "--json", "nameWithOwner"))["nameWithOwner"]
    run = cast(
        "dict[str, Any]",
        await gh_json("run", "view", run_id, "--json", "headBranch,status,headSha,createdAt,jobs"),
    )

    print(f"📊 CI Run {run_id} ({run['headBranch']})")
    print(f"{DIM}🔗 https://github.com/{repo}/actions/runs/{run_id}{RESET}")
    print(f"{DIM}commit: {run['headSha'][:7]} | started: {run['createdAt']}{RESET}")
    print()

    jobs = cast("list[dict[str, Any]]", run["jobs"])
    passed = sum(1 for j in jobs if j.get("conclusion") == "success")
    failed = sum(1 for j in jobs if j.get("conclusion") == "failure")
    cancelled = sum(1 for j in jobs if j.get("conclusion") == "cancelled")
    running = sum(1 for j in jobs if j.get("status") == "in_progress")
    print(f"✅ {passed} passed | ❌ {failed} failed | 🚫 {cancelled} cancelled | 🔄 {running} running")
    print()

    return repo, str(run["status"]), jobs


async def cmd_status(run_id: str | None) -> None:
    _, _, jobs = await _load_run(run_id)
    for j in jobs:
        icon = {"success": "✅", "failure": "❌", "cancelled": "🚫"}.get(
            j.get("conclusion") or "", "🔄" if j.get("status") == "in_progress" else "⏳"
        )
        print(f"{icon} {j['name']}")


async def cmd_diagnose(run_id: str | None) -> None:
    repo, status, jobs = await _load_run(run_id)

    failed = [j for j in jobs if j.get("conclusion") == "failure"]
    if not failed:
        print("✅ All jobs passed!" if status == "completed" else "🔄 No failures yet (run still in progress)")
        return

    # Fetch all logs concurrently
    results = await asyncio.gather(
        *[_process_job(repo, int(j["databaseId"])) for j in failed],
        return_exceptions=True,
    )

    # Deduplicate failures across jobs
    failure_map: dict[str, Failure] = {}
    for job, result in zip(failed, results, strict=True):
        short = _shorten_job(str(job["name"]))

        if isinstance(result, BaseException):
            fp = f"fetch-error-{job['databaseId']}"
            failure_map.setdefault(
                fp, Failure(title=f"Could not fetch logs: {result}", error_line=str(result))
            ).jobs.append(short)
            continue

        job_failures, monitor = result
        tag = f"{short}: {monitor}" if monitor else ""

        for fp, fail in job_failures.items():
            failure_map.setdefault(fp, fail)
            if fail.context and len(fail.context) > len(failure_map[fp].context):
                failure_map[fp].context = fail.context
            failure_map[fp].jobs.append(short)
        if tag:
            for fp in job_failures:
                failure_map[fp].monitors.append(tag)

    _print_results(sorted(failure_map.values(), key=lambda f: len(f.jobs), reverse=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="CI diagnostic tool")
    parser.add_argument("command", nargs="?", default="diagnose", choices=["status", "diagnose"])
    parser.add_argument("run_id", nargs="?", default=None)
    args = parser.parse_args()

    try:
        coro = cmd_status(args.run_id) if args.command == "status" else cmd_diagnose(args.run_id)
        asyncio.run(coro)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
