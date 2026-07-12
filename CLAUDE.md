# Welcome to exec-sandbox

See @README for project overview and @Makefile for available commands for this project.

## Code search — use `seek`

Prefer `seek` over grep/ripgrep for code search. It returns BM25-ranked results with context and symbol tags.

**All filters go in ONE quoted string.** Use single quotes to prevent shell expansion.

Key patterns: `sym:Name`, `file:path`, `-file:path`, `lang:python`, `content:regex`, `type:file`, `case:yes`.

Project examples:

```sh
seek 'sym:QemuVM'                                  # find class definition
seek 'sym:Session file:src -file:test'              # scoped symbol search
seek 'content:async def.*execute lang:python -file:test'  # regex + lang filter
seek 'content:EROFS lang:shell'                     # find EROFS references in shell scripts
seek 'type:file Makefile'                           # find files by name
```

Install (if missing): `curl -sSfL https://raw.githubusercontent.com/dualeai/seek/main/install.sh | sh` — requires `universal-ctags`.

When spawning sub-agents, pass: "Use `seek 'pattern'` for code search. All filters in ONE quoted string. Never use grep/rg."

## System version pins — versions.lock

Alpine, QEMU, Rust, and kernel versions (plus their content hashes: image digests, tarball sha256s) live in `versions.lock` at the repo root — never edit it manually, refresh with `make upgrade` (runs `scripts/upgrade-versions.sh`). The lock is the ONLY supported version source: every build script and sub-make reads it directly (grep/include — never sourced, never passed via env or args). Alpine's linux-virt base kernel configs are vendored in `images/kernel/alpine-virt-<arch>.config` at upgrade time. Builds are fail-closed: missing lock, missing key, or hash mismatch aborts. Kernel bumps land as reviewable git diffs (lock + vendored configs).

## GitHub Actions — pin to commit SHAs

Pin third-party actions to full commit SHA + version comment: `uses: actions/checkout@8e8c483db84b4bee98b60c0593521ed34d9990e8 # v6.0.1`. Never use mutable tags (`@v4`, `@v6.0.1`). Resolve SHAs with `gh api repos/OWNER/REPO/git/ref/tags/TAG --jq '.object.sha'` (dereference annotated tags via `git/tags/SHA`).
