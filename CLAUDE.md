# Welcome to exec-sandbox

See @README for project overview and @Makefile for available commands for this project.

## Writing — Orwell’s six rules

For prose—not code, identifiers, commands, exact quotations, or error text—follow George Orwell’s rules from “Politics and the English Language”:

1. Avoid stock metaphors, similes, and other figures of speech.
2. Prefer a short, familiar word when it is equally precise.
3. Cut every word that adds no meaning.
4. Prefer active voice when it is clearer.
5. Prefer everyday English to foreign phrases, needless scientific terms, or jargon; keep exact technical terms when precision requires them.
6. Break any rule before making the prose unclear, inaccurate, unsafe, or unnatural.

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

Alpine, QEMU, Rust, and kernel versions (plus their content hashes: image digests, tarball sha256s) live in `versions.lock` at the repo root — never edit it manually, refresh with `make upgrade` (runs `scripts/upgrade-versions.sh`). The lock is the ONLY supported version source: every build script and sub-make reads it directly (grep/include — never sourced, never passed via env or args). Upgrades apply a 7-day quarantine (uv exclude-newer equivalent): releases distributed less than 7 days ago are skipped for the previous one; each lock key carries a `# distributed <date>` comment and the header records the date of the last value-changing generation. Kernel patch bumps within an Alpine branch are exempt (they carry unannounced security fixes). Alpine's linux-virt base kernel configs are vendored in `images/kernel/alpine-virt-<arch>.config` at upgrade time. Builds are fail-closed: missing lock, missing key, or hash mismatch aborts. Kernel bumps land as reviewable git diffs (lock + vendored configs).

## GitHub Actions — pin to commit SHAs

Pin third-party actions to full commit SHA + version comment: `uses: actions/checkout@8e8c483db84b4bee98b60c0593521ed34d9990e8 # v6.0.1`. Never use mutable tags (`@v4`, `@v6.0.1`). Resolve SHAs with `gh api repos/OWNER/REPO/git/ref/tags/TAG --jq '.object.sha'` (dereference annotated tags via `git/tags/SHA`).
